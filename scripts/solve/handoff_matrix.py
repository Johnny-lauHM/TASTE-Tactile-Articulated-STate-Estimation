"""Handoff matrix: every env runs its own random valid skill chain (no resets, no recovery) until its first failure;
reports per-skill success and success of each transition prev -> cur (cur started from prev's end state).

    python scripts/solve/handoff_matrix.py --headless --num_envs 64 --length 40 --policy roll=... --policy regrip=... \
        --policy flick=... --set "flick_legacy_ckpt='...'" --mod ROLL_SHIFT=0.016 --tag base
Result: outputs/solve/handoff/<tag>.json
"""
import argparse
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--length", type=int, default=40, help="random moves per env (compiled to skills, regrips included)")
p.add_argument("--moves", default="U,U',R,R',L,L',x,x'")
p.add_argument("--ready_cache", default="cache/solve_ready_ud_all4_flick.npz")
p.add_argument("--policy", action="append", default=[], help="family=checkpoint (flick, roll, regrip; all: a chain-trained mixed checkpoint); default scripted")
p.add_argument("--skill_timeout", type=float, default=3.5)
p.add_argument("--set", action="append", default=[])
p.add_argument("--mod", action="append", default=[])
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tag", default="last")
p.add_argument("--settle", type=float, default=0.0, help="s to keep running a finished skill before starting the next")
p.add_argument("--settle_after", default="", help="comma list: settle only after these skills (default: after every skill)")
p.add_argument("--settle_fade", action="store_true", help="fade the finished skill's policy action to 0 over the settle")
p.add_argument("--seq", default="", help="run this fixed skill sequence in every env instead of random chains")
p.add_argument("--stop_alive", type=float, default=0.0, help="stop once at most this fraction of the chains is alive (the last few long chains cost a full batch each step)")
p.add_argument("--warmup", type=float, default=0.0, help="s of holding the ready grip (zero action) before the first skill")
p.add_argument("--dump", default="", help="save capture_full states at every skill start (a start pool for training)")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app
import sys as _sys, traceback as _tb
def _die(t, v, b):
    _tb.print_exception(t, v, b); _sys.stdout.flush(); os._exit(1)
_sys.excepthook = _die
import ast, json, math
import numpy as np
import torch
from rubik_lab.solve import skills as S
for kv in a.mod:
    k, v = kv.split("=", 1)
    setattr(S, k, ast.literal_eval(v))
from rubik_lab.solve.cube_logic import SkillCompiler
from rubik_lab.solve.binding import quarter_residual
from rubik_lab.solve.env import SolveEnv
from rubik_lab.solve.env_cfg import SolveEnvCfg
from rubik_lab.solve.policies import load_policies

FAMILY = {s: ("flick" if s in S.FLICK else "roll" if s in S.ROLL_DELTA else "regrip" if s in S.REGRIP_HAND else "hold") for s in S.SKILLS}

cfg = SolveEnvCfg()
cfg.scene.num_envs = a.num_envs
cfg.ready_cache = a.ready_cache
cfg.episode_length_s = 1e6
cfg.terminate_on_success = False
cfg.skills = ("hold",)
cfg.seed = a.seed
for kv in a.set:
    k, v = kv.split("=", 1)
    *path, k = k.split(".")
    obj = cfg
    for name in path:
        obj = getattr(obj, name)
    setattr(obj, k, ast.literal_eval(v))
env = SolveEnv(cfg)
env.chain_mode = True
dev, n = env.device, env.num_envs
policies = load_policies(env, dict(kv.split("=", 1) for kv in a.policy))

rng = np.random.default_rng(a.seed)
moves = a.moves.split(",")
seqs = [a.seq.split() for _ in range(n)] if a.seq else []
for _ in range(0 if a.seq else n):
    comp = SkillCompiler()
    for m in rng.choice(moves, a.length):
        comp._emit(str(m))
    seqs.append(comp.finish())

obs, _ = env.reset()
for _ in range(round(a.warmup / env.step_dt)):  # the reset grip settles (contacts) before the first skill
    obs, *_ = env.step(torch.zeros(n, 56, device=dev))
ptr = [0] * n
alive = [True] * n


def set_next(ids):  # next skill of each chain (obs_next_skill), -1 after the last
    env.next_skill[ids] = torch.tensor([S.SKILL_INDEX[seqs[i][ptr[i] + 1]] if ptr[i] + 1 < len(seqs[i]) else -1 for i in ids], device=dev)


env.start_skill(torch.arange(n, device=dev), torch.tensor([S.SKILL_INDEX[s[0]] for s in seqs], device=dev))
set_next(list(range(n)))
dumped = []
obs = env._get_observations()
skill_stat, trans_stat, fail_why = {}, {}, {}
handoff_stat = {}
start_dev = [None] * n
waiting = {}
dev_stat = {}


def hands_why(i):
    """which hand check fails: L/R + w (wrist beyond the tracking tolerance), f (finger error), v (finger speed)"""
    if bool(env.crit["hands"][i]):
        return ""
    out = []
    for h, side in enumerate("LR"):
        d = env.hand_dbg[i, h].tolist()
        tags = ("w" if d[0] > 1000 * cfg.wrist_track_tol[0] or d[1] > math.degrees(cfg.wrist_track_tol[1]) else "") + \
               ("f" if d[2] > cfg.finger_ready_tol else "") + ("v" if d[3] > cfg.finger_speed_tol else "")
        out += [side + tags] if tags else []
    return "[" + "".join(out) + "]"


def deviation(ids):
    import isaaclab.utils.math as mu
    o = env.scene.env_origins[ids]
    ri = env.ready_idx[ids]
    out = []
    for side in ("left", "right"):
        p, q = env.ctrl[side].ee_pose_w()
        hp, hq = env.ready[f"{side}_ee_pos_act"][ri], env.ready[f"{side}_ee_quat_act"][ri]
        out += [1000 * (p[ids] - o - hp).norm(dim=-1), torch.rad2deg(mu.quat_error_magnitude(q[ids], hq)),
                torch.rad2deg((env.hands[side].data.joint_pos[ids][:, env.finger_ids[side]] - env.ready_fingers_act[side][ri]).abs().max(-1).values)]
    core = env.cube.data.body_pos_w[ids, env.core_id] - o
    out.append(1000 * (core - env.c0[ids]).norm(dim=-1))
    out.append((1000 * (core - env.c0[ids])).T)
    out.append(torch.rad2deg(mu.quat_error_magnitude(env._logical_quat()[ids], env.ready["cube_quat"][ri])))
    out.append(torch.rad2deg(quarter_residual(env.cube.data.joint_pos[ids, env.twist_id]).abs()))
    out.append(10 * env.cube.data.body_ang_vel_w[ids, env.core_id].norm(dim=-1))
    out.append(torch.rad2deg(mu.axis_angle_from_quat(env._cube_deviation()[1][ids])).T)
    out.append(env.tactile.ft[ids][..., :3].norm(dim=-1).flatten(1).T)  # fingertip force [N]: L thumb..pinky, R thumb..pinky
    out = [x.unsqueeze(-1) if x.dim() == 1 else x.T for x in out]
    return torch.cat(out, -1).cpu()


start_dev = list(deviation(torch.arange(n, device=dev)))


fail_detail = []  # every failure: skill, prev, time in skill, why, cube offset (env frame, mm), rotation off the plan (axis-angle, deg)


def record(i, ok, why=""):
    cur = seqs[i][ptr[i]]
    if not ok:
        import isaaclab.utils.math as mu
        off = 1000 * (env.cube.data.root_pos_w[i] - env.scene.env_origins[i] - env.c0[i])
        dq = mu.quat_mul(env._logical_quat()[i : i + 1], mu.quat_inv(env.cube_ref[i : i + 1]))
        aa = torch.rad2deg(mu.axis_angle_from_quat(dq))[0]
        fail_detail.append(dict(skill=cur, prev=seqs[i][ptr[i] - 1] if ptr[i] else "start", t=round(float(env.t_skill[i]), 3), why=why,
                                peak_mm=round(1000 * float(env.peak_pos[i]), 1), peak_deg=round(float(torch.rad2deg(env.peak_rot[i])), 1),
                                off_mm=[round(float(x), 1) for x in off], rot_aa_deg=[round(float(x), 1) for x in aa],
                                roll=env.roll[i].round().tolist()))
    prev = seqs[i][ptr[i] - 1] if ptr[i] else "start"
    if start_dev[i] is not None:
        dev_stat.setdefault(f"{prev} -> {cur}", {}).setdefault("ok" if ok else "fail", []).append(start_dev[i].tolist())
    for d, key in ((skill_stat, cur), (trans_stat, f"{prev} -> {cur}")):
        d.setdefault(key, [0, 0])[0] += int(ok)
        d[key][1] += 1
    if not ok:
        prog = float(env.cube.data.joint_pos[i, env.twist_id] / env.layer_target[i]) if float(env.layer_target[i]) != 0 else 0.0
        why = f"{why} p{min(max(round(prog, 1), -1.0), 1.0)}" if cur in S.FLICK else why
        fail_why.setdefault(cur, {}).setdefault(why, 0)
        fail_why[cur][why] += 1


step = 0
while sum(alive) > a.stop_alive * n:
    with torch.no_grad():
        act = torch.zeros(n, 56, device=dev)
        for fam, pol in policies.items():
            if fam == "all":
                act = pol(obs["policy"])
                continue
            m = torch.tensor([FAMILY[S.SKILLS[k]] == fam for k in env.skill.tolist()], device=dev)
            if m.any():
                act[m] = pol(obs["policy"])[m]
        if a.settle_fade and waiting:  # residual of the finished skill fades out linearly while it settles
            n_settle = max(1, round(a.settle / env.step_dt))
            ids = list(waiting)
            fade = torch.tensor([max(waiting[i] - 1, 0) / n_settle for i in ids], device=dev).unsqueeze(-1)
            act[ids] = act[ids] * fade
        obs, _, _, _, _ = env.step(act)
    step += 1
    nxt_ids, nxt_sk = [], []
    for i in range(n):
        if not alive[i]:
            continue
        if i in waiting:  # settling after a success (the finished skill's end pose is held)
            waiting[i] -= 1
            if waiting[i] <= 0:
                del waiting[i]
                nxt_ids.append(i); nxt_sk.append(S.SKILL_INDEX[seqs[i][ptr[i]]])
            continue
        if env.dropped[i] or env.broken[i] or env.violation[i] or env.t_skill[i] > a.skill_timeout:
            why = "dropped" if env.dropped[i] or env.broken[i] else "violation" if env.violation[i] else \
                "timeout:" + ",".join(c for c, v in env.crit.items() if not bool(v[i])) + hands_why(i)
            record(i, False, why)
            alive[i] = False
        elif env.success[i]:
            record(i, True)
            k_ = S.SKILLS[env.skill[i].item()]
            hs = handoff_stat.setdefault(k_, [0, 0, [], []])  # handoff-ready at the success, n, offset mm, tilt deg
            hs[0] += int(env.crit["handoff"][i]); hs[1] += 1
            hs[2].append(1000 * float(env.handoff_pos[i])); hs[3].append(float(torch.rad2deg(env.handoff_rot[i])))
            ptr[i] += 1
            if ptr[i] >= len(seqs[i]):
                alive[i] = False
            elif a.settle > 0 and (not a.settle_after or k_ in a.settle_after.split(",")):
                waiting[i] = round(a.settle / env.step_dt)
            else:
                nxt_ids.append(i); nxt_sk.append(S.SKILL_INDEX[seqs[i][ptr[i]]])
    if nxt_ids and a.dump:
        dumped.append({k: v.cpu() for k, v in env.capture_full(torch.tensor(nxt_ids, device=dev)).items()})
    if nxt_ids:
        for i, dv in zip(nxt_ids, deviation(torch.tensor(nxt_ids, device=dev))):
            start_dev[i] = dv
    if nxt_ids:
        env.start_skill(torch.tensor(nxt_ids, device=dev), torch.tensor(nxt_sk, device=dev))
        set_next(nxt_ids)
    if step % 500 == 0:
        print(f"[handoff] t={step * env.step_dt:.0f}s alive {sum(alive)} skills done med {int(np.median(ptr))}", flush=True)

rate = lambda v: f"{v[0]}/{v[1]} = {v[0] / v[1]:.0%}"  # noqa: E731
print(f"[handoff] {a.tag}: chains complete {sum(p_ >= len(s) for p_, s in zip(ptr, seqs))}/{n}, skills done med {int(np.median(ptr))}, "
      f"mean len {np.mean([len(s) for s in seqs]):.0f}", flush=True)
for k, v in sorted(skill_stat.items(), key=lambda kv: kv[1][0] / kv[1][1]):
    print(f"[handoff] skill {k:14s} {rate(v):16s} fails {fail_why.get(k, {})}", flush=True)
print(f"[handoff] chain length (skills done before the first failure): mean {np.mean(ptr):.1f}, median {np.median(ptr):.0f}", flush=True)
mv = [sum(1 for s_ in s[:p_] if not s_.startswith("regrip")) for p_, s in zip(ptr, seqs)]  # moves (face turns, x) done
print(f"[handoff] moves done before the first failure (regrips not counted): mean {np.mean(mv):.1f}, median {np.median(mv):.0f}, "
      f"regrips per move {sum(len(s) for s in seqs) / max(1, sum(sum(1 for s_ in s if not s_.startswith('regrip')) for s in seqs)) - 1:.2f}", flush=True)
for k, v in sorted(handoff_stat.items()):
    print(f"[handoff] end state {k:14s} handoff-ready {v[0]}/{v[1]} = {v[0] / v[1]:.0%}   cube offset mm p50/p90 "
          f"{np.percentile(v[2], 50):.1f}/{np.percentile(v[2], 90):.1f}   tilt deg p50/p90 {np.percentile(v[3], 50):.1f}/{np.percentile(v[3], 90):.1f}", flush=True)
for k, v in sorted(trans_stat.items(), key=lambda kv: kv[1][0] / kv[1][1])[:15]:
    print(f"[handoff] trans {k:30s} {rate(v)}", flush=True)
if a.dump and dumped:
    pool = {k: torch.cat([d[k] for d in dumped]) for k in dumped[0]}
    torch.save(pool, a.dump)
    print(f"[handoff] dumped {len(pool['roll'])} start states -> {a.dump}", flush=True)
print("[handoff] start deviation (mean): L wrist mm, L wrist deg, L finger max deg, R wrist mm, R wrist deg, R finger max deg, core mm, core dx dy dz mm, cube rot deg, layer resid deg, 10 x core angvel, cube dev axis-angle x y z deg, tip force N L(th,ix,mi,ri,pi) R(..); fails: max layer progress", flush=True)
for k, v in sorted(dev_stat.items()):
    for r, rows in v.items():
        print(f"[handoff] dev {k:28s} {r:4s} n={len(rows):3d} " + " ".join(f"{x:6.1f}" for x in np.mean(rows, 0)), flush=True)
os.makedirs("outputs/solve/handoff", exist_ok=True)
json.dump(dict(args=vars(a), dev=dev_stat, skill=skill_stat, handoff={k: v[:2] for k, v in handoff_stat.items()}, trans=trans_stat, fail_why=fail_why, done=ptr, lens=[len(s) for s in seqs], seqs=seqs, fail_detail=fail_detail),
          open(f"outputs/solve/handoff/{a.tag}.json", "w"), indent=1, default=str)
os._exit(0)
