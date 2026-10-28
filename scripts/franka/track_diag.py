"""Wrist tracking and per-skill success of a chain teacher (true cube state) with the float wrist or the FR3 arms.

Per skill: wrist position / rotation error between the commanded wrist pose (nominal + residual) and the actual one
(mean, p90, max over the skill's steps), failures (drop / violation / time-out) and successes.

    python scripts/franka/track_diag.py --headless --teacher <ckpt> [--set "arm='fr3'"] [--num_envs 64] [--steps 1200]
"""
import argparse
import ast
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--teacher", required=True)
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--steps", type=int, default=1200)
p.add_argument("--set", action="append", default=[])
p.add_argument("--zero", action="store_true", help="zero policy residuals: the nominal skills alone")
p.add_argument("--trace", type=int, default=0, help="print per-step traces of the first N violations")
p.add_argument("--mod", action="append", default=["ROLL_SHIFT=0.016", "ROLL_RING_LIFT=0.6"])
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app
import sys
import traceback

sys.excepthook = lambda t, v, b: (traceback.print_exception(t, v, b), sys.stdout.flush(), os._exit(1))
import torch  # noqa: E402

import isaaclab.utils.math as mu  # noqa: E402
from rubik_lab.solve import skills as S  # noqa: E402

for kv in a.mod:
    k, v = kv.split("=", 1)
    setattr(S, k, ast.literal_eval(v))
from rubik_lab.solve.env import SolveEnv  # noqa: E402
from rubik_lab.solve.env_cfg import SolveEnvCfg  # noqa: E402
from rubik_lab.solve.policies import load_policies  # noqa: E402

cfg = SolveEnvCfg()
cfg.scene.num_envs = a.num_envs
cfg.ready_cache = "cache/solve_ready_ud_all4_flick.npz"
cfg.episode_length_s = 20.0
cfg.chain_train = True
cfg.obs_next_skill = True
cfg.terminate_on_success = False
cfg.skills = ("hold",)
cfg.link_contacts = False
cfg.detent_torque, cfg.detent_width = 0.08, 0.2618
cfg.contact_model = "tacmap"
cfg.flick_legacy_ckpt = "logs/rsl_rl/rubik_flick/2026-10-17_16-28-11/model_1900.pt"
cfg.flick_legacy_kw = dict(latch_steps=2, latch_min_s=0.1)
for kv in a.set:
    k, v = kv.split("=", 1)
    setattr(cfg, k, ast.literal_eval(v))
env = SolveEnv(cfg)
dev, n = env.device, env.num_envs
obs, _ = env.reset()
teacher = load_policies(env, {"all": a.teacher})["all"]
for side, hand in env.hands.items():
    fi = env.finger_ids[side][:5]
    print(f"[diag] {side} finger joints {[hand.joint_names[i] for i in fi]}: stiffness {hand.data.joint_stiffness[0, fi].tolist()} damping "
          f"{hand.data.joint_damping[0, fi].tolist()} armature {hand.data.joint_armature[0, fi].tolist()} friction "
          f"{hand.data.joint_friction_coeff[0, fi].tolist()}", flush=True)
    for name, act in hand.actuators.items():
        print(f"[diag] {side} actuator {name}: {type(act).__name__} stiffness {act.stiffness[0, :5].tolist()} damping {act.damping[0, :5].tolist()}", flush=True)
reasons = {}
_gd = env._get_dones


def _get_dones():  # the failure flags before the reset clears them
    out = _gd()
    env._why = (env.dropped.clone(), env.violation.clone())
    return out


env._get_dones = _get_dones
K = len(S.SKILLS)
ep = [[] for _ in range(K)]
er = [[] for _ in range(K)]
fails, oks = torch.zeros(K, device=dev), torch.zeros(K, device=dev)
hist = []
traced = 0
for step in range(a.steps):
    with torch.inference_mode():
        act = teacher(obs["policy"])
        if a.zero:
            act = torch.zeros_like(act)
    if a.trace:
        row = [env.skill.clone(), env.t_skill.clone(), env.peak_pos.clone() * 1000, torch.rad2deg(env.peak_rot.clone())]
        for h, side in enumerate(("left", "right")):
            pw, qw = env.ctrl[side].ee_pose_w()
            row.append((pw - env.scene.env_origins - env.wrist_tgt_pos[:, h]).norm(dim=-1) * 1000)
            if hasattr(env.ctrl[side], "q_cmd"):
                jq = env.hands[side].data.joint_pos[:, env.ctrl[side].joint_ids]
                row.append(torch.rad2deg((jq - env.ctrl[side].q_cmd).abs().max(-1).values))
                row.append(env.hands[side].data.joint_vel[:, env.ctrl[side].joint_ids].abs().max(-1).values)
                lo, hi = env.ctrl[side].chain.lo, env.ctrl[side].chain.hi
                row.append(torch.rad2deg(torch.min(env.ctrl[side].q_cmd - lo, hi - env.ctrl[side].q_cmd).min(-1).values))
        hist = (hist + [row])[-30:]
    k_before = env.skill.clone()
    obs, _, term, trunc, _ = env.step(act)
    live = ~env.chain_warm
    for h, side in enumerate(("left", "right")):
        pw, qw = env.ctrl[side].ee_pose_w()
        dp = (pw - env.scene.env_origins - env.wrist_tgt_pos[:, h]).norm(dim=-1) * 1000.0
        dr = torch.rad2deg(mu.quat_error_magnitude(qw, env.wrist_tgt_quat[:, h]))
        for k in range(K):
            m = live & (env.skill == k)
            if m.any():
                ep[k].append(dp[m])
                er[k].append(dr[m])
    fails.index_add_(0, k_before[term], torch.ones(int(term.sum()), device=dev))
    for i in term.nonzero().squeeze(-1).tolist():
        r = "drop" if env._why[0][i] else ("violation" if env._why[1][i] else "timeout")
        if a.trace and r == "violation" and traced < a.trace and (not os.environ.get("TRACE_SKILL") or S.SKILLS[int(k_before[i])] == os.environ["TRACE_SKILL"]):
            traced += 1
            print(f"[trace] env {i} violation in {S.SKILLS[int(k_before[i])]}: step | skill t | cube peak pos mm, peak rot deg | "
                  "L wrist err mm, L arm q err deg, L arm max qd, L cmd limit margin deg | R ...", flush=True)
            for j, row in enumerate(hist[-30:]):
                v = [float(x[i]) for x in row]
                print(f"[trace]   {j:2d} {S.SKILLS[int(v[0])]:12s} {v[1]:4.2f} | {v[2]:5.1f} {v[3]:5.1f} | " + " ".join(f"{x:6.2f}" for x in v[4:]), flush=True)
        key = (S.SKILLS[int(k_before[i])], r)
        reasons[key] = reasons.get(key, 0) + 1
    adv = (env.skill != k_before) & ~term
    oks.index_add_(0, k_before[adv], torch.ones(int(adv.sum()), device=dev))
print(f"[diag] arm={cfg.arm} {a.num_envs} envs x {a.steps} steps", flush=True)
for k in range(K):
    if not ep[k]:
        continue
    x, y = torch.cat(ep[k]), torch.cat(er[k])
    print(f"  {S.SKILLS[k]:13s} wrist pos err mm mean {x.mean():5.1f} p90 {x.quantile(0.9):5.1f} max {x.max():6.1f} | rot deg mean "
          f"{y.mean():5.1f} p90 {y.quantile(0.9):5.1f} max {y.max():6.1f} | done {int(oks[k])} failed {int(fails[k])}", flush=True)
print("[diag] failure reasons", sorted(reasons.items(), key=lambda x: -x[1]), flush=True)
os._exit(0)
