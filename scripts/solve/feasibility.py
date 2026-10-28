"""P0 feasibility probes with the scripted nominal skills (no learning).

    python scripts/solve/feasibility.py --headless --probe hold|roll|regrip|chain|friction
Prints per-step summaries of env 0..N and aggregate success; saves a JSON report and optional frames.
"""
import argparse, json, os, time
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--probe", default="hold")
p.add_argument("--num_envs", type=int, default=16)
p.add_argument("--physics_hz", type=int, default=240)
p.add_argument("--decimation", type=int, default=6)
p.add_argument("--seconds", type=float, default=2.0)
p.add_argument("--skills", default="")
p.add_argument("--out", default="outputs/solve/feasibility")
p.add_argument("--frames", action="store_true")
p.add_argument("--cache", default="")
p.add_argument("--index", type=int, default=-1)
p.add_argument("--tag", default="")
p.add_argument("--save", default="")
p.add_argument("--set", action="append", default=[])
p.add_argument("--press", default="", help="comma list of extra flexion (action units) of the HOLDING hands, one per env group")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if a.frames: a.enable_cameras = True
app = AppLauncher(a).app
import sys as _sys, traceback as _tb, os as _os
def _die(t, v, b):
    _tb.print_exception(t, v, b); _sys.stdout.flush(); _sys.stderr.flush(); _os._exit(1)
_sys.excepthook = _die  # the simulation app hangs on close after an exception
import torch, numpy as np
import isaaclab.utils.math as mu
from rubik_lab.solve.env import SolveEnv, SIDES
from rubik_lab.solve.env_cfg import SolveEnvCfg
from rubik_lab.solve import skills as S

cfg = SolveEnvCfg(); cfg.scene.num_envs = a.num_envs; cfg.physics_hz = a.physics_hz; cfg.decimation = a.decimation
cfg.episode_length_s = 100.0; cfg.ready_index = a.index; cfg.ready_sequential = a.probe in ("screen", "prepare", "margins", "cells")
if a.cache: cfg.ready_cache = a.cache
import ast
for kv in a.set:
    k, v = kv.split("=", 1); setattr(cfg, k, ast.literal_eval(v))
cfg.terminate_on_success = False
if a.frames:
    from rubik_lab.tasks.bimanual_twist.env_cfg import CUBE_CENTER
    cfg.viewer.origin_type = "env"; cfg.viewer.env_index = 0; cfg.viewer.eye = (-0.30, -0.28, 0.42); cfg.viewer.lookat = (0.0, 0.0, 0.3); cfg.viewer.resolution = (640, 480)
env = SolveEnv(cfg, render_mode="rgb_array" if a.frames else None)
env.chain_mode = True
env.reset()
dev = env.device; n = env.num_envs
zero = torch.zeros(n, 56, device=dev)
os.makedirs(a.out, exist_ok=True)
frames = []

def stat():
    d = env.cube.data
    pos = (d.root_pos_w - env.scene.env_origins - env.c0).norm(dim=-1) * 1000
    tw = torch.rad2deg(d.joint_pos[:, env.twist_id])
    lq = env._logical_quat()
    rot = torch.rad2deg(mu.quat_error_magnitude(lq, env.cube_goal))
    pads = (env.tactile.layer_forces().sum(-1) > 0.05).sum(-1)
    return dict(pos_mm=pos.mean().item(), pos_mm_max=pos.max().item(), twist=tw[:4].tolist(), rot_goal_deg=rot.mean().item(),
                pads=pads.float().mean(0).tolist(), dropped=env.dropped.float().mean().item(), success=env.success.float().mean().item(),
                layer_err=torch.rad2deg(env.layer_err).mean().item(), peak_rot=torch.rad2deg(env.peak_rot).mean().item(), finger_err=env.finger_err.mean().item())

press_vals = [float(x) for x in a.press.split(",")] if a.press else [0.0]
def make_action(skill):
    act = torch.zeros(n, 56, device=dev)
    fl = env.flex_all
    for i in range(n):
        v = press_vals[i % len(press_vals)]
        for h in range(2):
            moving = (env.k_roll[S.SKILL_INDEX[skill], h] != 0) or (env.k_regrip[S.SKILL_INDEX[skill]] == h)
            if not moving or skill in ("x", "x'"):
                act[i, h * 28 + 6 : h * 28 + 28] = v * fl
    return act

def run(skill, seconds):
    global zero
    zero = make_action(skill)
    ids = torch.arange(n, device=dev)
    env.start_skill(ids, torch.full((n,), S.SKILL_INDEX[skill], device=dev))
    steps = int(seconds / env.step_dt)
    log = []
    for i in range(steps):
        env.step(zero)
        if a.frames and i % 2 == 0:
            frames.append(env.render())
        if i % 10 == 9 or i == steps - 1:
            s = stat(); s["t"] = round((i + 1) * env.step_dt, 3); log.append(s)
    print(skill, json.dumps(log[-1]), flush=True)
    print("   criteria:", {c: round(v.float().mean().item(), 2) for c, v in env.crit.items()}, flush=True)
    if len(press_vals) > 1:
        tw = torch.rad2deg(env.cube.data.joint_pos[:, env.twist_id]).cpu().numpy()
        pr = torch.rad2deg(env.peak_rot).cpu().numpy(); pp = env.peak_pos.cpu().numpy() * 1000
        for g, v in enumerate(press_vals):
            m = np.arange(n) % len(press_vals) == g
            print(f"   press {v}: twist med {np.median(tw[m]):.1f} frac(|twist+90|<10) {np.mean(np.abs(tw[m]+90)<10):.2f} peak_rot med {np.median(pr[m]):.1f} peak_pos med {np.median(pp[m]):.1f}")
    return log

report = {}
t0 = time.time()
if a.probe == "hold":
    report["hold"] = run("hold", a.seconds)
elif a.probe == "contacts":
    # which hand links push on which cube body during a skill (bodies follow the skill's binding: top = free slab)
    from rubik_lab.cube_contacts import LINKS
    for sk in a.skills.split(","):
        act_c = make_action(sk)
        ids = torch.arange(n, device=dev)
        env.start_skill(ids, torch.full((n,), S.SKILL_INDEX[sk], device=dev))
        acc = torch.zeros(2, 2, 27, device=dev); first = torch.full((2, 2, 27), -1.0, device=dev)
        for i in range(int(a.seconds / env.step_dt)):
            env.step(act_c)
            f = env.cube_contacts.forces()  # (N, 2 bodies, 2 hands, 27)
            hot = (f > 0.2).float().mean(0)
            acc = torch.max(acc, hot)
            newly = (hot > 0.25) & (first < 0)
            first[newly] = (i + 1) * env.step_dt
        print("skill", sk)
        for b, bn in enumerate(["core", "top"]):
            for h, hn in enumerate(["left", "right"]):
                items = [(LINKS[j], round(acc[b, h, j].item(), 2), round(first[b, h, j].item(), 3)) for j in range(27) if acc[b, h, j] > 0.05]
                print(f"  {hn} on {bn}: {items}")
elif a.probe == "margins":
    # per candidate (sequential): pad contact positions in the cube frame at hold, then the outcome of each listed skill
    from rubik_lab.cube_contacts import LINKS
    run("hold", 0.3)
    lq = env._logical_quat(); cp = env.cube.data.root_pos_w
    marg = torch.full((n, 2), 1.0, device=dev); cnt = torch.zeros(n, 2, device=dev)
    for b in ("core", "top"):
        dd = env.cube_contacts.sensors[b].data
        f = torch.linalg.norm(dd.force_matrix_w[:, 0], dim=-1)  # (N, 54)
        pts = dd.contact_pos_w[:, 0]  # (N, 54, 3)
        loc = mu.quat_apply_inverse(lq.unsqueeze(1).expand(-1, 54, -1).reshape(-1, 4), (pts - cp.unsqueeze(1)).reshape(-1, 3)).view(n, 54, 3)
        for h in range(2):
            sl = slice(h * 27, h * 27 + 27)
            on = (f[:, sl] > 0.05) & torch.isfinite(loc[:, sl]).all(-1)
            sgn = 1.0 if h == 0 else -1.0  # left hand lives at +y, right at -y
            m = sgn * torch.nan_to_num(loc[:, sl, 1], nan=1.0) - env.cfg.cube_size / 6  # distance inside the own column
            m = torch.where(on, m, torch.ones_like(m))
            marg[:, h] = torch.min(marg[:, h], m.min(-1).values)
            cnt[:, h] += on.sum(-1)
    out = {"margin_mm": (marg * 1000).tolist()}
    for sk in a.skills.split(","):
        run(sk, a.seconds)
        tw = torch.rad2deg(env.cube.data.joint_pos[:, env.twist_id])
        out[sk] = {"twist": tw.tolist(), "peak_rot": torch.rad2deg(env.peak_rot).tolist(), "peak_pos": (env.peak_pos * 1000).tolist()}
        env.reset()
        run("hold", 0.3)
    report = out
elif a.probe == "trace":
    hd = env.hands["right"]
    print("joint names", hd.joint_names[:6])
    print("vel limits", hd.data.joint_vel_limits[0, :6].tolist(), "effort", hd.data.joint_effort_limits[0, :6].tolist())
    print("stiff", hd.data.joint_stiffness[0, :6].tolist(), "damp", hd.data.joint_damping[0, :6].tolist(), "armature", hd.data.joint_armature[0, :6].tolist())
    run("hold", 0.3)
    for sk in a.skills.split(","):
        act_c = make_action(sk)
        ids = torch.arange(n, device=dev)
        env.start_skill(ids, torch.full((n,), S.SKILL_INDEX[sk], device=dev))
        lq0 = env._logical_quat().clone(); p0 = env.cube.data.root_pos_w.clone()
        for i in range(int(a.seconds / env.step_dt)):
            env.step(act_c)
            rows = []
            for h, side in enumerate(SIDES):
                pw, qw = env.ctrl[side].ee_pose_w()
                pe = ((pw - env.scene.env_origins) - env.wrist_tgt_pos[:, h]).norm(dim=-1) * 1000
                qe = torch.rad2deg(mu.quat_error_magnitude(qw, env.wrist_tgt_quat[:, h]))
                rows.append((pe[0].item(), qe[0].item()))
            dq = mu.quat_mul(env._logical_quat(), mu.quat_inv(lq0))
            aa = torch.rad2deg(mu.axis_angle_from_quat(dq))[0]
            dp = (env.cube.data.root_pos_w - p0)[0] * 1000
            tw = torch.rad2deg(env.cube.data.joint_pos[0, env.twist_id]).item()
            print(f"t={((i+1)*env.step_dt):.3f} roll={env.roll[0].tolist()} wristL err {rows[0][0]:5.1f}mm {rows[0][1]:5.1f}deg wristR err {rows[1][0]:5.1f}mm {rows[1][1]:5.1f}deg twist {tw:7.1f} core rotvec {aa.tolist()} cube dp {dp.tolist()}")
elif a.probe == "cells":
    run("hold", 0.3)
    cells = env.cube_contacts.cells(env.cube.data.root_pos_w, env._logical_quat(), env.cfg.cube_size).cpu().numpy()
    iz = np.arange(27) % 3
    for h in range(2):
        print(["left", "right"][h], "touch D row", cells[:, h][:, iz == 0].any(1).mean(), "touch U row", cells[:, h][:, iz == 2].any(1).mean())
    both = (~cells[:, 0][:, iz == 0].any(1)) & (~cells[:, 1][:, iz == 0].any(1))
    print("no D-row contact (both hands)", both.sum(), "of", len(both))
    report = {"cells": cells.tolist()}
elif a.probe == "chain":
    for sk in a.skills.split(","):
        report.setdefault(sk, []).append(run(sk, a.seconds))
elif a.probe == "screen":
    # per candidate: drift after a two-hand hold, then after each single-hand regrip (in place, no roll)
    cfg_res = {}
    def snap():
        d = env.cube.data
        return (d.root_pos_w - env.scene.env_origins).clone(), env._logical_quat().clone()
    p0, q0 = snap()
    out = {}
    for sk, sec in (("hold", 1.0), ("regrip_right", 1.2), ("regrip_left", 1.2), ("hold", 0.5)):
        run(sk, sec)
        p1, q1 = snap()
        out.setdefault("pos_mm", []).append(((p1 - p0).norm(dim=-1) * 1000).tolist())
        out.setdefault("rot_deg", []).append(torch.rad2deg(mu.quat_error_magnitude(q1, q0)).tolist())
        out.setdefault("pads", []).append((env.tactile.layer_forces().sum(-1) > 0.05).sum(-1).tolist())
        out.setdefault("dropped", []).append(env.dropped.tolist())
    report = out
elif a.probe == "prepare":
    # settle every candidate under the ready targets and save the settled states as a new ready cache
    run("hold", a.seconds)
    st = {k: v.cpu().numpy() for k, v in env.capture_state().items()}
    np.savez(a.save, **st)
    print("saved", a.save, len(st["cube_pos"]))
elif a.probe == "friction":
    # free top layer in the ready grip: torque steps, layer angle / velocity response (implicit joint friction)
    taus = torch.tensor([0.0, 0.01, 0.02, 0.025, 0.028, 0.032, 0.04, 0.06, 0.1, -0.04], device=dev)
    env.extra_twist_effort[:] = taus[torch.arange(n, device=dev) % len(taus)]
    ids = torch.arange(n, device=dev)
    env.start_skill(ids, torch.full((n,), S.SKILL_INDEX["hold"], device=dev))
    angs, vels = [], []
    for i in range(int(0.5 / env.step_dt)):
        env.step(zero)
        angs.append(env.cube.data.joint_pos[:, env.twist_id].clone()); vels.append(env.cube.data.joint_vel[:, env.twist_id].clone())
    env.extra_twist_effort[:] = 0.0
    for i in range(int(0.5 / env.step_dt)):
        env.step(zero)
        angs.append(env.cube.data.joint_pos[:, env.twist_id].clone()); vels.append(env.cube.data.joint_vel[:, env.twist_id].clone())
    A = torch.stack(angs, 1)[: len(taus)]; V = torch.stack(vels, 1)[: len(taus)]
    k = A.shape[1] // 2
    for j in range(len(taus)):
        sgn = torch.sign(V[j, k:]); flips = int(((sgn[1:] * sgn[:-1]) < 0).sum())
        print(f"tau {taus[j]:+.3f}: angle@0.5s {torch.rad2deg(A[j, k-1]):+8.3f} deg, vel@0.5s {V[j, k-1]:+.4f}, after release: final vel {V[j, -1]:+.5f}, drift {torch.rad2deg(A[j, -1] - A[j, k-1]):+.4f} deg, vel sign flips {flips}")
    report = {"taus": taus.tolist(), "angle": A.tolist(), "vel": V.tolist()}
print("wall", time.time() - t0)
json.dump(report, open(os.path.join(a.out, f"{a.probe}{a.tag}.json"), "w"))
if a.frames and frames:
    import imageio.v2 as imageio
    imageio.mimsave(os.path.join(a.out, f"{a.probe}{a.tag}.mp4"), frames, fps=20)
os._exit(0)
