"""Time series of one flick skill from the ready grip (zero residual, legacy flick nominal): cube offset / rotation,
layer angle and per-pad normal forces (core / free layer) of both hands, averaged over envs.

    python scripts/solve/diag_flick.py --headless --skill D --num_envs 64 [--mod K=V] [--set k=v]
"""
import argparse
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--skill", default="D")
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--steps", type=int, default=60)
p.add_argument("--policy", default="", help="flick-family checkpoint for the residual (default zero)")
p.add_argument("--set", action="append", default=[])
p.add_argument("--mod", action="append", default=[])
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app
import ast
import math

import torch

from rubik_lab.solve import skills as S

for kv in a.mod:
    k, v = kv.split("=", 1)
    setattr(S, k, ast.literal_eval(v))
from rubik_lab.solve.env import SolveEnv
from rubik_lab.solve.env_cfg import SolveEnvCfg
from rubik_lab.solve.policies import load_policies

cfg = SolveEnvCfg()
cfg.scene.num_envs = a.num_envs
cfg.ready_cache = "cache/solve_ready_ud_all4_flick.npz"
cfg.skills = (a.skill,)
cfg.episode_length_s = 1e6
cfg.link_contacts = False
cfg.detent_torque, cfg.detent_width = 0.08, 0.2618
cfg.flick_legacy_ckpt = "logs/rsl_rl/rubik_flick/2026-10-17_16-28-11/model_1900.pt"
for kv in a.set:
    k, v = kv.split("=", 1)
    setattr(cfg, k, ast.literal_eval(v))
env = SolveEnv(cfg)
env.chain_mode = True
obs, _ = env.reset()
pol = load_policies(env, {"flick": a.policy})["flick"] if a.policy else None
dev = env.device
print("step  t    off_mm  rot_deg  layer_deg  viol  | pad force core/layer N: L th ix mi ri pi | R th ix mi ri pi")
for i in range(a.steps):
    act = pol(obs["policy"]) if pol else torch.zeros(env.num_envs, 56, device=dev)
    obs, *_ = env.step(act)
    lf = env.tactile.layer_forces().mean(0)  # (2, 5, 2)
    off = (env.cube.data.root_pos_w - env.scene.env_origins - env.c0).norm(dim=-1).mean() * 1000
    tw = torch.rad2deg(env.cube.data.joint_pos[:, env.twist_id]).mean()
    if i % 3 == 0:
        pads = " | ".join(" ".join(f"{lf[h, j, 0]:.1f}/{lf[h, j, 1]:.1f}" for j in range(5)) for h in range(2))
        print(f"{i:3d} {i * env.step_dt:5.2f} {off:6.1f} {math.degrees(env.rot_dev.mean()):7.1f} {tw:8.1f} {env.violation.float().mean():5.2f} | {pads}", flush=True)
        if env.cube_contacts is not None and i % 9 == 0:  # every hand link pressing on the free layer (mean N)
            from rubik_lab.cube_contacts import LINKS
            cf = env.cube_contacts.forces().mean(0)  # (2 bodies, 2 hands, 27)
            top = [f"{'LR'[h]}:{LINKS[j]} {cf[1, h, j]:.1f}" for h in range(2) for j in range(len(LINKS)) if cf[1, h, j] > 0.3]
            print("      free layer:", ", ".join(top), flush=True)
hd = env.hand_dbg.mean(0)
print("[diag] hands at the end (wrist mm, wrist deg, max finger err rad, finger speed): L", hd[0].tolist(), "R", hd[1].tolist())
print("[diag] crit pass rates:", {k: round(float(v.float().mean()), 2) for k, v in env.crit.items()})
print("[diag] finger err per joint > 0.2 rad (mean over envs), against the settled ready posture:")
from rubik_lab.assets import FINGER_JOINTS
for h, side in enumerate(("left", "right")):
    e = (env.hands[side].data.joint_pos[:, env.finger_ids[side]] - env._hand_ready(h, side, env.grip_mode_next[:, h])[3]).mean(0)
    print("   ", side, {FINGER_JOINTS[j]: round(float(e[j]), 2) for j in range(22) if abs(e[j]) > 0.2})
print(f"[diag] {a.skill}: success {env.success.float().mean():.2f} violation {env.violation.float().mean():.2f} dropped {env.dropped.float().mean():.2f}")
os._exit(0)
