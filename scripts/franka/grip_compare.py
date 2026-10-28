"""Fingertip / wrist positions relative to the cube right after a ready-grip reset (no stepping), float vs FR3."""
import argparse
import ast
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--set", action="append", default=[])
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app
import torch  # noqa: E402

import isaaclab.utils.math as mu  # noqa: E402
from rubik_lab.solve.env import SolveEnv  # noqa: E402
from rubik_lab.solve.env_cfg import SolveEnvCfg  # noqa: E402

cfg = SolveEnvCfg()
cfg.scene.num_envs = 2
cfg.ready_cache = "cache/solve_ready_ud_all4_flick.npz"
cfg.ready_sequential = True
cfg.skills = ("hold",)
cfg.contact_model = "tacmap"
cfg.link_contacts = False
for kv in a.set:
    k, v = kv.split("=", 1)
    setattr(cfg, k, ast.literal_eval(v))
env = SolveEnv(cfg)
env.reset()
for phase in ("reset", "after 1 step", "after 40 steps"):
    if phase != "reset":
        for _ in range(1 if phase == "after 1 step" else 39):
            env.step(torch.zeros(env.num_envs, 56, device=env.device))
    d = env.cube.data
    cp, cq = d.root_pos_w[0], d.root_quat_w[0]
    print(f"[grip] arm={cfg.arm} {phase}: cube {((cp - env.scene.env_origins[0]) * 1000).round().tolist()} mm", flush=True)
    for side, hand in env.hands.items():
        names = [f"{side}_hand_wrist"] + [f"{side}_{f}_DP" for f in ("thumb", "index", "middle", "ring", "pinky")]
        for nm in names:
            i = hand.find_bodies(nm)[0][0]
            rel = mu.quat_apply(mu.quat_inv(cq), hand.data.body_pos_w[0, i] - cp) * 1000
            print(f"[grip]   {nm:22s} {rel.round().tolist()}", flush=True)
os._exit(0)
