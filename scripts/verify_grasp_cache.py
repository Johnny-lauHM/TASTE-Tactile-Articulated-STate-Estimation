"""Restore every grasp of a cache through the env's reset path, hold it at real gravity (residual action 0) and keep only
grasps that stay put and keep edge-only contacts. Writes the filtered cache (all fields, same format).
    python scripts/verify_grasp_cache.py --headless --cache cache/edge_synth.npz --out cache/home_grip.npz
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--cache", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--hold_s", type=float, default=2.0)
parser.add_argument("--max_move", type=float, default=0.003, help="m")
parser.add_argument("--max_rot_deg", type=float, default=3.0)
parser.add_argument("--chunk", type=int, default=1024)
parser.add_argument("--repeats", type=int, default=3, help="a grasp must pass every repeat (GPU physics is not deterministic)")
parser.add_argument("--region_left", default="edge", choices=["edge", "center", "wide"])
parser.add_argument("--region_right", default="edge", choices=["edge", "center", "wide"])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as mu  # noqa: E402

import rubik_lab  # noqa: E402, F401
from rubik_lab.cube_contacts import FINGERTIP_LINK, face_layer_mask  # noqa: E402
from rubik_lab.tasks.bimanual_twist.env_cfg import BimanualTwistEnvCfg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = dict(np.load(os.path.join(ROOT, args.cache)))
N = len(data["cube_pos"])
n = min(args.chunk, N)
cfg = BimanualTwistEnvCfg()
cfg.scene.num_envs = n
cfg.grasp_cache_path = args.cache
cfg.gravity_curriculum = False
cfg.wrist_prior = False  # hold test: no nominal wrist roll for R/L moves
cfg.episode_length_s = 1e6
cfg.ee_init_pos_noise = 0.0
cfg.cube_drop_height = -10.0
env = gym.make("Rubik-Twist-Bimanual-Sharpa-v0", cfg=cfg)
u = env.unwrapped
dev = u.device
keep = np.ones(N, bool)
stats = []
for start in [st for _ in range(args.repeats) for st in range(0, N, n)]:
    idx = np.arange(start, min(start + n, N))
    pad = np.resize(idx, n)  # fill the last chunk by repeating
    u.forced_grasp_idx = torch.tensor(pad, device=dev)
    env.reset()
    c0, q0 = [x.clone() for x in u._cube_state()[:2]]
    zero = torch.zeros(n, cfg.action_space, device=dev)
    for _ in range(int(args.hold_s / u.step_dt)):
        env.step(zero)
    c1, q1 = u._cube_state()[:2]
    moved = torch.linalg.norm(c1 - c0, dim=-1)
    rot = mu.quat_error_magnitude(q1, q0)
    touch = u.cube_contacts.forces().sum(1) > 0.05
    nonpad = (touch & ~FINGERTIP_LINK.to(dev)).any(-1).any(-1)
    cells = u.hand_cells()
    mid_col = torch.tensor([(i // 3) % 3 == 1 for i in range(27)], device=dev)  # middle column (M slice)
    allow_l = (mid_col | face_layer_mask("L", dev)) if args.region_left in ("center", "wide") else face_layer_mask("L", dev)  # holding hand: L + M
    allow_r = (mid_col | face_layer_mask("R", dev)) if args.region_right in ("center", "wide") else face_layer_mask("R", dev)  # holding hand: R + M
    outside = (cells[:, 0] & ~allow_l).any(-1) | (cells[:, 1] & ~allow_r).any(-1)
    ntips = (touch & FINGERTIP_LINK.to(dev)).sum(-1).min(-1).values
    good = (moved < args.max_move) & (rot < np.radians(args.max_rot_deg)) & ~nonpad & ~outside & (ntips >= 2)
    g = good.cpu().numpy()[: len(idx)]
    keep[idx] &= g
    stats.append((moved.cpu().numpy()[: len(idx)], nonpad.cpu().numpy()[: len(idx)], outside.cpu().numpy()[: len(idx)]))
moved = np.concatenate([s[0] for s in stats])
print(f"[verify] per restore: moved <1 mm {np.mean(moved < 0.001):.0%}, <3 mm {np.mean(moved < 0.003):.0%}, <10 mm {np.mean(moved < 0.01):.0%}, "
      f"dropped (>5 cm) {np.mean(moved > 0.05):.1%}", flush=True)  # fmt: skip
print(f"[verify] {keep.sum()}/{N} grasps pass | moved median {np.median(moved) * 1000:.1f} mm, p90 {np.percentile(moved, 90) * 1000:.1f} mm "
      f"| non-fingertip contact {np.concatenate([s[1] for s in stats]).mean():.0%} | left own column {np.concatenate([s[2] for s in stats]).mean():.0%}", flush=True)  # fmt: skip
np.save(os.path.join(ROOT, args.out.replace(".npz", "_mask.npy")), keep)
moved_all = np.concatenate([s[0] for s in stats])
np.save(os.path.join(ROOT, args.out.replace(".npz", "_moved.npy")), moved_all)
if keep.any():
    np.savez(os.path.join(ROOT, args.out), **{k: v[keep] for k, v in data.items()})
    print(f"[verify] saved {keep.sum()} grasps -> {args.out}", flush=True)
env.close()
app.close()
