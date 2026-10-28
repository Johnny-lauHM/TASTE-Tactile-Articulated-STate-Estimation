"""Hold test in flick mode: zero trick action, hold `--hold_s`; cube drift / pad forces / drops vs flick_hold_press.
    python scripts/flick_hold_test.py --task Rubik-Flick-U-Sharpa-v0 --press 0.15 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Rubik-Flick-U-Sharpa-v0")
parser.add_argument("--press", type=float, default=0.0)
parser.add_argument("--push", type=float, default=0.0, help="constant trick-finger action (all 4 joints) to load the grip")
parser.add_argument("--wrist", type=float, default=-1.0, help="wrist stiffness override (-1 = cfg, 0 = USD)")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--hold_s", type=float, default=2.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as mu  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import rubik_lab  # noqa: E402, F401

cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
cfg.scene.num_envs = args.num_envs
if args.wrist >= 0:
    cfg.wrist_stiffness = None if args.wrist == 0 else args.wrist
cfg.flick_hold_press = args.press
cfg.episode_length_s = 1e6
env = gym.make(args.task, cfg=cfg)
u = env.unwrapped
env.reset()
a = torch.full((u.num_envs, 4), args.push, device=u.device)
for _ in range(int(0.3 / u.step_dt)):
    env.step(torch.zeros_like(a))
c0, q0 = [x.clone() for x in u._cube_state()[:2]]
for _ in range(int(args.hold_s / u.step_dt)):
    env.step(a)
c1, q1 = u._cube_state()[:2]
moved = torch.linalg.norm(c1 - c0, dim=-1) * 1000
rot = torch.rad2deg(mu.quat_error_magnitude(q1, q0))
lf = u.layer_f  # (N, 2, 5, [core, top])
core = lf[..., 0]
twist = torch.rad2deg(u.signed_err.abs())
from rubik_lab.cube_contacts import LINKS  # noqa: E402

cf = u.cube_contacts.forces()  # (N, 2 bodies, 2 hands, 27)
h = u.flick_hand[u.move_id].clamp(min=0)
for hh, hn in enumerate(("L", "R")):
    on = (cf[:, 1, hh] > 0.3).float().mean(0)
    print(f"[hold] links on the moving layer ({hn}): " + " ".join(f"{LINKS[i]}={on[i]:.0%}" for i in range(len(LINKS)) if on[i] > 0.05), flush=True)
    trick = (h == hh)
    if trick.any():
        on_t = (cf[trick, 1, hh] > 0.3).float().mean(0)
        print(f"[hold]   when {hn} is the trick hand: " + " ".join(f"{LINKS[i]}={on_t[i]:.0%}" for i in range(len(LINKS)) if on_t[i] > 0.05), flush=True)
for side in ("left", "right"):
    pw, _ = u.ctrl[side].ee_pose_w()
    off = torch.linalg.norm(pw - u.scene.env_origins - u.ee_pos_tgt[side], dim=-1) * 1000
    hp = torch.linalg.norm(pw - u.scene.env_origins - u.ee_home_pos[side], dim=-1) * 1000
    k = u.hands[side].actuators["float_base"].stiffness[0, :3].tolist()
    print(f"[hold] {side} wrist: target-actual offset median {off.median():.1f} mm | moved from home median {hp.median():.1f} mm | base stiffness {k}", flush=True)
print(
    f"[hold] press {args.press:.2f} push {args.push:.2f}: moved median {moved.median():.1f} mm (90% {moved.quantile(0.9):.1f}) | rot median {rot.median():.1f} deg (90% {rot.quantile(0.9):.1f}) | "
    f"dropped {(moved > 50).float().mean():.1%} | thumb/middle core force median L {core[:, 0, 0].median():.1f}/{core[:, 0, 2].median():.1f} R {core[:, 1, 0].median():.1f}/{core[:, 1, 2].median():.1f} N | "
    f"layer err median {twist.median():.0f} deg",
    flush=True,
)
env.close()
app.close()
