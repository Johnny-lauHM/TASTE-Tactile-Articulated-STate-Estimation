"""Scripted EE-pose demo of one R/L primitive from the home grip: the own-side hand rolls about its palm normal.

Starts from the grasp cache and prints EE tracking, layer angle, the completion label and per-layer tactile contacts.
This is a scripted sanity check of the sim (no regrip afterwards), not a policy.
    python scripts/demo_ee_pose.py --move R                       # GUI
    python scripts/demo_ee_pose.py --move L\\' --headless --enable_cameras --save_frames
"""

import argparse
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--move", default="R", choices=["R", "R'", "R2", "L", "L'", "L2"])
parser.add_argument("--move_s", type=float, default=1.5)
parser.add_argument("--save_frames", action="store_true", help="write PNGs to logs/demo_frames (needs --enable_cameras)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402

import rubik_lab  # noqa: E402, F401
from rubik_lab.moves import FACE_NORMALS, FACE_TWIST_HAND, move_angle, move_face  # noqa: E402
from rubik_lab.tasks.bimanual_twist.env_cfg import CUBE_CENTER, BimanualTwistEnvCfg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIDES = ("left", "right")

cfg = BimanualTwistEnvCfg()
cfg.gravity_curriculum = False  # real gravity
cfg.scene.num_envs = args.num_envs
cfg.ee_action_mode = "absolute"
cfg.moves = (args.move,)
cfg.episode_length_s = 1e6
cfg.ee_init_pos_noise = 0.0
cfg.cube_init_pos_noise = 0.0
cfg.viewer.origin_type = "env"
cfg.viewer.env_index = 0
cfg.viewer.eye = (-0.30, -0.25, 0.42)  # front-right, sees the thumbs
cfg.viewer.lookat = CUBE_CENTER
env = gym.make("Rubik-Twist-Bimanual-Sharpa-v0", cfg=cfg, render_mode="rgb_array" if args.save_frames else None)
u = env.unwrapped
dev, n = u.device, u.num_envs
env.reset()
center = torch.tensor(CUBE_CENTER, device=dev)
frame_dir = os.path.join(ROOT, "logs/demo_frames")
os.makedirs(frame_dir, exist_ok=True)
step = 0


def step_env():
    global step
    acts = []
    for side in SIDES:
        lo, hi = u.finger_lower[side], u.finger_upper[side]
        acts.append(torch.cat([u.ee_pos_tgt[side], u.ee_quat_tgt[side], 2 * (u.finger_tgt[side] - lo) / (hi - lo) - 1], -1))
    env.step(torch.cat(acts, -1))
    if args.save_frames and step % 3 == 0:
        import imageio.v2 as imageio

        imageio.imwrite(os.path.join(frame_dir, f"{step:04d}.png"), env.render())
    step += 1


def report(tag):
    core, _, twist, _ = u._cube_state()
    on = (u.extras["tactile_layer_force"] > cfg.tactile_contact_threshold).sum(2)[0].tolist()  # per hand: [core, layer]
    err = []
    for side in SIDES:
        p, q = u.ctrl[side].ee_pose_w()
        dp = torch.linalg.norm(p - u.scene.env_origins - u.ee_pos_tgt[side], dim=-1)[0].item()
        dr = math.degrees(math_utils.quat_error_magnitude(q, u.ee_quat_tgt[side])[0].item())
        err.append(f"{side} {dp * 1000:.1f}mm/{dr:.1f}deg")
    print(f"  [{tag:>7s}] layer={math.degrees(twist[0].item()):6.1f}deg done={bool(u.move_done[0])} "
          f"cube_z={core[0, 2].item():.3f} tips L[core,layer]={on[0]} R[core,layer]={on[1]}  EE err: {', '.join(err)}", flush=True)  # fmt: skip


for _ in range(int(0.5 / u.step_dt)):  # settle on the cached grasp
    step_env()
report("settled")

side = FACE_TWIST_HAND[move_face(args.move)]
axis = torch.tensor(FACE_NORMALS[move_face(args.move)], device=dev).expand(n, 3)
angle = move_angle(args.move)
p0, q0 = u.ee_pos_tgt[side].clone(), u.ee_quat_tgt[side].clone()
print(f"[move {args.move}] {side} hand rolls {math.degrees(angle):+.0f} deg about the {move_face(args.move)} normal", flush=True)
steps, hold = int(args.move_s / u.step_dt), int(1.0 / u.step_dt)
for i in range(steps + hold):
    s = min(1.0, (i + 1) / steps)
    dq = math_utils.quat_from_angle_axis(torch.full((n,), angle * s, device=dev), axis)
    u.ee_quat_tgt[side] = math_utils.quat_mul(dq, q0)
    u.ee_pos_tgt[side] = center + math_utils.quat_apply(dq, p0 - center)  # rotate about the cube's face axis
    step_env()
    if i in (steps // 2, steps - 1, steps + hold - 1):
        report(f"{100 * s:.0f}%" if i < steps else "+1s")

env.close()
app.close()
