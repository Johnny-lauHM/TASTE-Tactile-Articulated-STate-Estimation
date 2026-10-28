"""Demo clips of a trained single-finger flick policy: for each move of the task, play episodes until one succeeds
(env 0 is recorded), write one clip per move with an overlay (move, layer angle, cube motion) and a hold on the final
frame. One Isaac process per task (an env cannot be re-created in a process):
    python scripts/flick_reel.py --task Rubik-Flick-Ds-Sharpa-v0 --ckpt logs/rsl_rl/rubik_flick/<run>/model_700.pt --headless
Clips go to logs/demo/reel/<move>.mp4 (+ <move>_last.png); scripts/make_flick_reel.sh runs all tasks and joins the clips.
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--moves", default=None, help="subset, default: the task's flick moves")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--tries", type=int, default=8)
parser.add_argument("--out_dir", default="logs/demo/reel")
parser.add_argument("--eye", type=float, nargs=3, default=(-0.40, -0.30, 0.46))
parser.add_argument("--slow", type=int, default=4, help="every physics step is rendered and repeated this many times at 20 fps (step 25 ms -> 2x per repeat)")
parser.add_argument("--eye2", type=float, nargs=3, default=None, help="optional second camera (rendered for the same step, side by side)")
parser.add_argument("--stochastic", action="store_true")
parser.add_argument("--max_wobble_deg", type=float, default=12.0, help="prefer successes whose peak cube rotation is below this")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab.utils.math as mu  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import rubik_lab  # noqa: E402, F401
from rubik_lab.flick import FLICKS  # noqa: E402
from rubik_lab.moves import ALL_MOVES, MOVE_INDEX  # noqa: E402
from rubik_lab.tasks.bimanual_twist.env_cfg import CUBE_CENTER  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
agent = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
cfg.scene.num_envs = args.num_envs
cfg.viewer.origin_type = "env"
cfg.viewer.env_index = 0
cfg.viewer.eye = tuple(args.eye)
cfg.viewer.lookat = CUBE_CENTER
cfg.viewer.resolution = (960, 640)
env = gym.make(args.task, cfg=cfg, render_mode="rgb_array")
u = env.unwrapped
u.record = True
wenv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = OnPolicyRunner(wenv, agent.to_dict(), log_dir=None, device=agent.device)
runner.load(os.path.join(ROOT, args.ckpt))
policy = runner.get_inference_policy(device=u.device)
if args.stochastic:
    policy = lambda o: runner.alg.policy.act(o).detach()  # noqa: E731
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 24)
    small = ImageFont.truetype("DejaVuSans.ttf", 16)
except OSError:
    font = small = ImageFont.load_default()
moves = args.moves.split(",") if args.moves else list(cfg.flick_moves)
os.makedirs(os.path.join(ROOT, args.out_dir), exist_ok=True)
n, dev, T = u.num_envs, u.device, u.max_episode_length


def obs_of():
    o = wenv.get_observations()
    return o[0] if isinstance(o, tuple) else o


FACE_NAMES = {"U": "top", "D": "bottom", "F": "front", "B": "back", "R": "right", "L": "left"}


def caption(move):
    if move not in FLICKS:
        return ""
    hand, finger, face, _, _ = FLICKS[move]
    return f"{hand} {finger} pushes the {FACE_NAMES[face]} face of the {FACE_NAMES[move[0]]} layer"


def grab():
    """render the current state from the main camera and, with --eye2, from the second camera too (side by side)"""
    if args.eye2 is None:
        return env.render()
    imgs = []
    for eye in (args.eye, args.eye2):
        u.viewport_camera_controller.update_view_location(eye=tuple(eye), lookat=tuple(CUBE_CENTER))
        for _ in range(3):  # the viewport image lags the camera move by a frame or two
            u.sim.render()
        imgs.append(env.render()[..., :3])
    return np.concatenate(imgs, 1)


def overlay(img, move, k, err, rot, pos, status):
    im = Image.fromarray(img[..., :3])
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 620, 114), fill=(20, 20, 24))
    d.text((16, 12), f"{move}", fill=(255, 255, 255), font=font)
    d.text((16, 90), caption(move), fill=(150, 200, 255), font=small)
    d.text((16, 44), f"t = {k * u.step_dt:4.2f} s   layer {90 - abs(err):5.1f} / 90 deg   cube moved {rot:4.1f} deg, {pos:4.1f} mm", fill=(230, 230, 230), font=small)
    if status:
        d.text((16, 66), status, fill=(120, 255, 120) if status.startswith("DONE") else (255, 120, 120), font=small)
    return np.asarray(im)


summary = []
for move in moves:
    best = None  # (score, frames, last)
    for attempt in range(args.tries):
        u.forced_move_id = torch.full((n,), MOVE_INDEX[move], device=dev)
        env.reset()
        obs = obs_of()
        raw_prev, err_prev, rot_prev, pos_prev = grab(), 90.0, 0.0, 0.0
        frames, status, peak_rot, peak_pos = [], "", 0.0, 0.0
        for k in range(T - 1):
            with torch.no_grad():
                a = policy(obs)
            obs, _, dones, _ = wenv.step(a)
            snap = u.snapshot
            core, q, _, _ = u._cube_state()
            err = float(torch.rad2deg(snap["twist_error"][0]))
            rot = float(torch.rad2deg(mu.quat_error_magnitude(q[:1], u.cube_quat_goal[:1])))
            pos = float(torch.linalg.norm(core[0] - u.cube_center[0]) * 1000)
            if not dones[0]:
                peak_rot, peak_pos = max(peak_rot, rot), max(peak_pos, pos)
            if dones[0]:
                ok = bool(snap["primitive_done"][0])
                status = f"DONE in {(k + 1) * u.step_dt:.2f} s   (cube peak {peak_rot:.1f} deg / {peak_pos:.1f} mm)" if ok else ("DROPPED" if snap["dropped"][0] else "TIMEOUT")
                # the env auto-resets inside the terminating step: hold the last pre-reset image, not the next episode's grasp
                last = overlay(raw_prev, move, k, err_prev, rot_prev, pos_prev, status)
                frames += [last] * 30
                break
            raw_prev, err_prev, rot_prev, pos_prev = grab(), err, rot, pos
            frames += [overlay(raw_prev, move, k, err, rot, pos, status)] * args.slow
        else:
            ok, last = False, frames[-1]
        score = (1 if ok else 0, -max(peak_rot / args.max_wobble_deg, 1.0), -peak_rot)
        print(f"[reel] {move:3s} attempt {attempt + 1}: {'DONE' if ok else status or 'TIMEOUT'} after {(k + 1) * u.step_dt:.2f}s, cube peak {peak_rot:.1f} deg / {peak_pos:.1f} mm", flush=True)
        if best is None or score > best[0]:
            best = (score, frames, last, ok, peak_rot, peak_pos)
        if ok and peak_rot < args.max_wobble_deg:
            break
    _, frames, last, ok, peak_rot, peak_pos = best
    path = os.path.join(ROOT, args.out_dir, f"{move.replace(chr(39), 'p')}.mp4")
    w = imageio.get_writer(path, fps=20)
    for f in frames:
        w.append_data(f)
    w.close()
    imageio.imwrite(path.replace(".mp4", "_last.png"), last)
    summary.append((move, ok, peak_rot, peak_pos))
print("[reel] " + ", ".join(f"{m}: {'ok' if o else 'FAIL'} (peak {r:.1f} deg / {p:.1f} mm)" for m, o, r, p in summary), flush=True)
env.close()
app.close()
