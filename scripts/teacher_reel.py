"""Demo reel of the scripted teacher: every primitive once (env 0 recorded, retried until it succeeds), with an overlay
(primitive, phase, error, outcome) and the 10 delivered Wave SE tactile images, concatenated into one video plus a
contact sheet of key frames.
    python scripts/teacher_reel.py --headless --enable_cameras [--moves "R,R',L,L',U,U',x,x'"]
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--moves", default="R,R',L,L',U,U',x,x'")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--tries", type=int, default=4)
parser.add_argument("--max_steps", type=int, default=440)
parser.add_argument("--out", default="logs/demo/teacher_reel.mp4")
parser.add_argument("--grasps", default="logs/demo/demo_grasps.json", help="per-primitive good grasps from find_demo_grasps.py")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import rubik_lab  # noqa: E402, F401
from rubik_lab.moves import MOVE_INDEX  # noqa: E402
from rubik_lab.tasks.bimanual_twist.env_cfg import CUBE_CENTER, BimanualTwistEnvCfg  # noqa: E402
from rubik_lab.teacher import PHASES, ScriptedTeacher  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = BimanualTwistEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.gravity_curriculum = False
cfg.grasp_cache_path = "cache/home_grip_AB.npz"
cfg.wrist_prior = False
cfg.ee_init_pos_noise = 0.0
cfg.viewer.origin_type = "env"
cfg.viewer.env_index = 0
cfg.viewer.eye = (-0.40, -0.30, 0.46)
cfg.viewer.lookat = CUBE_CENTER
cfg.viewer.resolution = (960, 640)
env = gym.make("Rubik-Twist-Bimanual-Sharpa-v0", cfg=cfg, render_mode="rgb_array")
u = env.unwrapped
u.record = True
dev, n = u.device, u.num_envs
teacher = ScriptedTeacher(u)
magma = colormaps["magma"]
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 22)
    small = ImageFont.truetype("DejaVuSans.ttf", 15)
except OSError:
    font = small = ImageFont.load_default()
NAMES = {"R": "R  (right wrist turns the R layer)", "R'": "R'", "L": "L  (left wrist turns the L layer)", "L'": "L'",
         "U": "U  (edge hand moves up a row, orbits the vertical axis)", "U'": "U'",
         "x": "x  (whole-cube rotation, grip state A <-> B)", "x'": "x'"}  # fmt: skip


def tactile_strip(width):
    img = u.tactile.image[0].cpu().numpy()  # (2, 5, 60, 60)
    tiles = [(magma(img[h, f] / 255.0)[..., :3] * 255).astype(np.uint8) for h in range(2) for f in range(5)]
    strip = np.concatenate(tiles, 1)  # 60 x 600
    return np.asarray(Image.fromarray(strip).resize((width, int(60 * width / 600)), Image.NEAREST))


def frame(move, k, status):
    img = Image.fromarray(env.render()[..., :3])
    ImageDraw.Draw(img).text((16, 12), move, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def card(text, shape):
    img = Image.new("RGB", (shape[1], shape[0]), (20, 20, 24))
    d = ImageDraw.Draw(img)
    d.text((40, shape[0] // 2 - 20), text, fill=(255, 255, 255), font=font)
    return np.asarray(img)


import json

good = json.load(open(os.path.join(ROOT, args.grasps))) if os.path.exists(os.path.join(ROOT, args.grasps)) else {}
writer = imageio.get_writer(os.path.join(ROOT, args.out), fps=20)
sheet = []
summary = []
for move in args.moves.split(","):
    best, ok = None, False
    for attempt in range(args.tries):
        u.forced_move_id = torch.full((n,), MOVE_INDEX[move], device=dev)
        if good.get(move):
            u.forced_grasp_idx = torch.full((n,), good[move][attempt % len(good[move])], device=dev)
        env.reset()
        teacher.reset(torch.arange(n, device=dev))
        frames, status, keyf = [], "", None
        for k in range(args.max_steps):
            env.step(teacher.act())
            snap = u.snapshot
            if k % 2 == 0:
                f = frame(move, k, status)
                frames.append(f)
                if PHASES[int(teacher.phase[0])] == "release" and keyf is None:
                    keyf = f
            if snap["primitive_done"][0] or snap["dropped"][0] or snap["episode_step"][0] >= u.max_episode_length - 1:
                status = "DONE" if snap["primitive_done"][0] else ("DROPPED" if snap["dropped"][0] else "TIMEOUT")
                frames += [frame(move, k, status)] * 15
                break
        ok = status == "DONE"
        best = (frames, keyf if keyf is not None else frames[len(frames) // 2], frames[-1])
        print(f"[reel] {move:3s} attempt {attempt + 1}: {status or 'TIMEOUT'} after {k * u.step_dt:.1f}s", flush=True)
        if ok:
            break
    summary.append((move, ok, attempt + 1))
    frames, key, last = best
    for _ in range(20):
        writer.append_data(card(f"{move}   (attempt {attempt + 1}, {'success' if ok else 'failed'})", frames[0].shape))
    for f in frames:
        writer.append_data(f)
    sheet.append(np.concatenate([key, last], 1))
writer.close()
sheet_img = np.concatenate([np.asarray(Image.fromarray(s).resize((s.shape[1] // 2, s.shape[0] // 2))) for s in sheet], 0)
imageio.imwrite(os.path.join(ROOT, args.out.replace(".mp4", "_sheet.png")), sheet_img)
print("[reel] " + ", ".join(f"{m}: {'ok' if o else 'FAIL'} (try {t})" for m, o, t in summary), flush=True)
print(f"[reel] video -> {args.out}", flush=True)
env.close()
app.close()
