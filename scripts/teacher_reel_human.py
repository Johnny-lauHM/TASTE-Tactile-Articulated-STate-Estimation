"""Demo reel of the human-style teacher: each move once (env 0 recorded, retried until it succeeds), with only the move
name as overlay. Per-move grasp caches (U/U': human_U, D/D': human_D, R/L: home_grip_AB) and per-move finger close-up
cameras; concatenated into one video plus a contact sheet (one key frame per move).
    python scripts/teacher_reel_human.py --headless --enable_cameras [--moves "U,U',D,D',R,L"] [--tries 4]
"""

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--moves", default="U,U',D,D',R,L")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--tries", type=int, default=4)
parser.add_argument("--grasps", default="logs/demo/human_grasps.json", help="per-move curated grasp indices (find_demo_grasps.py)")
parser.add_argument("--max_steps", type=int, default=400)
parser.add_argument("--out", default="logs/demo/human_reel.mp4")
parser.add_argument("--single", action="store_true", help="internal: all moves share one grasp cache (one Isaac env)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

import subprocess  # noqa: E402
import sys  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLICK_MOVES = ("U", "U'", "D", "D'")
# move -> (grasp cache, camera eye offset from the cube centre [m]): finger close-ups
SETUP = {
    "U": ("cache/human_U.npz", (-0.20, -0.14, 0.20)),
    "U'": ("cache/human_U.npz", (-0.20, -0.14, 0.20)),
    "D": ("cache/human_D2.npz", (0.26, 0.03, -0.14)),
    "D'": ("cache/human_D2.npz", (0.26, 0.03, -0.14)),
}
for m in ("R", "R'", "L", "L'"):
    SETUP[m] = ("cache/home_grip_AB.npz", (-0.42, -0.25, 0.42))

if not args.single:
    # a second gym.make in one Isaac process hangs after env.close(): one subprocess per grasp cache, then concatenate
    import imageio.v2 as imageio
    import numpy as np

    moves = args.moves.split(",")
    caches = []
    for m in moves:
        if SETUP[m][0] not in caches:
            caches.append(SETUP[m][0])
    parts = []
    for i, cache in enumerate(caches):
        group = ",".join(m for m in moves if SETUP[m][0] == cache)
        part = args.out.replace(".mp4", f"_part{i}.mp4")
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "--single", "--headless", "--enable_cameras", "--moves", group,
               "--out", part, "--tries", str(args.tries), "--num_envs", str(args.num_envs), "--max_steps", str(args.max_steps)]
        print(f"[reel] {group} -> {part}", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        parts.append(part)
    out = os.path.join(ROOT, args.out)
    writer = imageio.get_writer(out, fps=20)
    sheets = []
    for part in parts:
        for f in imageio.get_reader(os.path.join(ROOT, part)):
            writer.append_data(f)
        sheets.append(imageio.imread(os.path.join(ROOT, part.replace(".mp4", "_sheet.png"))))
    writer.close()
    imageio.imwrite(out.replace(".mp4", "_sheet.png"), np.concatenate(sheets, 0))
    for part in parts:
        print(open(os.path.join(ROOT, part.replace(".mp4", "_summary.txt"))).read(), end="", flush=True)
    print(f"[reel] video -> {args.out}", flush=True)
    sys.exit(0)

app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import rubik_lab  # noqa: E402, F401
from rubik_lab.moves import MOVE_INDEX  # noqa: E402
from rubik_lab.tasks.bimanual_twist.env_cfg import CUBE_CENTER, BimanualTwistEnvCfg  # noqa: E402
from rubik_lab.teacher import F_DONE, ScriptedTeacher  # noqa: E402

try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 40)
except OSError:
    font = ImageFont.load_default()


def frame(img, move):
    img = Image.fromarray(img[..., :3])
    ImageDraw.Draw(img).text((20, 14), move, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def card(move, shape):
    img = Image.new("RGB", (shape[1], shape[0]), (20, 20, 24))
    d = ImageDraw.Draw(img)
    w = d.textlength(move, font=font)
    d.text(((shape[1] - w) // 2, shape[0] // 2 - 24), move, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def make_env(cache, eye):
    cfg = BimanualTwistEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.gravity_curriculum = False
    cfg.grasp_cache_path = cache
    cfg.flick_moves = FLICK_MOVES
    cfg.wrist_prior = False
    cfg.ee_init_pos_noise = 0.0
    cfg.viewer.origin_type = "env"
    cfg.viewer.env_index = 0
    cfg.viewer.eye = tuple(float(c) + float(o) for c, o in zip(CUBE_CENTER, eye))
    cfg.viewer.lookat = CUBE_CENTER
    cfg.viewer.resolution = (960, 640)
    env = gym.make("Rubik-Twist-Bimanual-Sharpa-v0", cfg=cfg, render_mode="rgb_array")
    env.unwrapped.record = True
    return env


def set_eye(env, eye):
    vc = getattr(env.unwrapped, "viewport_camera_controller", None)
    if vc is None:
        return
    e = [float(c) + float(o) for c, o in zip(CUBE_CENTER, eye)]
    vc.update_view_location(eye=e, lookat=list(CUBE_CENTER))


moves = args.moves.split(",")
writer = imageio.get_writer(os.path.join(ROOT, args.out), fps=20)
sheet, summary = [], []
# one env per cache, in the order the moves first use it
caches = []
for m in moves:
    if SETUP[m][0] not in caches:
        caches.append(SETUP[m][0])
for cache in caches:
    group = [m for m in moves if SETUP[m][0] == cache]
    good = json.load(open(os.path.join(ROOT, args.grasps))) if os.path.exists(os.path.join(ROOT, args.grasps)) else {}
    env = make_env(cache, SETUP[group[0]][1])
    u = env.unwrapped
    dev, n = u.device, u.num_envs
    teacher = ScriptedTeacher(u)
    for move in group:
        set_eye(env, SETUP[move][1])
        best, ok, attempt = None, False, 0
        for attempt in range(args.tries):
            u.forced_move_id = torch.full((n,), MOVE_INDEX[move], device=dev)
            if good.get(move):  # curated grasps (scripts/find_demo_grasps.py), indices into this move's cache
                u.forced_grasp_idx = torch.full((n,), good[move][attempt % len(good[move])], device=dev)
            else:
                u.forced_grasp_idx = None
            env.reset()
            teacher.reset(torch.arange(n, device=dev))
            frames, status, keyf = [], "", None
            for k in range(args.max_steps):
                env.step(teacher.act())
                snap = u.snapshot
                if k % 2 == 0:
                    f = frame(env.render(), move)
                    frames.append(f)
                    if keyf is None and abs(float(u.signed_err[0])) < np.radians(60):  # layer under way
                        keyf = f
                done = bool(snap["primitive_done"][0])
                gave_up = int(teacher.phase[0]) == F_DONE and not done
                if done or snap["dropped"][0] or gave_up or snap["episode_step"][0] >= u.max_episode_length - 1:
                    status = "DONE" if done else ("DROPPED" if snap["dropped"][0] else "TIMEOUT")
                    frames += [frame(env.render(), move)] * 12
                    break
            ok = status == "DONE"
            dur = k * u.step_dt
            best = (frames, keyf if keyf is not None else frames[len(frames) // 2])
            print(f"[reel] {move:3s} attempt {attempt + 1}: {status or 'TIMEOUT'} after {dur:.1f}s", flush=True)
            if ok:
                break
        summary.append((move, ok, attempt + 1, dur))
        frames, key = best
        for _ in range(20):
            writer.append_data(card(move, frames[0].shape))
        for f in frames:
            writer.append_data(f)
        sheet.append(key)
    env.close()
writer.close()
sheet_img = np.concatenate([np.asarray(Image.fromarray(s).resize((s.shape[1] // 2, s.shape[0] // 2))) for s in sheet], 0)
imageio.imwrite(os.path.join(ROOT, args.out.replace(".mp4", "_sheet.png")), sheet_img)
lines = "".join(f"[reel] {m:3s}: {'ok' if o else 'FAIL'} (attempt {t}, {d:.1f}s)\n" for m, o, t, d in summary)
print(lines, end="", flush=True)
open(os.path.join(ROOT, args.out.replace(".mp4", "_summary.txt")), "w").write(lines)
app.close()
