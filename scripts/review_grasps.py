"""Render grasps from a cache for review: each grasp is restored, held 1.5 s at real gravity (residual action 0) and
rendered from the front-right and from above, annotated with the touching links and cubie cells.
    python scripts/review_grasps.py --headless --enable_cameras --cache cache/edge_test2.npz --num 8
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--cache", required=True)
parser.add_argument("--num", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import rubik_lab  # noqa: E402, F401
from rubik_lab.cube_contacts import LINKS  # noqa: E402
from rubik_lab.tasks.bimanual_twist.env_cfg import CUBE_CENTER, BimanualTwistEnvCfg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = np.load(os.path.join(ROOT, args.cache))
N = len(data["cube_pos"])
pick = np.random.default_rng(args.seed).choice(N, size=min(args.num, N), replace=False)
k = len(pick)

cfg = BimanualTwistEnvCfg()
cfg.scene.num_envs = k
cfg.scene.env_spacing = 1.5
cfg.grasp_cache_path = args.cache
cfg.gravity_curriculum = False
cfg.wrist_prior = False  # hold test: no nominal wrist roll for R/L moves
cfg.episode_length_s = 1e6
cfg.ee_init_pos_noise = 0.0
cfg.moves = ("R",)
cfg.viewer.resolution = (640, 480)
cfg.viewer.origin_type = "env"
env = gym.make("Rubik-Twist-Bimanual-Sharpa-v0", cfg=cfg, render_mode="rgb_array")
u = env.unwrapped
u.forced_grasp_idx = torch.tensor(pick, device=u.device)
env.reset()
c0 = u._cube_state()[0].clone()
zero = torch.zeros(k, cfg.action_space, device=u.device)
for _ in range(int(1.5 / u.step_dt)):
    env.step(zero)
moved = torch.linalg.norm(u._cube_state()[0] - c0, dim=-1).cpu().numpy()
touch = (u.cube_contacts.forces().sum(1) > 0.05).cpu().numpy()  # (k, 2, 27)
cells = u.hand_cells().cpu().numpy()


def cell_str(c):
    out = []
    for i in np.flatnonzero(c):
        ix, iy, iz = i // 9, (i // 3) % 3, i % 3
        out.append("FmB"[ix] + "RmL"[iy] + "DmU"[iz])
    return ",".join(out)


views = [((-0.42, -0.10, 0.40), "front (F face, thumbs)"), ((0.40, -0.25, 0.42), "back-right (B face, fingers)"), ((0.0, 0.0, 0.68), "top")]
cc = u.viewport_camera_controller
for _ in range(20):  # renderer warm-up (the first frames are black)
    u.sim.render()
fig, axes = plt.subplots(k, 3, figsize=(13, 3.3 * k))
axes = np.atleast_2d(axes)
for r in range(k):
    cc.set_view_env_index(r)
    for c, (eye, name) in enumerate(views):
        cc.update_view_location(eye=eye, lookat=CUBE_CENTER)
        for _ in range(3):
            u.sim.render()
        axes[r, c].imshow(env.render())
        axes[r, c].axis("off")
        axes[r, c].set_title(name, fontsize=8)
    lt = [", ".join(LINKS[j].replace("elastomer", "pad") for j in np.flatnonzero(touch[r, h])) for h in range(2)]
    axes[r, 0].text(0, -0.02, f"#{pick[r]}  cube moved {moved[r] * 1000:.1f} mm in 1.5 s\nL: {lt[0]}\n   cells {cell_str(cells[r, 0])}\n"
                    f"R: {lt[1]}\n   cells {cell_str(cells[r, 1])}", transform=axes[r, 0].transAxes, fontsize=6.5, va="top")  # fmt: skip
fig.tight_layout()
out = args.out or os.path.join(ROOT, f"logs/grasp_review/{os.path.basename(args.cache)[:-4]}.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=90)
print(f"[review] {k} grasps, cube moved (mm): {np.round(moved * 1000, 1).tolist()}  -> {out}", flush=True)
env.close()
app.close()
