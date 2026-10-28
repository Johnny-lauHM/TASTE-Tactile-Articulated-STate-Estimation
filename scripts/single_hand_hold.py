"""Can one hand hold the cube alone? Restore grasps, let one hand release (open + lift off), hold `--hold_s` with the
other, and measure the cube motion. Pass = moved < 5 mm and rotated < 5 deg.
    python scripts/single_hand_hold.py --headless --cache cache/home_grip_teach.npz [--torsion 0.005]
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--cache", default="cache/home_grip_teach.npz")
parser.add_argument("--hold_s", type=float, default=2.0)
parser.add_argument("--torsion", type=float, default=None, help="override the hands' torsional patch radius [m]")
parser.add_argument("--copies", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as mu  # noqa: E402

import rubik_lab  # noqa: E402, F401
from rubik_lab import assets  # noqa: E402
from rubik_lab.constants import FINGER_JOINTS  # noqa: E402
from rubik_lab.tasks.bimanual_twist.env_cfg import BimanualTwistEnvCfg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if args.torsion is not None:
    assets.CONTACT_MODELS["tacmap"]["torsional_patch_radius"] = args.torsion
N = len(np.load(os.path.join(ROOT, args.cache))["cube_pos"])
grasps = np.tile(np.arange(N), 2 * args.copies)
released = np.repeat([0, 1], N * args.copies)  # which hand lets go
n = len(grasps)
cfg = BimanualTwistEnvCfg()
cfg.scene.num_envs = n
cfg.grasp_cache_path = args.cache
cfg.gravity_curriculum = False
cfg.wrist_prior = False
cfg.ee_init_pos_noise = 0.0
cfg.cube_drop_height = -10.0
cfg.episode_length_s = 1e6
env = gym.make("Rubik-Twist-Bimanual-Sharpa-v0", cfg=cfg)
u = env.unwrapped
dev = u.device
u.forced_grasp_idx = torch.tensor(grasps, device=dev)
env.reset()
flex = torch.zeros(len(FINGER_JOINTS), device=dev)
flex[[i for i, j in enumerate(FINGER_JOINTS) if j.split("_")[0] in ("thumb", "index", "middle") and ("FE" in j or j.endswith(("PIP", "DIP", "IP")))]] = 1.0  # fmt: skip
rel = torch.tensor(released, device=dev)
P = u.per_hand


def act(release):
    a = torch.zeros(n, cfg.action_space, device=dev)
    if release:
        for h, side in enumerate(("left", "right")):
            me = (rel == h).float().unsqueeze(-1)
            palm_n = mu.quat_apply(u.ee_quat_tgt[side], torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3))
            a[:, h * P : h * P + 3] = me * (-palm_n * 0.012) / cfg.residual_pos_range
            a[:, h * P + 6 : (h + 1) * P] = me * (-0.7 * flex / cfg.residual_finger_range)
    return a.clamp(-1, 1)


for _ in range(int(0.3 / u.step_dt)):
    env.step(act(False))
c0, q0 = [x.clone() for x in u._cube_state()[:2]]
for _ in range(int(args.hold_s / u.step_dt)):
    env.step(act(True))
c1, q1 = u._cube_state()[:2]
moved = torch.linalg.norm(c1 - c0, dim=-1).cpu().numpy()
rot = np.degrees(mu.quat_error_magnitude(q1, q0).cpu().numpy())
ok = (moved < 0.005) & (rot < 5.0)
for h, side in enumerate(("left", "right")):
    holder = "right" if side == "left" else "left"
    m = released == h
    print(f"[hold] {holder:5s} hand alone ({side} released): pass {ok[m].mean():.0%} | moved median {np.median(moved[m]) * 1000:.1f} mm | "
          f"rot median {np.median(rot[m]):.1f} deg | dropped (>5 cm) {np.mean(moved[m] > 0.05):.0%}", flush=True)  # fmt: skip
env.close()
app.close()
