"""Ground-truth check of the TacMap images and timing breakdown.

For every pad: the TacMap ray starts (map points on the elastomer, world frame) are tested analytically against the two
cube boxes (core / top). Points inside the cube = where the image must be non-zero. Compared with the TacMap image and the
cube-only contact force (force_matrix).
    python scripts/diag_tactile.py --headless
"""

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--no_image", action="store_true")
parser.add_argument("--no_points", action="store_true", help="contact sensors without contact point tracking")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as mu  # noqa: E402

import rubik_lab  # noqa: E402, F401
from rubik_lab.assets import FINGERTIPS  # noqa: E402
from rubik_lab.tasks.bimanual_twist.env_cfg import BimanualTwistEnvCfg  # noqa: E402

SIDES = ("left", "right")
cfg = BimanualTwistEnvCfg()
cfg.gravity_curriculum = False  # real gravity
cfg.scene.num_envs = args.num_envs
cfg.ee_action_mode = "absolute"
cfg.episode_length_s = 1e6
cfg.ee_init_pos_noise = 0.0
cfg.cube_drop_height = -10.0
cfg.tactile.image = not args.no_image
if args.no_points:
    cfg.tactile.contact_cfg.track_contact_points = False
env = gym.make("Rubik-Twist-Bimanual-Sharpa-v0", cfg=cfg)
u = env.unwrapped
n = u.num_envs
env.reset()
tac = u.tactile


def hold():
    acts = []
    for s in SIDES:
        lo, hi = u.finger_lower[s], u.finger_upper[s]
        acts.append(torch.cat([u.ee_pos_tgt[s], u.ee_quat_tgt[s], 2 * (u.finger_tgt[s] - lo) / (hi - lo) - 1], -1))
    return torch.cat(acts, -1)


for _ in range(20):
    env.step(hold())

def timed(fn, k=20):
    torch.cuda.synchronize(); t = time.time()
    for _ in range(k):
        fn()
    torch.cuda.synchronize(); return (time.time() - t) / k * 1000



s = cfg.cube_size
d = u.cube.data
boxes = [(u.core_id, torch.tensor([0, 0, -s / 6]), torch.tensor([s / 2, s / 2, s / 3])),
         (u.top_id, torch.tensor([0, 0, s / 3]), torch.tensor([s / 2, s / 2, s / 6]))]  # fmt: skip


def inside_depth(p):
    """max over boxes of penetration depth (>0 inside) for points p (n, R, 3)."""
    best = torch.full(p.shape[:2], -1.0, device=p.device)
    for body, c, h in boxes:
        q = d.body_quat_w[:, body].unsqueeze(1).expand(-1, p.shape[1], -1)
        local = mu.quat_apply_inverse(q, p - d.body_pos_w[:, body].unsqueeze(1)) - c.to(p.device)
        depth = (h.to(p.device) - local.abs()).min(-1).values  # >0 inside
        best = torch.max(best, depth)
    return best


if args.no_image:
    print(f"env.step (images off{', no contact points' if args.no_points else ''}): {timed(lambda: env.step(hold()), 10):7.2f} ms", flush=True)
    env.close(); app.close(); raise SystemExit
img = tac.raw_image()  # forces a TacMap update now
f_cube = tac.layer_forces().sum(-1)  # (n, 2, 5) cube-only contact force
f_net = torch.linalg.norm(tac.raw_ft()[..., :3], dim=-1)
print("pad            | map pts inside cube (analytic) | max depth mm | TacMap active px | F cube N | F net N   (env-mean)")
for h, side in enumerate(SIDES):
    for f, finger in enumerate(FINGERTIPS):
        sensor = tac.images[side][f]
        starts = sensor._ray_starts_w  # (n, R, 3)
        dep = inside_depth(starts)
        ins = (dep > 0).sum(-1).float()
        act = (img[:, h, f] > 0).flatten(1).sum(-1).float()
        print(f"{side:5s} {finger:6s}   | {ins.mean():8.1f}                     | {dep.max(-1).values.clamp(min=0).mean() * 1000:6.2f}       "
              f"| {act.mean():8.1f}         | {f_cube[:, h, f].mean():6.2f}   | {f_net[:, h, f].mean():6.2f}")  # fmt: skip

# timing: physics-only vs F/T vs images
print(f"\nraw_ft (contact sensors -> 6D)  : {timed(tac.raw_ft):7.2f} ms")
for side in SIDES:
    for t in tac.images[side]:
        t._is_outdated[:] = True
def all_images():
    for side in SIDES:
        for t in tac.images[side]:
            t._is_outdated[:] = True
    tac.raw_image()
print(f"raw_image (10 TacMap sensors)   : {timed(all_images, 5):7.2f} ms")
print(f"env.step (incl. 6 physics steps): {timed(lambda: env.step(hold()), 10):7.2f} ms", flush=True)
env.close()
app.close()
