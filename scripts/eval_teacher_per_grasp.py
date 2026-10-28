"""Per-grasp, per-primitive success of the scripted teacher on a grasp cache; writes `teach_rate` (N grasps x 8 primitives,
NaN = not evaluated) into a copy of the cache. The env then samples, per primitive, grasps with rate >= cfg threshold.
    python scripts/eval_teacher_per_grasp.py --headless --cache cache/home_grip_AB.npz --out cache/home_grip_AB_teach.npz
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--cache", default="cache/home_grip_AB.npz")
parser.add_argument("--out", default="cache/home_grip_AB_teach.npz")
parser.add_argument("--copies", type=int, default=3)
parser.add_argument("--max_grasps", type=int, default=400, help="per primitive (random subset)")
parser.add_argument("--max_steps", type=int, default=440)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import rubik_lab  # noqa: E402, F401
from rubik_lab.moves import MOVE_INDEX, PRIMITIVES  # noqa: E402
from rubik_lab.tasks.bimanual_twist.env_cfg import BimanualTwistEnvCfg  # noqa: E402
from rubik_lab.teacher import ScriptedTeacher  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = dict(np.load(os.path.join(ROOT, args.cache)))
N = len(data["cube_pos"])
rng = np.random.default_rng(0)

# usable grasps per primitive come from the env's own sampling mask
cfg = BimanualTwistEnvCfg()
cfg.scene.num_envs = 16
cfg.grasp_cache_path = args.cache
probe = gym.make("Rubik-Twist-Bimanual-Sharpa-v0", cfg=cfg)
mask = probe.unwrapped._grasp_move_mask.cpu().numpy()
probe.close()
pairs = []
for m in PRIMITIVES:
    g = np.flatnonzero(mask[MOVE_INDEX[m]] > 0)
    if len(g) > args.max_grasps:
        g = rng.choice(g, args.max_grasps, replace=False)
    pairs += [(gi, MOVE_INDEX[m]) for gi in g for _ in range(args.copies)]
grasps = np.array([p[0] for p in pairs])
moves = np.array([p[1] for p in pairs])
n = len(pairs)
print(f"[eval] {n} episodes ({args.copies} copies) over {len(PRIMITIVES)} primitives", flush=True)

cfg = BimanualTwistEnvCfg()
cfg.scene.num_envs = n
cfg.grasp_cache_path = args.cache
cfg.gravity_curriculum = False
cfg.wrist_prior = False
cfg.ee_init_pos_noise = 0.0
env = gym.make("Rubik-Twist-Bimanual-Sharpa-v0", cfg=cfg)
u = env.unwrapped
u.record = True
dev = u.device
u.forced_grasp_idx = torch.tensor(grasps, device=dev)
u.forced_move_id = torch.tensor(moves, device=dev)
env.reset()
teacher = ScriptedTeacher(u)
teacher.reset(torch.arange(n, device=dev))
outcome = np.full(n, -1)
for k in range(args.max_steps):
    env.step(teacher.act())
    snap = u.snapshot
    ended = (snap["primitive_done"] | snap["dropped"]).cpu().numpy() & (outcome < 0)
    outcome[ended] = snap["primitive_done"].cpu().numpy()[ended].astype(int)
    if (outcome >= 0).all():
        break
outcome[outcome < 0] = 0
rate = np.full((N, len(PRIMITIVES)), np.nan, np.float32)
for pi, m in enumerate(PRIMITIVES):
    sel = moves == MOVE_INDEX[m]
    for gi in np.unique(grasps[sel]):
        rate[gi, pi] = outcome[sel & (grasps == gi)].mean()
    r = rate[:, pi][~np.isnan(rate[:, pi])]
    print(f"[eval] {m:3s}: success {outcome[sel].mean():5.1%} over {sel.sum()} episodes | grasps with rate >=0.5: {(r >= 0.5).sum()}/{len(r)}, "
          f">=0.67: {(r >= 0.67).sum()}", flush=True)  # fmt: skip
data["teach_rate"] = rate
np.savez(os.path.join(ROOT, args.out), **data)
print(f"[eval] wrote {args.out}", flush=True)
env.close()
app.close()
