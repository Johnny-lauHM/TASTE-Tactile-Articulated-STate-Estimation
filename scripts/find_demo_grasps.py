"""Pass 1 of the demo reel: run the teacher on many envs per primitive (no rendering) and store, per primitive, the
grasps it completed fastest -> logs/demo/demo_grasps.json (used by teacher_reel.py --grasps).
    python scripts/find_demo_grasps.py --headless
"""

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--moves", default="R,R',L,L',U,U',x,x'")
parser.add_argument("--per_move", type=int, default=64)
parser.add_argument("--cache", default="cache/home_grip_AB.npz")
parser.add_argument("--flick", default="", help="moves executed as finger tricks, e.g. \"U,U'\"")
parser.add_argument("--out", default="logs/demo/demo_grasps.json")
parser.add_argument("--top", type=int, default=5)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import rubik_lab  # noqa: E402, F401
from rubik_lab.moves import MOVE_INDEX  # noqa: E402
from rubik_lab.tasks.bimanual_twist.env_cfg import BimanualTwistEnvCfg  # noqa: E402
from rubik_lab.teacher import F_DONE, ScriptedTeacher  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
moves = args.moves.split(",")
n = len(moves) * args.per_move
cfg = BimanualTwistEnvCfg()
cfg.scene.num_envs = n
cfg.grasp_cache_path = args.cache
cfg.gravity_curriculum = False
cfg.wrist_prior = False
cfg.ee_init_pos_noise = 0.0
if args.flick:
    cfg.flick_moves = tuple(args.flick.split(","))
env = gym.make("Rubik-Twist-Bimanual-Sharpa-v0", cfg=cfg)
u = env.unwrapped
u.record = True
dev = u.device
mv = torch.tensor([MOVE_INDEX[m] for m in moves for _ in range(args.per_move)], device=dev)
u.forced_move_id = mv
env.reset()
teacher = ScriptedTeacher(u)
teacher.reset(torch.arange(n, device=dev))
# remember which grasp each env started from (the env picks it at reset)
start_grasp = torch.full((n,), -1, dtype=torch.long, device=dev)
orig = u._reset_idx
grasp_log = {}
done_t = np.full(n, np.inf)
succ_grasp = np.full(n, -1)
first_grasp = None
for k in range(int(u.max_episode_length) - 2):
    env.step(teacher.act())
    snap = u.snapshot
    fin = snap["primitive_done"].cpu().numpy() & np.isinf(done_t)
    done_t[fin] = snap["episode_step"].cpu().numpy()[fin] * u.step_dt
    succ_grasp[fin] = snap["grasp_idx"].cpu().numpy()[fin]  # pre-reset: the grasp of the successful episode
    gave_up = (teacher.phase == F_DONE) & ~snap["primitive_done"]
    ended = (snap["primitive_done"] | snap["dropped"] | gave_up).nonzero().flatten()
    if len(ended):
        teacher.reset(ended)
out = {}
idx_all = u.episode_grasp_idx.cpu().numpy() if hasattr(u, "episode_grasp_idx") else None
for i, m in enumerate(moves):
    sl = slice(i * args.per_move, (i + 1) * args.per_move)
    t = done_t[sl]
    order = np.argsort(t)
    good = [int(succ_grasp[sl][j]) for j in order if np.isfinite(t[j])][: args.top]
    out[m] = good
    print(f"[find] {m:3s}: {np.isfinite(t).sum()}/{args.per_move} succeeded, fastest {t[order[0]]:.1f}s -> grasps {good}", flush=True)
os.makedirs(os.path.join(ROOT, "logs/demo"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, args.out), "w"))
print(f"[find] wrote {args.out}", flush=True)
env.close()
app.close()
