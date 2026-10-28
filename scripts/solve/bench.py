"""Throughput of SolveEnv steps (zero actions) and where the time goes (physics vs. python / torch around it).

    python scripts/solve/bench.py --headless --num_envs 8192 --skills "x,x',regrip_left,regrip_right,R,R',L,L'" [--set k=v]
"""
import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=4096)
p.add_argument("--skills", default="U,U'")
p.add_argument("--steps", type=int, default=200)
p.add_argument("--set", action="append", default=[])
p.add_argument("--no_tactile", action="store_true")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app
import ast
import os
import time

import torch

from rubik_lab.solve.env import SolveEnv
from rubik_lab.solve.env_cfg import SolveEnvCfg

cfg = SolveEnvCfg()
cfg.scene.num_envs = a.num_envs
cfg.skills = tuple(a.skills.split(","))
cfg.ready_cache = "cache/solve_ready_ud_all4_flick.npz"
for kv in a.set:
    k, v = kv.split("=", 1)
    *path, last = k.split(".")
    tgt = cfg
    for name in path:
        tgt = getattr(tgt, name)
    setattr(tgt, last, ast.literal_eval(v))
env = SolveEnv(cfg)
if a.no_tactile:
    env.tactile.step = lambda: None
env.reset()
act = torch.zeros(a.num_envs, 56, device=env.device)
t_sim = [0.0]
orig = env.sim.step


def timed(*args, **kw):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    r = orig(*args, **kw)
    torch.cuda.synchronize()
    t_sim[0] += time.perf_counter() - t0
    return r


for i in range(20):
    env.step(act)
env.sim.step = timed
torch.cuda.synchronize()
t0 = time.perf_counter()
for i in range(a.steps):
    env.step(act)
torch.cuda.synchronize()
dt = time.perf_counter() - t0
print(f"[bench] envs {a.num_envs} sets {a.set} no_tactile {a.no_tactile}: {a.steps * a.num_envs / dt:.0f} steps/s, "
      f"physics {100 * t_sim[0] / dt:.0f}% of the time, {1000 * dt / a.steps:.1f} ms/step", flush=True)
os._exit(0)
