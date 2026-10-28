"""Offline planner reach of the finger tricks for every grasp of a cache (pure torch, no Isaac): adds `flick_reach`
(N grasps x len(ALL_MOVES), degrees, -1 = not a flick) so the env can sample, per move, grasps that can be flicked.
    python scripts/flick_reach.py --cache cache/human_D2.npz [--out cache/human_D2.npz] [--moves "D,D'"]
"""

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "source/rubik_lab"))
from rubik_lab.flick import FLICKS, FlickPlanner  # noqa: E402
from rubik_lab.hand_kinematics import quat_to_mat  # noqa: E402
from rubik_lab.moves import ALL_MOVES, MOVE_INDEX  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--cache", required=True)
p.add_argument("--out", default=None, help="default: overwrite --cache")
p.add_argument("--moves", default=",".join(FLICKS))
p.add_argument("--batch", type=int, default=256)
a = p.parse_args()
dev = "cuda"
path = os.path.join(ROOT, a.cache)
d = dict(np.load(path))
N = len(d["cube_pos"])
T = lambda k, sl: torch.tensor(d[k][sl], device=dev, dtype=torch.float32)  # noqa: E731
reach = d.get("flick_reach", np.full((N, len(ALL_MOVES)), -1.0, np.float32))
pl = FlickPlanner(dev)
for m in a.moves.split(","):
    side = FLICKS[m][0]
    out = np.zeros(N, np.float32)
    for s0 in range(0, N, a.batch):
        sl = slice(s0, min(N, s0 + a.batch))
        Rc = quat_to_mat(T("cube_quat", sl))
        wp = torch.einsum("bji,bj->bi", Rc, T(f"{side}_ee_pos", sl) - T("cube_pos", sl))
        wR = Rc.transpose(1, 2) @ quat_to_mat(T(f"{side}_ee_quat", sl))
        with torch.enable_grad():
            _, _, _, ok = pl.solve(m, wp, wR, T(f"{side}_fingers", sl))
        out[sl] = pl.reach_deg(ok).cpu().numpy()
    reach[:, MOVE_INDEX[m]] = out
    print(f"[reach] {m:3s}: median {np.median(out):.0f} deg | >=40: {(out >= 40).mean():.0%} ({int((out >= 40).sum())}) | >=45: {(out >= 45).mean():.0%}", flush=True)
d["flick_reach"] = reach
np.savez(os.path.join(ROOT, a.out or a.cache), **d)
print(f"[reach] wrote {a.out or a.cache}", flush=True)
