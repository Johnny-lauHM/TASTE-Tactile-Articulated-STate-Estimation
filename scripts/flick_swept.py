"""Swept-volume check of the finger tricks for every grasp of a cache (pure torch, no Isaac): while the layer turns, its
corners sweep a cylinder of radius H*sqrt(2) around the layer axis, so any holding link (any link but the trick finger's
and the lifted fingers') inside that cylinder, within the layer's band, jams the layer part-way (diag: thumb_DP of the
holding hand stops U at ~60 deg). Grasps that fail get `flick_reach[:, move] = -1` so the env never samples them.
    python scripts/flick_swept.py --cache cache/g70_U.npz --moves "U,U'" [--out cache/g70_U_s.npz] [--margin 0.0025]
"""

import argparse
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "source/rubik_lab"))
from rubik_lab.constants import CUBE_SIZE  # noqa: E402
from rubik_lab.flick import FLICKS  # noqa: E402
from rubik_lab.hand_kinematics import HandFK, quat_to_mat  # noqa: E402
from rubik_lab.moves import ALL_MOVES, MOVE_INDEX  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--cache", required=True)
p.add_argument("--out", default=None, help="default: overwrite --cache")
p.add_argument("--moves", default=",".join(FLICKS))
p.add_argument("--margin", type=float, default=0.0025, help="[m] clearance around the swept cylinder and the layer band")
p.add_argument("--batch", type=int, default=512)
a = p.parse_args()
dev = "cuda"
H, ROW = CUBE_SIZE / 2, CUBE_SIZE / 3
AXIS = {"U": (0, 0, 1.0), "D": (0, 0, -1.0), "F": (-1.0, 0, 0), "B": (1.0, 0, 0), "R": (0, -1.0, 0), "L": (0, 1.0, 0)}
FINGERS = ("thumb", "index", "middle", "ring", "pinky")

d = dict(np.load(os.path.join(ROOT, a.cache)))
N = len(d["cube_pos"])
# settled states (`*_act`), not the pressed targets: the target pads sit ~8 mm inside the cube
T = lambda k, sl: torch.tensor(d[k + "_act" if k + "_act" in d else k][sl], device=dev, dtype=torch.float32)  # noqa: E731
reach = d.get("flick_reach", np.full((N, len(ALL_MOVES)), -1.0, np.float32))
fks = {s: HandFK(s, dev, pts_per_link=24) for s in ("left", "right")}
r_sweep = H * math.sqrt(2) + a.margin
for m in a.moves.split(","):
    t_side, t_finger = FLICKS[m][0], FLICKS[m][1]
    ax = torch.tensor(AXIS[m[0]], device=dev)
    bad = np.zeros(N, bool)
    worst = np.zeros(N, np.float32)
    for s0 in range(0, N, a.batch):
        sl = slice(s0, min(N, s0 + a.batch))
        Rc = quat_to_mat(T("cube_quat", sl))
        hit = torch.zeros(sl.stop - sl.start, dtype=torch.bool, device=dev)
        depth = torch.zeros(sl.stop - sl.start, device=dev)
        for side in ("left", "right"):
            fk = fks[side]
            wp = torch.einsum("bji,bj->bi", Rc, T(f"{side}_ee_pos", sl) - T("cube_pos", sl))
            wR = Rc.transpose(1, 2) @ quat_to_mat(T(f"{side}_ee_quat", sl))
            fr = fk.forward(wp, wR, T(f"{side}_fingers", sl))
            pads = fk.pads(fr)
            # fingers whose pad rests on the moving layer are lifted by the env (never thumb/middle); the trick finger is free
            lifted = {f: ((pads[f][0] @ ax) > H - ROW - 0.004) & (f not in ("thumb", "middle")) for f in FINGERS}
            for name, P in fk.points(fr).items():
                finger = name.split("_")[1]  # names are <side>_<finger>_<link>
                if side == t_side and finger == t_finger:
                    continue
                al = P @ ax  # (B, P) coordinate along the layer axis
                rad = torch.linalg.norm(P - al.unsqueeze(-1) * ax, dim=-1)
                inside = (al > H - ROW - a.margin) & (al < H + a.margin) & (rad < r_sweep)
                if finger in FINGERS:
                    inside &= ~lifted[finger].unsqueeze(-1)
                hit |= inside.any(-1)
                depth = torch.max(depth, ((r_sweep - rad) * inside.float()).max(-1).values)
        bad[sl] = hit.cpu().numpy()
        worst[sl] = depth.cpu().numpy()
    mi = MOVE_INDEX[m]
    ok_before = reach[:, mi] >= 40
    reach[bad, mi] = -1.0
    print(f"[swept] {m:3s}: {bad.mean():.0%} of grasps have a holding link in the swept volume (median depth {np.median(worst[bad]) * 1000 if bad.any() else 0:.1f} mm); "
          f"usable (reach >= 40 deg) {int(ok_before.sum())} -> {int((reach[:, mi] >= 40).sum())}", flush=True)  # fmt: skip
d["flick_reach"] = reach
np.savez(os.path.join(ROOT, a.out or a.cache), **d)
