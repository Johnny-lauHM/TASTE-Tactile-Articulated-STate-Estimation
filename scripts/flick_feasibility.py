"""Kinematic feasibility of human finger tricks from the human home grip (pure torch IK, no Isaac).

For a move (layer + direction), a finger keeps its pad on a point of the moving layer while the layer turns by theta
(the contact point rotates with the layer: no sliding). For each theta (5 deg steps) IK over that finger's joints
(optionally + a small wrist motion) must put the pad on the point with the pad facing into the surface. Reports the
largest continuously reachable theta. A fast flick only needs ~45 deg (the detent then snaps the layer to 90).
    python scripts/flick_feasibility.py [--cache cache/home_human.npz] [--n 60]
"""

import argparse
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "source/rubik_lab"))
from rubik_lab.constants import FINGER_JOINTS  # noqa: E402
from rubik_lab.hand_kinematics import HandFK, axis_angle_to_mat, quat_to_mat  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--cache", default="cache/home_human.npz")
p.add_argument("--n", type=int, default=60, help="grasps to test")
p.add_argument("--iters", type=int, default=60)
p.add_argument("--moves", default=None, help="subset, e.g. \"D,D',F,F'\"")
a = p.parse_args()
dev = "cuda"
from rubik_lab.constants import CUBE_SIZE  # noqa: E402

H = CUBE_SIZE / 2
NORM = {"U": (0, 0, 1.0), "D": (0, 0, -1.0), "F": (-1.0, 0, 0), "B": (1.0, 0, 0), "R": (0, -1.0, 0), "L": (0, 1.0, 0)}
# the human table: move -> (hand, finger); X' uses the other hand by default (mirrored trick)
TRICKS = {"U": ("right", "index"), "U'": ("left", "index"), "D": ("left", "ring"), "D'": ("right", "ring"),
          "F": ("right", "index"), "F'": ("left", "index"), "B": ("left", "index"), "B'": ("right", "index")}  # fmt: skip

d = np.load(os.path.join(ROOT, a.cache))
sel = np.random.default_rng(0).choice(len(d["cube_pos"]), size=min(a.n, len(d["cube_pos"])), replace=False)
Rc = quat_to_mat(torch.tensor(d["cube_quat"][sel], device=dev, dtype=torch.float32))
cpos = torch.tensor(d["cube_pos"][sel], device=dev, dtype=torch.float32)
fks = {s: HandFK(s, dev, pts_per_link=8) for s in ("left", "right")}
G = len(sel)


def rot(axis, th):
    return axis_angle_to_mat(torch.tensor(axis, device=dev, dtype=torch.float32).expand(G, 3) * th)


print(f"{G} home grasps; theta = continuously reachable layer rotation (median / share of grasps >= 45 deg / >= 90 deg)")
for move, (side, finger) in TRICKS.items():
    if a.moves and move not in a.moves.split(","):
        continue
    face = move[0]
    n_ax = np.array(NORM[face])
    angle = -math.pi / 2 if not move.endswith("'") else math.pi / 2  # about the outward normal
    fk = fks[side]
    wp = torch.einsum("bji,bj->bi", Rc, torch.tensor(d[f"{side}_ee_pos"][sel], device=dev, dtype=torch.float32) - cpos)
    wR = Rc.transpose(1, 2) @ quat_to_mat(torch.tensor(d[f"{side}_ee_quat"][sel], device=dev, dtype=torch.float32))
    q0 = torch.tensor(d[f"{side}_fingers"][sel], device=dev, dtype=torch.float32)
    cols = [i for i, j in enumerate(FINGER_JOINTS) if j.startswith(finger)]
    with torch.no_grad():
        pad0, _ = fk.pads(fk.forward(wp, wR, q0))[finger]
    # candidate contact faces of the moving layer: its outer face (friction drive) + the four side faces (push)
    layer_faces = [f for f in NORM if np.abs(np.dot(NORM[f], n_ax)) < 0.5] + [face]
    results = []
    for cf in layer_faces:
        cn = torch.tensor(NORM[cf], device=dev, dtype=torch.float32)
        # start point: the pad projected onto that face, clamped into the moving layer
        p0 = pad0 - ((pad0 @ cn) - (H - 0.001)).unsqueeze(-1) * cn  # 1 mm into the face
        lay = torch.tensor(n_ax, device=dev, dtype=torch.float32)
        along = p0 @ lay
        p0 = p0 + (torch.clamp(along, H / 3 + 0.003, H - 0.003) - along).unsqueeze(-1) * lay if cf != face else p0
        for wrist in (False, True):
            raw = q0[:, cols].clone().requires_grad_(True)
            dpos = torch.zeros(G, 3, device=dev, requires_grad=wrist)
            drot = torch.zeros(G, 3, device=dev, requires_grad=wrist)
            params = [raw] + ([dpos, drot] if wrist else [])
            reach = torch.zeros(G, device=dev)
            alive = torch.ones(G, dtype=torch.bool, device=dev)
            for k, th in enumerate(np.radians(np.arange(0, 95, 5))):
                Rl = rot(NORM[face], angle / abs(angle) * th)
                target = torch.einsum("bij,bj->bi", Rl, p0)
                tn = torch.einsum("bij,j->bi", Rl, cn)
                groups = [{"params": [raw], "lr": 0.03}]
                if wrist:  # metres / radians: much smaller steps than the finger joints
                    groups += [{"params": [dpos], "lr": 1e-3}, {"params": [drot], "lr": 1e-2}]
                opt = torch.optim.Adam(groups)
                for _ in range(a.iters * (3 if k == 0 else 1)):
                    q = q0.clone()
                    q[:, cols] = torch.max(torch.min(raw, fk.upper[cols]), fk.lower[cols])
                    R = axis_angle_to_mat(drot) @ wR if wrist else wR
                    c, m = fk.pads(fk.forward(wp + (dpos if wrist else 0), R, q))[finger]
                    loss = ((c - target) ** 2).sum(-1) + 1e-4 * (1 + (m * tn).sum(-1))
                    if wrist:  # small wrist motion only (+-1 cm, +-10 deg)
                        loss = loss + 10 * ((dpos.norm(dim=-1) - 0.01).clamp(min=0) ** 2) + 1e-3 * ((drot.norm(dim=-1) - math.radians(10)).clamp(min=0) ** 2)
                    opt.zero_grad()
                    loss.sum().backward()
                    opt.step()
                with torch.no_grad():
                    q = q0.clone()
                    q[:, cols] = torch.max(torch.min(raw, fk.upper[cols]), fk.lower[cols])
                    R = axis_angle_to_mat(drot) @ wR if wrist else wR
                    c, m = fk.pads(fk.forward(wp + (dpos if wrist else 0), R, q))[finger]
                    ok = (torch.linalg.norm(c - target, dim=-1) < 0.003) & ((m * tn).sum(-1) < -0.2)
                    if wrist:
                        ok &= (dpos.norm(dim=-1) < 0.012) & (drot.norm(dim=-1) < math.radians(12))
                    alive &= ok
                    reach = torch.where(alive, torch.full_like(reach, float(np.degrees(th))), reach)
            r = reach.cpu().numpy()
            results.append((cf, wrist, np.median(r), np.mean(r >= 45), np.mean(r >= 90)))
    best = max(results, key=lambda x: (x[3], x[2]))
    line = " | ".join(f"{cf}{'+wrist' if w else ''}: {med:.0f}deg {p45:.0%}/{p90:.0%}" for cf, w, med, p45, p90 in results)
    print(f"{move:3s} {side:5s} {finger:6s} -> best {best[0]}{'+wrist' if best[1] else ''} ({best[2]:.0f}deg, >=45: {best[3]:.0%})\n      {line}", flush=True)
