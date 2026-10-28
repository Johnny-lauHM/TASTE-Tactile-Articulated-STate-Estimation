"""Keep only handoff-ready start states (cube within 2 mm / 3 deg of its ready-grip pose modulo symmetry, layer within
3 deg of a quarter) of recorded pools (handoff_matrix.py --dump) and merge them.

    python scripts/solve/filter_pool.py <out.pt> <pool.pt>... [--pos 0.002 --rot 3 --layer 3]
"""
import argparse
import itertools
import math

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("pools", nargs="+")
p.add_argument("--ready_cache", default="cache/solve_ready_ud_all4_flick.npz")
p.add_argument("--pos", type=float, default=0.002)
p.add_argument("--rot", type=float, default=3.0)
p.add_argument("--layer", type=float, default=3.0)
a = p.parse_args()


def quat_to_matrix(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
                        2 * (y * z - x * w), 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).view(*q.shape[:-1], 3, 3)


group = []
for perm in itertools.permutations(range(3)):
    for sign in itertools.product((1, -1), repeat=3):
        m = np.zeros((3, 3))
        for i, j in enumerate(perm):
            m[i, j] = sign[i]
        if np.linalg.det(m) > 0:
            group.append(m)
group = torch.tensor(np.array(group), dtype=torch.float32)
ready = np.load(a.ready_cache)
cube_pos, cube_quat = torch.tensor(ready["cube_pos"]).float(), torch.tensor(ready["cube_quat"]).float()
parts = [torch.load(f) for f in a.pools]
pool = {k: torch.cat([q[k] for q in parts]) for k in parts[0]}
i = pool["ready_idx"].long()
offset = (pool["cube_pose"][:, :3] - cube_pos[i]).norm(dim=-1)
rel = quat_to_matrix(cube_quat[i]).transpose(-1, -2) @ quat_to_matrix(pool["cube_pose"][:, 3:7]) @ pool["bind_S"].transpose(-1, -2)
trace = (rel.unsqueeze(1) @ group.transpose(-1, -2).unsqueeze(0)).diagonal(dim1=-2, dim2=-1).sum(-1).amax(-1)
tilt = torch.rad2deg(torch.acos(((trace - 1) / 2).clamp(-1, 1)))
q = 0.5 * math.pi
resid = torch.rad2deg((pool["twist"] - torch.round(pool["twist"] / q) * q).abs())
keep = (offset < a.pos) & (tilt < a.rot) & (resid < a.layer)
torch.save({k: v[keep] for k, v in pool.items()}, a.out)
states, counts = torch.unique(pool["roll"][keep].long(), dim=0, return_counts=True)
print(f"[filter_pool] {int(keep.sum())}/{len(keep)} handoff-ready states -> {a.out}; roll (left, right) states "
      f"{ {tuple(s_.tolist()): int(c) for s_, c in zip(states, counts)} }")
