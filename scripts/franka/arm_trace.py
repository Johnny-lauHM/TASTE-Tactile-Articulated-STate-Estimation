"""What the FR3 arms do before a drop: from an executor --record (FR3 env), per env the arm joints' distance to their
limits, the Jacobian's smallest singular value, the largest per-frame joint jump and the drift from the ready
configuration, over the run (the last seconds before the drop in detail).

    python scripts/franka/arm_trace.py --record outputs/solve/fr3c3_rec/record.pt --report outputs/solve/fr3c3_rec/report.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from place_duo import Chain  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
ap = argparse.ArgumentParser()
ap.add_argument("--record", required=True)
ap.add_argument("--report", required=True)
ap.add_argument("--urdf_dir", default=os.path.join(ROOT, "assets_franka/left_m90"))
ap.add_argument("--layout", default=os.path.join(ROOT, "assets_franka/layout_left_m90_c3.json"))
a = ap.parse_args()
rec = torch.load(a.record, weights_only=False)
rep = json.load(open(a.report))["results"]
L = json.load(open(a.layout))[0]
q_all = rec["hand_q"].float().numpy()  # (envs, frames, 2, joints)
dt = float(rec["step_dt"])
chains = {s: Chain(os.path.join(a.urdf_dir, f"{s}/robot.urdf"), f"{s}_hand_wrist") for s in ("left", "right")}
print(f"[arm] {q_all.shape[0]} envs x {q_all.shape[1]} frames ({dt:.3f} s), {q_all.shape[3]} joints per hand; arm = the first 7", flush=True)


def sv(ch, q):
    M, ax, org = ch.fk(q)
    J = np.vstack([np.cross(ax, M[:3, 3] - org).T, ax.T])
    return float(np.linalg.svd(J, compute_uv=False).min())


for e, r in enumerate(rep):
    T = len(q_all[e])
    last = T
    for f in range(T - 1, 0, -1):
        if not np.array_equal(q_all[e, f], q_all[e, f - 1]):
            last = f + 1
            break
    cz = rec["cube"][e][:, 2].float().numpy()  # the drop: the cube 4 cm below its start (the arms flail afterwards)
    fall = np.nonzero(cz < cz[0] - 0.04)[0]
    if len(fall):
        last = min(last, int(fall[0]))
    line = f"env {e:2d} {r['status']:8s} skills {r['skills_done']:2d}/{r['n_skills']} end {last * dt:5.1f}s |"
    for h, s in enumerate(("left", "right")):
        ch = chains[s]
        q = q_all[e, :last, h, :7]
        q0 = np.array(L[s]["q"]["0"])
        margin = np.degrees(np.minimum(q - ch.lo, ch.hi - q))  # (frames, 7)
        jump = np.degrees(np.abs(np.diff(q, axis=0)).max(-1)) if last > 1 else np.zeros(1)
        drift = np.degrees(np.abs(q - q0).max(-1))
        tail = slice(max(0, last - int(3.0 / dt)), last)  # the last 3 s
        svs = [sv(ch, qq) for qq in q[tail][:: max(1, int(0.1 / dt))]]
        jmin = int(np.argmin(margin[tail].min(0)))
        line += (f" {s[0].upper()}: min margin {margin.min():5.1f} (last 3 s {margin[tail].min():5.1f}, j{jmin + 1}) sv last3s {min(svs):.3f}"
                 f" max jump {jump.max():5.1f} deg drift max {drift.max():5.0f} |")
    print(line, flush=True)
