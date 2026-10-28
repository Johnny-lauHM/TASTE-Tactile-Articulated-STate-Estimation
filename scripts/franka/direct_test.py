"""Test a direct-mount assembly JSON (~/workspace/fr3-sharpa-rubik/assets/poses/direct_mount.json: base poses and the
hand clocking on each FR3 flange): over cube positions in front of the two bases, can each arm hold the solve env's
ready grip and roll its wrist continuously by +-90 deg (R / L / x) away from the joint limits?

    python scripts/franka/direct_test.py [--pose ~/workspace/fr3-sharpa-rubik/assets/poses/direct_mount.json]
"""
import argparse
import itertools
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import fr3_layout as L  # noqa: E402

ROOT = L.ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", default=os.path.expanduser("~/workspace/fr3-sharpa-rubik/assets/poses/direct_mount.json"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(ROOT, "assets_franka/direct_test.json"))
    a = ap.parse_args()
    P = json.load(open(a.pose))["robots"]
    b = P["left"]["base_position"][1]
    assert P["right"]["base_position"][1] == -b and all(P[s]["base_quaternion_wxyz"] == [1.0, 0.0, 0.0, 0.0] for s in P)
    yaw = {s: math.degrees(P[s]["arm_to_hand"]["rpy_rad"][2]) for s in P}
    print(f"[direct] bases y +-{b}, clocking {yaw}", flush=True)
    # cube at (x, 0, z) in the base frame: fr3_layout's dx = x, cz = z, b, psi 0
    grid = [(dx, b, cz, 0, yaw) for dx, cz in itertools.product(np.round(np.arange(0.25, 0.76, 0.05), 2), np.round(np.arange(0.15, 0.81, 0.05), 2))]
    with Pool(a.workers, initializer=L.init, initargs=(os.path.join(ROOT, "assets_franka/direct"),)) as pool:
        res = list(pool.imap_unordered(L.evaluate, grid, chunksize=2))
    res.sort(key=lambda x: (-x["score_deg"], -(x["left"]["reached_steps"] + x["right"]["reached_steps"])))
    json.dump(dict(pose_file=a.pose, clocking_deg=yaw, results=res), open(a.out, "w"))
    ok = [x for x in res if x["score_deg"] > 0]
    print(f"[direct] cube positions tried {len(res)}, all rolls feasible at {len(ok)} -> {a.out}", flush=True)
    for x in res[:8]:
        print(f"  cube x {x['dx']:.2f} z {x['cz']:.2f}: margin L {x['left']['margin_deg']:.1f} R {x['right']['margin_deg']:.1f} deg, "
              f"roll steps reached L {x['left']['reached_steps']}/18 R {x['right']['reached_steps']}/18", flush=True)
    old = [x for x in res if abs(x["dx"] - 0.5) < 1e-6 and abs(x["cz"] - 0.55) < 1e-6]
    if old:
        x = old[0]
        print(f"  (old cube spot x 0.50 z 0.55: margin L {x['left']['margin_deg']:.1f} R {x['right']['margin_deg']:.1f}, steps "
              f"{x['left']['reached_steps']}/{x['right']['reached_steps']})", flush=True)


if __name__ == "__main__":
    main()
