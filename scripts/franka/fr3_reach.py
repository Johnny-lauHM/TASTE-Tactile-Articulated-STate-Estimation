"""Can the two real FR3 + adapter + Sharpa arms (~/workspace/fr3-sharpa-rubik) hold the solve env's grip and do its
wrist rolls? Bounded IK of each <side>_hand_wrist to the ready cache pose and to that pose rolled about the world y
axis through the cube (R / R': right wrist; L / L': left wrist; x / x': both), each roll continuing from the ready
solution. Prints residuals, joint-limit margins and joint travel; writes the configurations for fr3_scene.py.

    python scripts/franka/fr3_reach.py [--cache cache/solve_ready_ud_all4_flick.npz] [--out assets_franka/fr3_reach.json]
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
FR3 = os.path.expanduser("~/workspace/fr3-sharpa-rubik")
sys.path.insert(0, os.path.join(FR3, "scripts"))
from prepare_grasp import chain  # noqa: E402  (FK of assets/<side>/robot.urdf to <side>_hand_wrist, bounds 0.02 in)

BASE = {"left": np.array([0.0, 0.4, 0.0]), "right": np.array([0.0, -0.4, 0.0])}
CUBE_WORLD = np.array([0.5, 0.0, 0.55])
SEED = np.array([0, -0.6, 0, -2.1, 0, 1.5, 0.7854])
POSES = {"ready": {}, "R": {"right": 90}, "R'": {"right": -90}, "L": {"left": 90}, "L'": {"left": -90},
         "x": {"left": 90, "right": 90}, "x'": {"left": -90, "right": -90}}


def solve(fk, lo, hi, pos, rot, seeds):
    def err(q):
        t = fk(q)
        return np.r_[t[:3, 3] - pos, 0.2 * R.from_matrix(rot.T @ t[:3, :3]).as_rotvec()]
    best = None
    for s in seeds:
        sol = least_squares(err, np.clip(s, lo, hi), bounds=(lo, hi), max_nfev=300, gtol=1e-10)
        if best is None or np.linalg.norm(sol.fun) < np.linalg.norm(best.fun):
            best = sol
    t = fk(best.x)
    return best.x, float(np.linalg.norm(t[:3, 3] - pos)), float(np.degrees(np.linalg.norm(R.from_matrix(rot.T @ t[:3, :3]).as_rotvec())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(ROOT, "cache/solve_ready_ud_all4_flick.npz"))
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "assets_franka/fr3_reach.json"))
    a = ap.parse_args()
    d = np.load(a.cache)
    off = CUBE_WORLD - d["cube_pos"][a.idx]
    rng = np.random.default_rng(0)
    res = dict(cache=a.cache, idx=a.idx, cube_position=CUBE_WORLD.tolist(), cube_quaternion=d["cube_quat"][a.idx].tolist(),
               base=dict((s, v.tolist()) for s, v in BASE.items()), poses={},
               fingers={s: d[f"{s}_fingers_act"][a.idx].tolist() for s in BASE})
    ready_q = {}
    for name, rolls in POSES.items():
        res["poses"][name] = {}
        for side in ("left", "right"):
            fk, lo, hi = chain(side)
            p = d[f"{side}_ee_pos_act"][a.idx] + off
            rot = R.from_quat(d[f"{side}_ee_quat_act"][a.idx][[1, 2, 3, 0]]).as_matrix()
            deg = rolls.get(side, 0)
            Ry = R.from_rotvec([0, np.radians(deg), 0]).as_matrix()
            p, rot = CUBE_WORLD + Ry @ (p - CUBE_WORLD), Ry @ rot
            if name == "ready" or deg == 0:
                seeds = [SEED] + [rng.uniform(lo, hi) for _ in range(25)] if name == "ready" else [ready_q[side]]
                q, pe, re = solve(fk, lo, hi, p - BASE[side], rot, seeds)
                ready_q.setdefault(side, q)
            else:  # track the roll in 5 deg steps from the ready configuration (a continuous joint path)
                p0 = d[f"{side}_ee_pos_act"][a.idx] + off
                r0 = R.from_quat(d[f"{side}_ee_quat_act"][a.idx][[1, 2, 3, 0]]).as_matrix()
                q, reached = ready_q[side], 0
                for k in range(1, abs(deg) // 5 + 1):
                    Rk = R.from_rotvec([0, np.radians(np.sign(deg) * 5 * k), 0]).as_matrix()
                    qk, pe, re = solve(fk, lo, hi, CUBE_WORLD + Rk @ (p0 - CUBE_WORLD) - BASE[side], Rk @ r0, [q])
                    if pe > 0.002 or re > 1.0:
                        break
                    q, reached = qk, 5 * k
                pe, re = (pe, re) if reached == abs(deg) else (pe, re)
                res.setdefault("reached", {})[f"{name}/{side}"] = reached
            margin = float(np.degrees(np.min(np.minimum(q - (lo - 0.02), (hi + 0.02) - q))))
            travel = float(np.degrees(np.abs(q - ready_q[side]).max()))
            res["poses"][name][side] = dict(arm_q=q.tolist(), pos_err_mm=pe * 1000, rot_err_deg=re, margin_deg=margin, travel_deg=travel)
            ok = "OK " if pe < 0.002 and re < 1.0 else "NO "
            if deg:
                ok += f"(path to {res['reached'][name + '/' + side]:+d}/{deg:+d} deg) "
            print(f"{ok}{name:6s} {side:5s} pos err {pe * 1000:6.1f} mm  rot err {re:5.1f} deg  limit margin {margin:5.1f} deg  "
                  f"max joint travel from ready {travel:5.0f} deg  q {np.round(np.degrees(q)).astype(int).tolist()}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("->", a.out)


if __name__ == "__main__":
    main()
