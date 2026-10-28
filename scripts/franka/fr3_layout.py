"""Search a table layout for the two FR3 + adapter + Sharpa arms in which each arm holds the solve env's ready grip AND
can roll its wrist continuously by +-90 deg about the cube's y axis (the R / L / x skills), away from joint limits.

Layout: cube at (0, 0, cz); left base at (-dx, +b, 0) yawed +psi about z, right base mirrored. For each arm, IK
branches of the ready pose (many seeds) are each tracked through both rolls in 10 deg steps; an arm's score is the
smallest joint-limit margin along both paths (or -1 if a path breaks). Layout score = the worse arm.

    python scripts/franka/fr3_layout.py [--workers 10] [--out assets_franka/fr3_layout.json]
"""
import argparse
import itertools
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
FR3 = os.path.expanduser("~/workspace/fr3-sharpa-rubik")
sys.path.insert(0, os.path.dirname(__file__))
from place_duo import Chain, T, axis_mat, quat_mat  # noqa: E402

CACHE = os.path.join(ROOT, "cache/solve_ready_ud_all4_flick.npz")
_ch, _wr = {}, {}
# FR3 joint speed limits [rad/s]; a roll is 90 deg along a smoothstep in skills.ROLL_TIME = 0.95 s: the joint speed peaks at
# (its change per 10 deg of roll) x 1.5 x 9 / 0.95 per second
VMAX = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])
SPEED_MAX = float(os.environ.get("FR3_SPEED_MAX", "0"))  # >0: reject IK branches whose peak speed / limit exceeds this
MIN_SV = float(os.environ.get("FR3_MIN_SV", "0"))  # >0: reject IK branches whose roll paths come this close to a singularity


def init(urdf_dir=os.path.join(FR3, "assets")):
    d = np.load(CACHE)
    for s in ("left", "right"):
        _ch[s] = Chain(os.path.join(urdf_dir, f"{s}/robot.urdf"), f"{s}_hand_wrist")
        # wrist pose relative to the cube centre (env frame axes)
        _wr[s] = T(quat_mat(d[f"{s}_ee_quat_act"][0]), d[f"{s}_ee_pos_act"][0] - d["cube_pos"][0])


def arm_score(side, base, cube, rng):
    ch = _ch[side]
    Binv = np.linalg.inv(base)

    def target(deg):
        return Binv @ T(p=cube) @ T(axis_mat(np.array([0, 1.0, 0]), np.radians(deg))) @ _wr[side]

    seeds = [np.array([0, -0.6, 0, -2.1, 0, 1.5, 0.7854])] + [rng.uniform(ch.lo, ch.hi) for _ in range(14)]
    branches = []
    for q0 in seeds:
        q, pe, re = ch.ik(target(0), q0, np.eye(4), iters=150)
        if pe < 0.002 and re < 0.02 and all(np.abs(q - b).max() > 0.3 for b in branches):
            branches.append(q)
    best = (-1.0, None, 0)
    for q0 in branches:
        margin, paths, reached, speed = float(np.min(np.minimum(q0 - ch.lo, ch.hi - q0))), {0: q0.tolist()}, 0, 0.0
        paths_sv = np.inf
        for sgn in (1, -1):
            q = q0
            for k in range(1, 10):
                qp = q
                q, pe, re = ch.ik(target(sgn * 10 * k), q, np.eye(4), iters=60)
                speed = max(speed, float(np.max(np.abs(q - qp) * 1.5 * 9 / 0.95 / VMAX)))
                if MIN_SV > 0:
                    M_, ax_, org_ = ch.fk(q)
                    sv = float(np.linalg.svd(np.vstack([np.cross(ax_, M_[:3, 3] - org_).T, ax_.T]), compute_uv=False).min())
                    paths_sv = min(paths_sv, sv)
                if pe > 0.002 or re > 0.02 or np.abs(q - q0).max() > np.radians(200) or (SPEED_MAX > 0 and speed > SPEED_MAX) or (MIN_SV > 0 and paths_sv < MIN_SV):
                    margin = -1.0
                    break
                margin = min(margin, float(np.min(np.minimum(q - ch.lo, ch.hi - q))))
                reached += 1
            if margin < 0:
                break
            paths[sgn * 90] = q.tolist()
        paths["peak_speed_ratio"] = speed
        paths["min_sv"] = float(paths_sv) if np.isfinite(paths_sv) else None
        if margin > best[0] or (best[0] < 0 and reached > best[2]):
            best = (margin, paths, reached)
    return best


def evaluate(params):
    dx, b, cz, psi, yaw = params
    rng = np.random.default_rng(0)
    cube = np.array([0.0, 0.0, cz])
    out = {"dx": dx, "b": b, "cz": cz, "psi_deg": psi, "ee_yaw_deg": yaw}
    if yaw is not None:  # direct mount: the hand's clocking on the FR3 flange (one angle, or {side: angle})
        for s_, ch in _ch.items():
            ch.set_ee_yaw(np.radians(yaw[s_] if isinstance(yaw, dict) else yaw))
    score = np.inf
    for side, sy in (("left", 1), ("right", -1)):
        base = T(axis_mat(np.array([0, 0, 1.0]), np.radians(sy * psi)), np.array([-dx, sy * b, 0.0]))
        m, paths, reached = arm_score(side, base, cube, rng)
        out[side] = {"margin_deg": float(np.degrees(m)) if m >= 0 else -1, "reached_steps": reached, "q": paths}
        score = min(score, m)
    out["score_deg"] = float(np.degrees(score)) if score >= 0 else -1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(ROOT, "assets_franka/fr3_layout.json"))
    ap.add_argument("--direct", action="store_true", help="hand flange straight on the FR3 flange (assets_franka/direct), clocking searched")
    ap.add_argument("--urdf_dir", default="", help="<side>/robot.urdf under this directory (default: the fr3-sharpa-rubik assets)")
    ap.add_argument("--fixed_bases", type=float, default=0.0, help=">0: bases fixed at y = +-this, unrotated; only the cube position is searched")
    a = ap.parse_args()
    yaws = list(range(-180, 180, 45)) if a.direct else [None]
    grid = list(itertools.product([0.3, 0.4, 0.5, 0.6], [0.15, 0.25, 0.35, 0.45], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7], [-40, -20, 0, 20, 40], yaws))
    if a.fixed_bases > 0:
        grid = list(itertools.product(np.round(np.arange(0.25, 0.76, 0.05), 2), [a.fixed_bases], np.round(np.arange(0.15, 0.81, 0.05), 2), [0], yaws))
    res = []
    urdf_dir = a.urdf_dir or (os.path.join(ROOT, "assets_franka/direct") if a.direct else os.path.join(FR3, "assets"))
    with Pool(a.workers, initializer=init, initargs=(urdf_dir,)) as pool:
        for i, r in enumerate(pool.imap_unordered(evaluate, grid, chunksize=2)):
            res.append(r)
            if (i + 1) % 200 == 0:
                ok = sorted([x for x in res if x["score_deg"] > 0], key=lambda x: -x["score_deg"])
                print(f"[layout] {i + 1}/{len(grid)}: feasible {len(ok)}; best "
                      f"{[(x['dx'], x['b'], x['cz'], x['psi_deg'], x['ee_yaw_deg'], round(x['score_deg'], 1)) for x in ok[:3]]}", flush=True)
    res.sort(key=lambda x: (-x["score_deg"], -(x["left"]["reached_steps"] + x["right"]["reached_steps"])))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"))
    print(f"[layout] feasible {sum(x['score_deg'] > 0 for x in res)}/{len(res)} -> {a.out}")
    for x in res[:8]:
        sp = [(x[s_]['q'] or {}).get('peak_speed_ratio', 0.0) for s_ in ('left', 'right')]
        print(f"  dx {x['dx']} b {x['b']} cz {x['cz']} psi {x['psi_deg']} ee_yaw {x['ee_yaw_deg']} peak speed/limit L {sp[0]:.2f} R {sp[1]:.2f}: margin L {x['left']['margin_deg']:.1f} "
              f"R {x['right']['margin_deg']:.1f} deg, steps {x['left']['reached_steps']}/{x['right']['reached_steps']}")


if __name__ == "__main__":
    main()
