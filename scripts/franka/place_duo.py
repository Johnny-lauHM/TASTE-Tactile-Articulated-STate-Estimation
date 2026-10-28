"""Place the FR3 duo mount so both arms reach the solve env's wrist poses, and solve the arm IK.

Targets (env frame, from the ready cache): each <side>_hand_wrist at its ready pose and rolled -90/0/+90 deg about the
world y axis through the cube (the R/L/x wrist rolls). The mount faces +x (thumbs / the robot's body on the -x side).
A grid over the mount position scores the worst IK residual and the smallest joint-limit margin; the best placement
and its joint solutions go to --out (json).

    python scripts/franka/place_duo.py [--urdf assets_franka/fr3_duo_sharpa.urdf] [--out assets_franka/duo_placement.json]
"""
import argparse
import json
import os
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
FR3_READY = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


def rpy_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_mat(ax, q):
    ax = ax / np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K


def quat_mat(q):  # wxyz
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_quat(R):
    w = np.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 0.1:
        return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(max(1e-12, 1 + R[i, i] - R[j, j] - R[k, k])) * 2
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = s / 4
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def T(R=np.eye(3), p=np.zeros(3)):
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, p
    return M


class Chain:
    """the joints from the URDF root to one link; revolute joints are the variables."""

    def __init__(self, urdf, tip):
        r = ET.parse(urdf).getroot()
        by_child = {j.find("child").get("link"): j for j in r.findall("joint")}
        js, link = [], tip
        while link in by_child:
            js.append(by_child[link])
            link = by_child[link].find("parent").get("link")
        self.steps, self.names, self.lo, self.hi = [], [], [], []
        for j in reversed(js):
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            mov = j.get("type") in ("revolute", "continuous")
            ax = np.array([float(v) for v in j.find("axis").get("xyz").split()]) if mov else None
            self.steps.append((T(rpy_mat(*rpy), xyz), ax))
            if j.get("name").endswith("_flange_joint"):
                self.flange_step = len(self.steps) - 1
            if mov:
                lim = j.find("limit")
                self.names.append(j.get("name"))
                self.lo.append(float(lim.get("lower")))
                self.hi.append(float(lim.get("upper")))
        self.lo, self.hi = np.array(self.lo), np.array(self.hi)

    def set_ee_yaw(self, yaw):
        Tj, ax = self.steps[self.flange_step]
        self.steps[self.flange_step] = (T(rpy_mat(0, 0, yaw), Tj[:3, 3]), ax)

    def fk(self, q, base=np.eye(4)):
        M, i, axes, origins = base.copy(), 0, [], []
        for Tj, ax in self.steps:
            M = M @ Tj
            if ax is not None:
                axes.append(M[:3, :3] @ ax)
                origins.append(M[:3, 3].copy())
                M = M @ T(axis_mat(ax, q[i]))
                i += 1
        return M, np.array(axes), np.array(origins)

    def ik(self, target, q0, base, iters=200, lam=0.02):
        q = q0.copy()
        for _ in range(iters):
            M, axes, org = self.fk(q, base)
            ep = target[:3, 3] - M[:3, 3]
            Re = target[:3, :3] @ M[:3, :3].T
            er = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
            e = np.concatenate([ep, 0.3 * er])
            if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 1e-3:
                break
            J = np.vstack([np.cross(axes, M[:3, 3] - org).T, 0.3 * axes.T])
            dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo + 1e-3, self.hi - 1e-3)
        M = self.fk(q, base)[0]
        perr = np.linalg.norm(target[:3, 3] - M[:3, 3])
        Rr = target[:3, :3] @ M[:3, :3].T
        rerr = np.arccos(np.clip((np.trace(Rr) - 1) / 2, -1, 1))
        return q, perr, rerr


def targets(cache, side, cube):
    d = np.load(cache)
    p, q = d[f"{side}_ee_pos"][0], d[f"{side}_ee_quat"][0]
    W = T(quat_mat(q), p)
    out = {}
    for deg in (-90, 0, 90):
        Ry = T(axis_mat(np.array([0, 1.0, 0]), np.radians(deg)))
        out[deg] = T(p=cube) @ Ry @ T(p=-cube) @ W
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=os.path.join(ROOT, "assets_franka/fr3_duo_sharpa.urdf"))
    ap.add_argument("--cache", default=os.path.join(ROOT, "cache/solve_ready_ud_all4_flick.npz"))
    ap.add_argument("--out", default=os.path.join(ROOT, "assets_franka/duo_placement.json"))
    a = ap.parse_args()
    cube = np.load(a.cache)["cube_pos"][0]
    chains = {s: Chain(a.urdf, f"{s}_hand_wrist") for s in ("left", "right")}
    tg = {s: targets(a.cache, s, cube) for s in chains}
    rng = np.random.default_rng(0)
    seeds = [FR3_READY] + [FR3_READY + rng.normal(0, 0.6, 7) for _ in range(7)]
    best = None
    for yaw in np.radians(np.arange(-180, 180, 45)):
        for ch in chains.values():
            ch.set_ee_yaw(yaw)
        for mx in np.arange(-0.8, -0.2, 0.05):
            for mz in np.arange(-0.7, 0.15, 0.05):
                base = T(p=np.array([mx, 0.0, mz]))
                margin, travel, sol, ok = np.inf, 0.0, {}, True
                for s, ch in chains.items():
                    cands = []
                    for q0 in seeds:  # ready pose; the rolls from it (and from the seeds), least joint travel
                        q, pe, re = ch.ik(tg[s][0], q0, base, iters=120)
                        if pe > 0.003 or re > 0.03:
                            continue
                        rolls, tr = {0: q}, 0.0
                        for deg in (-90, 90):
                            opts = [ch.ik(tg[s][deg], q1, base, iters=120) for q1 in [q] + seeds[:3]]
                            opts = [o for o in opts if o[1] < 0.003 and o[2] < 0.03]
                            if not opts:
                                break
                            q2 = min(opts, key=lambda o: np.abs(o[0] - q).max())[0]
                            rolls[deg], tr = q2, max(tr, float(np.abs(q2 - q).max()))
                        if len(rolls) < 3:
                            continue
                        m = min(float(np.min(np.minimum(v - ch.lo, ch.hi - v))) for v in rolls.values())
                        cands.append((m - 0.1 * tr, m, tr, rolls))
                    if not cands:
                        ok = False
                        break
                    sc, m, tr, rolls = max(cands, key=lambda c: c[0])
                    sol[s], margin, travel = rolls, min(margin, m), max(travel, tr)
                if not ok:
                    continue
                score = min(margin, 0.35) - 0.1 * travel
                if best is None or score > best[0]:
                    best = (score, mx, mz, yaw, margin, travel, sol)
        print(f"[place] ee_yaw {np.degrees(yaw):+.0f}: best so far {None if best is None else (round(best[1], 2), round(best[2], 2), round(np.degrees(best[3])), round(np.degrees(best[4]), 1), round(np.degrees(best[5])))}", flush=True)
    if best is None:
        print("[place] no placement reaches all targets")
        return
    score, mx, mz, yaw, margin, travel, sol = best
    res = dict(mount_pos=[round(float(mx), 3), 0.0, round(float(mz), 3)], ee_yaw=float(yaw), min_limit_margin_rad=margin, max_roll_travel_rad=travel,
               cube_pos=cube.tolist(), joint_names={s: chains[s].names for s in chains},
               arm_q={s: {str(k): v.tolist() for k, v in sol[s].items()} for s in sol})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[place] mount at x={mx:.2f} z={mz:.2f}, ee_yaw {np.degrees(yaw):.0f} deg (env frame, cube at {np.round(cube, 3).tolist()}): "
          f"min joint-limit margin {np.degrees(margin):.1f} deg, max joint travel per roll {np.degrees(travel):.0f} deg -> {a.out}")
    for s in sol:
        for k, v in sol[s].items():
            print(f"  {s} roll {k:+d}: {np.round(np.degrees(v), 1).tolist()}")


if __name__ == "__main__":
    main()
