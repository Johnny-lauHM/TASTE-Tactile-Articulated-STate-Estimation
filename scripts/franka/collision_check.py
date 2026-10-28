"""Clearances of a two-arm layout along its wrist rolls (no collision library: points sampled on every collision mesh,
nearest-point distances). Each arm tracks its rolls (0 -> +-90 deg about the cube's y axis, 10 deg steps) from its
ready configuration; x / x' roll both arms and the cube together.

Reported per pose (minimum over the path): left vs right bodies (arms, palms, fingers), arm links (FR3 links +
flange) vs the cube, and each hand's fingers vs its own upper arm links (1-5).

    python scripts/franka/collision_check.py --layout assets_franka/direct_layout.json [--rank 0] [--urdf_dir assets_franka/direct]
"""
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.dirname(__file__))
from place_duo import Chain, T, axis_mat, quat_mat, rpy_mat  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CACHE = os.path.join(ROOT, "cache/solve_ready_ud_all4_flick.npz")
FINGER_ORDER = ["thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA", "thumb_IP", "index_MCP_FE", "index_MCP_AA",
                "index_PIP", "index_DIP", "middle_MCP_FE", "middle_MCP_AA", "middle_PIP", "middle_DIP", "ring_MCP_FE",
                "ring_MCP_AA", "ring_PIP", "ring_DIP", "pinky_CMC", "pinky_MCP_FE", "pinky_MCP_AA", "pinky_PIP", "pinky_DIP"]
CUBE = 0.070


class Tree:
    """FK of every link of a URDF; collision points per link (sampled once, link frame)."""

    def __init__(self, urdf, n_pts=400, seed=0):
        r = ET.parse(urdf).getroot()
        self.joints = {}
        for j in r.findall("joint"):
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            ax = j.find("axis")
            self.joints[j.find("child").get("link")] = (j.find("parent").get("link"), T(rpy_mat(*rpy), np.array(xyz)), j.get("type"),
                                                         np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else None, j.get("name"))
        self.fixed_yaw = None
        rng = np.random.default_rng(seed)
        self.pts = {}
        for link in r.findall("link"):
            P = []
            for c in link.findall("collision"):
                m = c.find("geometry/mesh")
                if m is None:
                    continue
                o = c.find("origin")
                xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
                rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
                sc = [float(v) for v in m.get("scale", "1 1 1").split()]
                mesh = trimesh.load(m.get("filename"), force="mesh")
                mesh.apply_scale(sc)
                hull = mesh.convex_hull  # the simulator's collider approximation
                pts = np.vstack([trimesh.sample.sample_surface(hull, n_pts, seed=int(rng.integers(1 << 30)))[0], hull.vertices])
                M = T(rpy_mat(*rpy), np.array(xyz))
                P.append(pts @ M[:3, :3].T + M[:3, 3])
            if P:
                self.pts[link.get("name")] = np.vstack(P)

    def set_ee_yaw(self, yaw):
        for c, (p, M, t, ax, nm) in self.joints.items():
            if nm.endswith("_flange_joint"):
                self.joints[c] = (p, T(rpy_mat(0, 0, yaw), M[:3, 3]), t, ax, nm)

    def fk(self, q: dict, base):
        out = {}

        def get(link):
            if link in out:
                return out[link]
            if link not in self.joints:
                out[link] = base
                return base
            p, M, t, ax, nm = self.joints[link]
            W = get(p) @ M
            if t in ("revolute", "continuous"):
                W = W @ T(axis_mat(ax, q.get(nm, 0.0)))
            out[link] = W
            return W

        for link in self.pts:
            get(link)
        return out

    def world_points(self, q, base, links):
        F = self.fk(q, base)
        return {l: self.pts[l] @ F[l][:3, :3].T + F[l][:3, 3] for l in links if l in self.pts}


def mind(A, B):
    a, b = torch.as_tensor(A, dtype=torch.float32), torch.as_tensor(B, dtype=torch.float32)
    return float(torch.cdist(a, b).min())


def cube_points(R, c, n=12):
    g = np.linspace(-CUBE / 2, CUBE / 2, n)
    P = []
    for ax in range(3):
        for s in (-CUBE / 2, CUBE / 2):
            u, v = np.meshgrid(g, g)
            p = np.zeros((u.size, 3))
            p[:, ax] = s
            p[:, (ax + 1) % 3], p[:, (ax + 2) % 3] = u.ravel(), v.ravel()
            P.append(p)
    return np.vstack(P) @ R.T + c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default=os.path.join(ROOT, "assets_franka/direct_layout.json"))
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--urdf_dir", default=os.path.join(ROOT, "assets_franka/direct"))
    a = ap.parse_args()
    L = json.load(open(a.layout))
    L = L["results"] if isinstance(L, dict) else L
    x = L[a.rank]
    d = np.load(CACHE)
    cube = np.array([0.0, 0.0, x["cz"]])
    Rc0 = quat_mat(d["cube_quat"][0])
    sides = {"left": 1, "right": -1}
    trees, chains, bases, qready, wr, fingers = {}, {}, {}, {}, {}, {}
    for s, sy in sides.items():
        urdf = os.path.join(a.urdf_dir, f"{s}/robot.urdf")
        trees[s], chains[s] = Tree(urdf), Chain(urdf, f"{s}_hand_wrist")
        y = x[s].get("ee_yaw_deg")
        if y is not None:
            trees[s].set_ee_yaw(np.radians(y))
            chains[s].set_ee_yaw(np.radians(y))
        bases[s] = T(axis_mat(np.array([0, 0, 1.0]), np.radians(sy * x["psi_deg"])), np.array([-x["dx"], sy * x["b"], 0.0]))
        qready[s] = np.array(x[s]["q"]["0"])
        wr[s] = T(quat_mat(d[f"{s}_ee_quat_act"][0]), d[f"{s}_ee_pos_act"][0] - d["cube_pos"][0])
        fingers[s] = {f"{s}_{n}": float(v) for n, v in zip(FINGER_ORDER, d[f"{s}_fingers_act"][0])}
    arm_links = [f"fr3_link{i}" for i in range(9)]

    def arm_q(s, deg):
        """track the roll from the ready configuration in 10 deg steps."""
        ch, q = chains[s], qready[s]
        Binv = np.linalg.inv(bases[s])
        for k in range(1, abs(deg) // 10 + 1):
            tgt = Binv @ T(p=cube) @ T(axis_mat(np.array([0, 1.0, 0]), np.radians(np.sign(deg) * 10 * k))) @ wr[s]
            q, pe, re = ch.ik(tgt, q, np.eye(4), iters=60)
        return q

    print(f"[coll] layout rank {a.rank}: dx {x['dx']} b {x['b']} cz {x['cz']} psi {x['psi_deg']} clocking "
          f"L {x['left'].get('ee_yaw_deg')} R {x['right'].get('ee_yaw_deg')}", flush=True)
    poses = {"ready": {}, "R": {"right": 90}, "R'": {"right": -90}, "L": {"left": 90}, "L'": {"left": -90},
             "x": {"left": 90, "right": 90}, "x'": {"left": -90, "right": -90}}
    worst = {}
    for name, rolls in poses.items():
        steps = max([abs(v) for v in rolls.values()] + [0]) // 10
        res = {"L-R": np.inf, "arm-cube": np.inf, "fingers-own-arm": np.inf}
        where = {}
        for k in range(steps + 1):
            frac = k / max(steps, 1)
            W = {}
            for s in sides:
                deg = int(round(rolls.get(s, 0) * frac))
                q = dict(fingers[s])
                q.update({f"fr3_joint{i + 1}": v for i, v in enumerate(arm_q(s, deg) if deg else qready[s])})
                W[s] = trees[s].world_points(q, bases[s], trees[s].pts.keys())
            cdeg = rolls.get("left", 0) * frac if name.startswith("x") else 0.0
            Pc = cube_points(axis_mat(np.array([0, 1.0, 0]), np.radians(cdeg)) @ Rc0, cube)
            for la, Pa in W["left"].items():
                for lb, Pb in W["right"].items():
                    if la in arm_links or lb in arm_links or "flange" in la or "flange" in lb:
                        v = mind(Pa, Pb)
                        if v < res["L-R"]:
                            res["L-R"], where["L-R"] = v, (la, lb, k * 10)
            for s in sides:
                for l, P in W[s].items():
                    if l in arm_links or l.endswith("_hand_flange"):
                        v = mind(P, Pc)
                        if v < res["arm-cube"]:
                            res["arm-cube"], where["arm-cube"] = v, (l, k * 10)
                own = np.vstack([W[s][l] for l in [f"fr3_link{i}" for i in range(1, 6)] if l in W[s]])
                fin = np.vstack([P for l, P in W[s].items() if any(f in l for f in ("thumb", "index", "middle", "ring", "pinky"))])
                v = mind(own, fin)
                if v < res["fingers-own-arm"]:
                    res["fingers-own-arm"], where["fingers-own-arm"] = v, (s, k * 10)
        worst[name] = res
        print(f"  {name:6s} " + "  ".join(f"{kk} {vv * 1000:6.1f} mm {where.get(kk)}" for kk, vv in res.items()), flush=True)
    m = min(min(v.values()) for v in worst.values())
    print(f"[coll] smallest clearance over all poses {m * 1000:.1f} mm (sampling resolution ~3-5 mm)", flush=True)


if __name__ == "__main__":
    main()
