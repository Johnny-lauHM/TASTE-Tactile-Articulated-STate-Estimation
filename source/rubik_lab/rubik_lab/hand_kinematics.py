"""Batched, differentiable forward kinematics of a Sharpa Wave hand (pure PyTorch, no Isaac), for grasp synthesis.

Root = <side>_hand_wrist (the EE frame of the env). Collision links are represented by surface points sampled from their
URDF collision meshes; fingertip pads by the TacMap sensing-surface center and outward normal (elastomer frame).
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np
import torch

from .constants import ASSET_DIR, FINGER_JOINTS

PROJECT_DIR = os.path.dirname(ASSET_DIR)
# TacMap map centers (mm, elastomer frame) and outward normals (see assets/tacmap_maps)
PAD_CENTER = {"4F": (-0.010, -20.56, -4.64), "TH": (-0.24, -22.52, -3.65)}
PAD_NORMAL = {"4F": (-0.023, -0.249, -0.968), "TH": (-0.020, -0.457, -0.889)}


def _rpy(r, p, y):
    cx, sx, cy, sy, cz, sz = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ np.array(
        [[1, 0, 0], [0, cx, -sx], [0, sx, cx]]
    )


def _origin(el):
    T = np.eye(4)
    if el is not None:
        T[:3, :3] = _rpy(*[float(v) for v in el.get("rpy", "0 0 0").split()])
        T[:3, 3] = [float(v) for v in el.get("xyz", "0 0 0").split()]
    return T


def quat_to_mat(q):  # (..., 4) wxyz
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
         2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
         2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1,
    ).view(*q.shape[:-1], 3, 3)  # fmt: skip


def axis_angle_to_mat(v):  # (..., 3)
    th = torch.linalg.norm(v, dim=-1, keepdim=True).clamp(min=1e-9)
    k = v / th
    K = torch.zeros(*v.shape[:-1], 3, 3, device=v.device, dtype=v.dtype)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K = K - K.transpose(-1, -2)
    th = th.unsqueeze(-1)
    return torch.eye(3, device=v.device, dtype=v.dtype) + torch.sin(th) * K + (1 - torch.cos(th)) * K @ K


def mat_to_quat(R):  # (..., 3, 3) -> wxyz, robust branch-free variant
    m = R
    t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    q = torch.stack([1 + t, 1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], 1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2],
                     1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2]], -1).clamp(min=0).sqrt() / 2  # fmt: skip
    q[..., 1] = torch.copysign(q[..., 1], m[..., 2, 1] - m[..., 1, 2])
    q[..., 2] = torch.copysign(q[..., 2], m[..., 0, 2] - m[..., 2, 0])
    q[..., 3] = torch.copysign(q[..., 3], m[..., 1, 0] - m[..., 0, 1])
    return q / torch.linalg.norm(q, dim=-1, keepdim=True)


class HandFK:
    def __init__(self, side: str, device="cuda", pts_per_link: int = 64, seed: int = 0, dense_fingers=()):
        import trimesh

        self.side, self.device = side, device
        urdf = os.path.join(ASSET_DIR, f"sharpa_wave_float_base/{side}/{side}_sharpa_wave_with_float_base_with_wrist.urdf")
        mesh_dir = os.path.join(PROJECT_DIR, f"sharpa_wave/{side}_sharpa_wave/meshes")
        root = ET.parse(urdf).getroot()
        joints = {j.find("child").get("link"): j for j in root.findall("joint")}
        children = {}
        for c, j in joints.items():
            children.setdefault(j.find("parent").get("link"), []).append(c)
        # topological order below the wrist
        self.order, stack = [], [f"{side}_hand_wrist"]
        while stack:
            link = stack.pop(0)
            self.order.append(link)
            stack += children.get(link, [])
        self.parent, self.T0, self.axis, self.qidx = {}, {}, {}, {}
        names = [f"{side}_{j}" for j in FINGER_JOINTS]
        for link in self.order[1:]:
            j = joints[link]
            self.parent[link] = j.find("parent").get("link")
            self.T0[link] = torch.tensor(_origin(j.find("origin")), dtype=torch.float32, device=device)
            if j.get("type") == "revolute":
                self.axis[link] = torch.tensor([float(v) for v in j.find("axis").get("xyz").split()], device=device)
                self.qidx[link] = names.index(j.get("name"))
            lim = j.find("limit")
        self.lower = torch.zeros(len(names), device=device)
        self.upper = torch.zeros(len(names), device=device)
        for j in root.findall("joint"):
            if j.get("name") in names:
                lim = j.find("limit")
                self.lower[names.index(j.get("name"))] = float(lim.get("lower"))
                self.upper[names.index(j.get("name"))] = float(lim.get("upper"))
        # collision surface points per link (link frame)
        rng = np.random.default_rng(seed)
        self.link_pts = {}
        for l in root.findall("link"):
            name = l.get("name")
            col = l.find("collision")
            if col is None or name not in self.order:
                continue
            fn = col.find("geometry/mesh").get("filename").split("/")[-1]
            m = trimesh.load(os.path.join(mesh_dir, fn), force="mesh")
            m.apply_transform(_origin(col.find("origin")))
            if any('_'+finger+'_' in name for finger in dense_fingers):
                self.link_pts[name] = torch.tensor(np.concatenate([m.vertices,m.triangles_center]),dtype=torch.float32,device=device)
                continue
            pts, _ = trimesh.sample.sample_surface_even(m, pts_per_link, seed=int(rng.integers(1 << 30)))
            if len(pts) < pts_per_link:
                pts = np.concatenate([pts, m.vertices[rng.integers(len(m.vertices), size=pts_per_link - len(pts))]])
            self.link_pts[name] = torch.tensor(pts[:pts_per_link], dtype=torch.float32, device=device)
        self.pad_links = {f: f"{side}_{f}_elastomer" for f in ("thumb", "index", "middle", "ring", "pinky")}

    def forward(self, wrist_pos, wrist_R, q):
        """wrist pose (B,3),(B,3,3) and finger joints (B,22) -> dict link -> (pos (B,3), R (B,3,3))."""
        B = q.shape[0]
        out = {self.order[0]: (wrist_pos, wrist_R)}
        for link in self.order[1:]:
            pp, PR = out[self.parent[link]]
            T0 = self.T0[link]
            R = PR @ T0[:3, :3]
            p = pp + (PR @ T0[:3, 3].expand(B, 3).unsqueeze(-1)).squeeze(-1)
            if link in self.axis:
                R = R @ axis_angle_to_mat(self.axis[link] * q[:, self.qidx[link] : self.qidx[link] + 1])
            out[link] = (p, R)
        return out

    def points(self, frames, links=None):
        """world points of link surfaces: dict link -> (B, P, 3)."""
        res = {}
        for name, pts in self.link_pts.items():
            if links is not None and name not in links:
                continue
            p, R = frames[name]
            res[name] = p.unsqueeze(1) + torch.einsum("bij,pj->bpi", R, pts)
        return res

    def pads(self, frames):
        """pad center (B,3) and outward normal (B,3) for each finger."""
        out = {}
        for f, link in self.pad_links.items():
            kind = "TH" if f == "thumb" else "4F"
            p, R = frames[link]
            c = torch.tensor(PAD_CENTER[kind], device=self.device, dtype=p.dtype) * 1e-3
            n = torch.nn.functional.normalize(torch.tensor(PAD_NORMAL[kind], device=self.device, dtype=p.dtype), dim=0)
            out[f] = (p + R @ c, R @ n)
        return out
