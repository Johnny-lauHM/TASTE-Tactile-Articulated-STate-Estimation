"""FR3 arm under each Sharpa hand (assets of ~/workspace/fr3-sharpa-rubik: FR3 v2 + printed adapter + hand).

`FR3EEController` has FloatBaseEEController's interface (joint_ids, ee_pose_w, joint_targets, set_base_pose,
reset_targets): a wrist pose target becomes 7 arm joint targets by damped least squares on a batched torch FK of the
robot URDF (base -> <side>_hand_wrist), warm-started from the previous command, so the solve env, its policies and
the executor keep commanding wrist poses.

Layout (scripts/franka/fr3_layout.py): relative to the cube, each base sits at (-dx, +-b, -cz) yawed +-psi; the ready
arm configuration (one IK branch that tracks the +-90 deg wrist rolls away from the limits) seeds every reset.
"""

from __future__ import annotations

import json
import math
import os
import xml.etree.ElementTree as ET

import torch

import isaaclab.utils.math as math_utils

FR3_DIR = os.path.expanduser("~/workspace/fr3-sharpa-rubik")
ARM_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]]


class TorchChain:
    """batched FK / DLS IK of the URDF joints from the root to `tip` (revolute joints are the variables)."""

    def __init__(self, urdf: str, tip: str, device):
        r = ET.parse(urdf).getroot()
        by_child = {j.find("child").get("link"): j for j in r.findall("joint")}
        js, link = [], tip
        while link in by_child:
            js.append(by_child[link])
            link = by_child[link].find("parent").get("link")
        self.fixed, self.axes, lo, hi = [], [], [], []
        for j in reversed(js):
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            M = torch.eye(4)
            M[:3, :3] = torch.tensor(_rpy(*rpy))
            M[:3, 3] = torch.tensor(xyz)
            mov = j.get("type") in ("revolute", "continuous")
            self.fixed.append(M.to(device))
            self.axes.append(torch.tensor([float(v) for v in j.find("axis").get("xyz").split()], device=device) if mov else None)
            if mov:
                lim = j.find("limit")
                lo.append(float(lim.get("lower")))
                hi.append(float(lim.get("upper")))
        self.lo, self.hi = torch.tensor(lo, device=device), torch.tensor(hi, device=device)

    def fk(self, q: torch.Tensor):
        """q (n, 7) -> tip transform (n, 4, 4) in the root frame, joint axes / origins (n, 7, 3) for the Jacobian."""
        n = q.shape[0]
        M = torch.eye(4, device=q.device).expand(n, 4, 4).clone()
        axes, orgs, i = [], [], 0
        for F, ax in zip(self.fixed, self.axes):
            M = M @ F
            if ax is not None:
                axes.append(M[:, :3, :3] @ ax)
                orgs.append(M[:, :3, 3])
                R = math_utils.matrix_from_quat(math_utils.quat_from_angle_axis(q[:, i], ax.expand(n, 3)))
                Mi = torch.eye(4, device=q.device).expand(n, 4, 4).clone()
                Mi[:, :3, :3] = R
                M = M @ Mi
                i += 1
        return M, torch.stack(axes, 1), torch.stack(orgs, 1)

    def ik(self, pos, rot, q0, iters=3, lam=0.01, w_rot=0.3, step=0.25, q_rest=None, k_null=0.0):
        """DLS steps toward the tip pose (pos (n,3), rot (n,3,3) in the root frame) from q0, within the limits.
        q_rest / k_null: the 7th (redundant) degree of freedom is pulled toward q_rest in the Jacobian's null space
        (the wrist pose is unaffected), so repeated rolls do not let the elbow drift into the limits / singularities."""
        q = q0.clone()
        eye = torch.eye(6, device=q.device)
        eye7 = torch.eye(7, device=q.device)
        for _ in range(iters):
            M, ax, org = self.fk(q)
            ep = pos - M[:, :3, 3]
            Re = rot @ M[:, :3, :3].transpose(1, 2)
            er = 0.5 * torch.stack([Re[:, 2, 1] - Re[:, 1, 2], Re[:, 0, 2] - Re[:, 2, 0], Re[:, 1, 0] - Re[:, 0, 1]], -1)
            e = torch.cat([ep, w_rot * er], -1)
            Jp = torch.cross(ax, M[:, None, :3, 3] - org, dim=-1).transpose(1, 2)
            J = torch.cat([Jp, w_rot * ax.transpose(1, 2)], 1)  # (n, 6, 7)
            J_inv = J.transpose(1, 2) @ torch.linalg.inv(J @ J.transpose(1, 2) + lam**2 * eye)  # (n, 7, 6)
            dq = (J_inv @ e.unsqueeze(-1)).squeeze(-1)
            if q_rest is not None and k_null > 0:
                dq = dq + ((eye7 - J_inv @ J) @ (k_null * (q_rest - q)).unsqueeze(-1)).squeeze(-1)
            q = torch.max(torch.min(q + dq.clamp(-step, step), self.hi - 1e-3), self.lo + 1e-3)
        return q


def load_layout(path: str, side: str, cube_pos: list[float]):
    """base position (env frame), base quaternion (wxyz) and the ready arm configuration of one side."""
    L = json.load(open(path))
    x = L[0] if isinstance(L, list) else L
    sy = 1.0 if side == "left" else -1.0
    pos = [cube_pos[0] - x["dx"], cube_pos[1] + sy * x["b"], cube_pos[2] - x["cz"]]
    h = math.radians(sy * x["psi_deg"]) / 2
    return pos, (math.cos(h), 0.0, 0.0, math.sin(h)), x[side]["q"]["0"]


class FR3EEController:
    def __init__(self, robot, side: str, base_pos_w: torch.Tensor, base_quat: tuple, seed_q: list[float], urdf_dir: str | None = None):
        self.robot, self.side = robot, side
        dev = robot.device
        self.joint_ids, _ = robot.find_joints(ARM_JOINTS, preserve_order=True)
        self.ee_id = robot.find_bodies(f"{side}_hand_wrist")[0][0]
        self.chain = TorchChain(os.path.join(urdf_dir or os.path.join(FR3_DIR, "assets"), f"{side}/robot.urdf"), f"{side}_hand_wrist", dev)
        n = base_pos_w.shape[0]
        self.base_pos_w = base_pos_w.clone()
        self.base_quat_w = torch.tensor(base_quat, device=dev).expand(n, 4).clone()
        self.seed = torch.tensor(seed_q, device=dev)
        self.q_cmd = self.seed.expand(n, 7).clone()
        self.k_null = float(os.environ.get("FR3_K_NULL", "0"))  # null-space pull toward the ready configuration (off: it fought the rolls)
        self.roll_table, self.roll_grid, self.roll_ref = None, None, None
        self.warm_prev = os.environ.get("FR3_WARM", "ref") == "prev"
        self.k_ref = float(os.environ.get("FR3_K_REF", "0.3"))

    def build_roll_table(self, pos_w, quat_w, rolls):
        """reference arm configurations along the wrist roll (rolls (K,) quarter turns, ascending, 0 included; the
        nominal wrist poses pos_w / quat_w (K, ...) of env 0): tracked by IK outward from the ready configuration in
        both directions, so every step's IK starts on this one continuous path (a 7-joint arm otherwise drifts in its
        redundant direction over repeated rolls, into the joint limits)."""
        base_p, base_q = self.base_pos_w[:1].expand(len(rolls), 3), self.base_quat_w[:1].expand(len(rolls), 4)
        p, q = math_utils.subtract_frame_transforms(base_p, base_q, pos_w, quat_w)
        R = math_utils.matrix_from_quat(q)
        k0 = int(torch.argmin(rolls.abs()))
        table = torch.zeros(len(rolls), 7, device=rolls.device)
        table[k0] = self.chain.ik(p[k0:k0 + 1], R[k0:k0 + 1], self.seed.unsqueeze(0), iters=100, step=0.3)[0]
        for rng_ in (range(k0 + 1, len(rolls)), range(k0 - 1, -1, -1)):
            prev = table[k0]
            for k in rng_:
                prev = self.chain.ik(p[k:k + 1], R[k:k + 1], prev.unsqueeze(0), iters=60)[0]
                table[k] = prev
        self.roll_table, self.roll_grid = table, rolls
        self.q_cmd[:] = table[k0]

    def reference(self, roll):
        """the reference configuration at roll (N,) quarter turns (linear in the table)."""
        g = self.roll_grid
        x = ((roll - g[0]) / (g[1] - g[0])).clamp(0, len(g) - 1 - 1e-6)
        i = x.floor().long()
        w = (x - i).unsqueeze(-1)
        return self.roll_table[i] * (1 - w) + self.roll_table[(i + 1).clamp(max=len(g) - 1)] * w

    def set_base_pose(self, pos_w, quat_w, env_ids):
        pass  # the arm base is bolted down

    def ee_pose_w(self):
        d = self.robot.data
        return d.body_pos_w[:, self.ee_id], d.body_quat_w[:, self.ee_id]

    def _local(self, pos_w, quat_w, env_ids):
        p, q = math_utils.subtract_frame_transforms(self.base_pos_w[env_ids], self.base_quat_w[env_ids], pos_w, quat_w)
        return p, math_utils.matrix_from_quat(q)

    def joint_targets(self, ee_pos_w, ee_quat_w, joint_pos=None, env_ids=slice(None)):
        p, R = self._local(ee_pos_w, ee_quat_w, env_ids)
        q_rest, k_null = self.seed, self.k_null
        if joint_pos is not None:
            q0 = joint_pos
        elif self.roll_table is not None and self.roll_ref is not None:
            ref = self.reference(self.roll_ref[env_ids])
            if self.warm_prev:  # continuous: from the last command, the redundancy pulled toward the reference path
                q0, q_rest, k_null = self.q_cmd[env_ids], ref, self.k_ref
            else:  # on the reference roll path, not the last command
                q0 = ref
        else:
            q0 = self.q_cmd[env_ids]
        q = self.chain.ik(p, R, q0, iters=int(os.environ.get("FR3_IK_ITERS", "3")), q_rest=q_rest, k_null=k_null)
        self.q_cmd[env_ids] = q
        return q

    def reset_targets(self, ee_pos_w, ee_quat_w, env_ids):
        """full IK from the layout's ready configuration (a reset can jump anywhere)."""
        p, R = self._local(ee_pos_w, ee_quat_w, env_ids)
        q0 = self.reference(torch.zeros(p.shape[0], device=p.device)) if self.roll_table is not None else self.seed.expand(p.shape[0], 7)
        q = self.chain.ik(p, R, q0, iters=60, step=0.3)
        self.q_cmd[env_ids] = q
        return q
