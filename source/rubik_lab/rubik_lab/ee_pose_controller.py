"""End-effector pose control of a float-base Sharpa hand.

The official float-base asset chains 6 virtual joints between a fixed `base_root` and `<side>_hand_wrist`:
    x(prismatic) -> y(prismatic) -> z(prismatic) -> roll(about x) -> pitch(about y) -> yaw(about z)
so the wrist pose in the base frame is  p = (x, y, z),  R = Rx(roll) @ Ry(pitch) @ Rz(yaw)  (intrinsic XYZ).
A target EE pose therefore maps to joint targets in closed form (no IK). When the hand is later mounted on an
arm, only this class needs to be swapped for a differential-IK controller; the EE-pose action interface stays.
"""

from __future__ import annotations

import math

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation

from .assets import FLOAT_BASE_JOINTS


def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x + math.pi, 2 * math.pi) - math.pi


def euler_xyz_intrinsic_from_matrix(R: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Both solutions (roll, pitch, yaw) of R = Rx(roll) Ry(pitch) Rz(yaw). Shape (..., 3) each."""
    sb = R[..., 0, 2].clamp(-1.0, 1.0)
    b1 = torch.asin(sb)
    a1 = torch.atan2(-R[..., 1, 2], R[..., 2, 2])
    c1 = torch.atan2(-R[..., 0, 1], R[..., 0, 0])
    sol1 = torch.stack([a1, b1, c1], dim=-1)
    sol2 = torch.stack([a1 + math.pi, math.pi - b1, c1 + math.pi], dim=-1)
    return sol1, _wrap_to_pi(sol2)


def matrix_from_euler_xyz_intrinsic(rpy: torch.Tensor) -> torch.Tensor:
    a, b, c = rpy.unbind(-1)
    one, zero = torch.ones_like(a), torch.zeros_like(a)
    Rx = torch.stack([one, zero, zero, zero, a.cos(), -a.sin(), zero, a.sin(), a.cos()], -1).view(*a.shape, 3, 3)
    Ry = torch.stack([b.cos(), zero, b.sin(), zero, one, zero, -b.sin(), zero, b.cos()], -1).view(*a.shape, 3, 3)
    Rz = torch.stack([c.cos(), -c.sin(), zero, c.sin(), c.cos(), zero, zero, zero, one], -1).view(*a.shape, 3, 3)
    return Rx @ Ry @ Rz


class FloatBaseEEController:
    """Maps world-frame EE pose targets of one hand to its 6 virtual joint targets."""

    def __init__(self, robot: Articulation, side: str, ee_body: str | None = None):
        self.robot = robot
        self.side = side
        self.joint_ids, _ = robot.find_joints([f"{side}_{j}" for j in FLOAT_BASE_JOINTS], preserve_order=True)
        self.base_id = robot.find_bodies("base_root")[0][0]
        self.wrist_id = robot.find_bodies(f"{side}_hand_wrist")[0][0]
        self.ee_id = robot.find_bodies(ee_body or f"{side}_hand_wrist")[0][0]
        self._offset = None  # wrist -> ee, constant (rigidly attached)
        # the base is fixed (only moved by set_base_pose at resets): keep its pose instead of reading possibly stale
        # link data right after a teleport
        self.base_pos_w = robot.data.body_pos_w[:, self.base_id].clone()
        self.base_quat_w = robot.data.body_quat_w[:, self.base_id].clone()

    def set_base_pose(self, pos_w: torch.Tensor, quat_w: torch.Tensor, env_ids):
        """teleport the fixed base (root) of these envs; virtual joints are then relative to the new base."""
        self.base_pos_w[env_ids] = pos_w
        self.base_quat_w[env_ids] = quat_w
        ids = None if isinstance(env_ids, slice) else env_ids  # isaaclab wants None (not slice) for all envs
        self.robot.write_root_pose_to_sim(torch.cat([pos_w, quat_w], -1), env_ids=ids)

    def _ee_offset(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._offset is None:
            d = self.robot.data
            self._offset = math_utils.subtract_frame_transforms(
                d.body_pos_w[:, self.wrist_id], d.body_quat_w[:, self.wrist_id],
                d.body_pos_w[:, self.ee_id], d.body_quat_w[:, self.ee_id],
            )  # fmt: skip
        return self._offset

    def ee_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.robot.data
        return d.body_pos_w[:, self.ee_id], d.body_quat_w[:, self.ee_id]

    def joint_targets(
        self,
        ee_pos_w: torch.Tensor,
        ee_quat_w: torch.Tensor,
        joint_pos: torch.Tensor | None = None,
        env_ids: torch.Tensor | slice = slice(None),
    ) -> torch.Tensor:
        """(N,3),(N,4 wxyz) world EE pose -> (N,6) virtual joint targets. Rotations are unwrapped next to joint_pos."""
        d = self.robot.data
        off_p, off_q = (x[env_ids] for x in self._ee_offset())
        inv_q = math_utils.quat_inv(off_q)
        wrist_q = math_utils.quat_mul(ee_quat_w, inv_q)
        wrist_p = ee_pos_w - math_utils.quat_apply(wrist_q, off_p)
        p_b, q_b = math_utils.subtract_frame_transforms(
            self.base_pos_w[env_ids], self.base_quat_w[env_ids], wrist_p, wrist_q
        )
        sol1, sol2 = euler_xyz_intrinsic_from_matrix(math_utils.matrix_from_quat(q_b))
        cur = d.joint_pos[env_ids][:, self.joint_ids[3:]] if joint_pos is None else joint_pos[:, 3:]
        sol1 = cur + _wrap_to_pi(sol1 - cur)
        sol2 = cur + _wrap_to_pi(sol2 - cur)
        pick2 = (sol2 - cur).abs().sum(-1, keepdim=True) < (sol1 - cur).abs().sum(-1, keepdim=True)
        rpy = torch.where(pick2, sol2, sol1)
        return torch.cat([p_b, rpy], dim=-1)
