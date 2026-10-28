"""Which hand link touches which part of the cube (ground truth for grasp selection, rewards and labels).

One contact sensor on each cube body (core, top) filtered by every collision link of both hands (explicit paths, so
column i of the force matrix is HAND_LINKS[i]). Contact points are mapped to the 27 cubie cells of the *logical* cube
(independent of the per-move internal twist axis): cell index = 9 * ix + 3 * iy + iz, i = 0..2 along -..+ of each axis.
Logical faces: R=-y (iy=0), L=+y (iy=2), U=+z (iz=2), D=-z (iz=0), F=-x (ix=0), B=+x (ix=2).
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.sensors import ContactSensor, ContactSensorCfg

LINKS = [
    "hand_wrist", "hand_C_MC",
    "thumb_MC", "thumb_PP", "thumb_DP", "thumb_elastomer",
    "index_MCP_VL", "index_PP", "index_MP", "index_DP", "index_elastomer",
    "middle_MCP_VL", "middle_PP", "middle_MP", "middle_DP", "middle_elastomer",
    "ring_MCP_VL", "ring_PP", "ring_MP", "ring_DP", "ring_elastomer",
    "pinky_MC", "pinky_MCP_VL", "pinky_PP", "pinky_MP", "pinky_DP", "pinky_elastomer",
]  # fmt: skip  collision links of one hand (URDF)
SIDES = ("left", "right")
HAND_LINKS = [f"{s}_{link}" for s in SIDES for link in LINKS]  # 54 columns: left links, then right links
FINGERTIP_LINK = torch.tensor([("_DP" in n) or ("elastomer" in n) for n in LINKS])  # allowed for an edge grasp
FACE_CELLS = {"R": (1, 0), "L": (1, 2), "U": (2, 2), "D": (2, 0), "F": (0, 0), "B": (0, 2)}  # face -> (axis, index)


def face_layer_mask(face: str, device) -> torch.Tensor:
    """(27,) bool: cells belonging to the layer of `face`."""
    axis, idx = FACE_CELLS[face]
    i = torch.arange(27, device=device)
    coord = torch.stack([i // 9, (i // 3) % 3, i % 3], -1)
    return coord[:, axis] == idx


def make_cfgs(threshold_bodies=("core", "top")) -> dict[str, ContactSensorCfg]:
    return {
        b: ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/cube/{b}",
            history_length=1,
            track_contact_points=True,
            max_contact_data_count_per_prim=256,
            filter_prim_paths_expr=[f"/World/envs/env_.*/{n.split('_')[0]}_hand/{n}" for n in HAND_LINKS],
        )
        for b in threshold_bodies
    }


class CubeContacts:
    def __init__(self, scene):
        self.sensors = {b: ContactSensor(c) for b, c in make_cfgs().items()}
        for b, s in self.sensors.items():
            scene.sensors[f"cube_{b}_links"] = s

    def forces(self) -> torch.Tensor:
        """(N, 2 bodies [core, top], 2 hands, 27 links) normal contact force norms [N] (tangential friction excluded)."""
        f = torch.stack([torch.linalg.norm(self.sensors[b].data.force_matrix_w[:, 0], dim=-1) for b in ("core", "top")], 1)
        return f.view(f.shape[0], 2, 2, len(LINKS))

    def cells(self, core_pos_w, logical_quat, cube_size, threshold=0.05) -> torch.Tensor:
        """(N, 2 hands, 27 cells) bool: cubie cells touched by each hand (contact points of all its links)."""
        n = core_pos_w.shape[0]
        out = torch.zeros(n, 2, 27, dtype=torch.bool, device=core_pos_w.device)
        for b in ("core", "top"):
            d = self.sensors[b].data
            f = torch.linalg.norm(d.force_matrix_w[:, 0], dim=-1)  # (N, 54)
            p = d.contact_pos_w[:, 0]  # (N, 54, 3), NaN where no contact
            q = logical_quat.unsqueeze(1).expand(-1, p.shape[1], -1)
            local = math_utils.quat_apply_inverse(q.reshape(-1, 4), (p - core_pos_w.unsqueeze(1)).reshape(-1, 3)).view_as(p)
            idx = ((torch.nan_to_num(local, nan=0.0) + cube_size / 2) / (cube_size / 3)).floor().clamp(0, 2).long()
            cell = idx[..., 0] * 9 + idx[..., 1] * 3 + idx[..., 2]  # (N, 54)
            touch = (f > threshold) & torch.isfinite(p).all(-1)
            for h in range(2):
                sl = slice(h * len(LINKS), (h + 1) * len(LINKS))
                hot = torch.zeros(n, 27, device=core_pos_w.device).scatter_add_(1, cell[:, sl], touch[:, sl].float())
                out[:, h] |= hot > 0
        return out
