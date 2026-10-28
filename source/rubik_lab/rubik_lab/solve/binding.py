"""Face binding of the two-body cube (pure torch).

The simulated cube is one articulation: root body `core` (2/3 of the cube) and body `top` (1/3) on a revolute joint
about the root's local +z, both frames at the cube center. Its outer shape is a cube whenever the joint is at a multiple
of 90 deg, and it is invariant under the 24 rotations of the cube group. So the free layer can be moved to any face by
re-expressing the root orientation as R_core @ S (S in the group, S e_z -> desired face) with the joint at 0: the outer
surface does not move, only which slab is free. The residual layer angle (bounded by the caller's alignment
tolerance) is snapped to the nearest quarter turn, which models the corner cutting of a real cube that lets an
orthogonal layer start turning while the previous one is a few degrees off.
"""

from __future__ import annotations

import itertools

import torch


def cube_group(device=None) -> torch.Tensor:
    """(24, 3, 3) proper rotation matrices of the cube (signed permutation matrices with det +1)."""
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            m = torch.zeros(3, 3)
            for r, (c, s) in enumerate(zip(perm, signs)):
                m[r, c] = s
            if torch.det(m) > 0:
                mats.append(m)
    return torch.stack(mats).to(device)


def quarter_residual(angle: torch.Tensor) -> torch.Tensor:
    """signed distance [rad] to the nearest multiple of 90 deg."""
    q = 0.5 * torch.pi
    return angle - torch.round(angle / q) * q


def rebind_rotation(R_core: torch.Tensor, normal: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
    """New root rotation matrices (N,3,3) whose local +z points along `normal` (N,3, env frame).

    Among the four group elements with S e_z closest to the normal the one closest to the identity is used (any of
    them gives the same outer shape).
    """
    n_local = torch.einsum("nji,nj->ni", R_core, normal)  # R_core^T normal
    align = n_local @ group[:, :, 2].T  # (N, 24): normal . S e_z
    trace = group.diagonal(dim1=-2, dim2=-1).sum(-1)
    best = (align * 10.0 + trace.unsqueeze(0) * 1e-3).argmax(-1)
    return R_core @ group[best]


def face_index(normal: torch.Tensor) -> torch.Tensor:
    """(N,) slot of a logical face normal (N,3): 2 * axis + (1 if it points the negative way)."""
    axis = normal.abs().argmax(-1)
    neg = normal.gather(-1, axis.unsqueeze(-1)).squeeze(-1) < 0
    return 2 * axis + neg.long()
