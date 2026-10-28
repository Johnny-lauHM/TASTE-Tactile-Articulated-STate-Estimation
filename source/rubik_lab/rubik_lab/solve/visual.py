"""Paint the simulated cube's stickers from the logical cube state (videos only; physics never reads colors).

The sticker prims are attached to the two physics bodies, so after a face re-binding they show the wrong pattern.
Call `recolor` right after a binding while the cube is aligned: every sticker's body-local slot is mapped to its
hand-frame slot through the (snapped) core orientation relative to the ready grip, and gets that facelet's color.
"""

from __future__ import annotations

import numpy as np

from .cube_logic import SLOT_INDEX

PALETTE = {"U": (0.95, 0.95, 0.95), "D": (1.0, 0.85, 0.0), "F": (0.0, 0.6, 0.2), "B": (0.0, 0.25, 0.8),
           "L": (1.0, 0.45, 0.0), "R": (0.8, 0.05, 0.05)}  # fmt: skip


class StickerPainter:
    def __init__(self, env_path: str = "/World/envs/env_0/cube"):
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        self.items = []  # (attr, local pos, local normal)
        for body in ("core", "top"):
            vis = stage.GetPrimAtPath(f"{env_path}/{body}/visuals")
            for prim in vis.GetChildren():
                name = prim.GetName()
                if "_st_" not in name:
                    continue
                ijk = name.split("_")[1]
                pos = np.array([int(c) - 1 for c in ijk])
                ax = "xyz".index(name[-2])
                nrm = np.zeros(3, dtype=int)
                nrm[ax] = 1 if name[-1] == "p" else -1
                self.items.append((UsdGeom.Gprim(prim).GetDisplayColorAttr(), pos, nrm))

    def recolor(self, R_rel: np.ndarray, facelets: str):
        """R_rel: core orientation relative to the ready grip's cube orientation (3x3, ~ a cube symmetry)."""
        from pxr import Gf, Vt

        M = np.round(R_rel).astype(int)
        for attr, p, nrm in self.items:
            hp, hn = tuple(int(v) for v in M @ p), tuple(int(v) for v in M @ nrm)
            slot = SLOT_INDEX.get((hp, hn))
            if slot is None:
                continue
            attr.Set(Vt.Vec3fArray([Gf.Vec3f(*PALETTE[facelets[slot]])]))
