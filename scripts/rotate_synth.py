"""Turn synthesized grasps (cube frame) by 180 deg about the cube's y axis: the F grip (thumb on D, index on U:front)
becomes a B grip (thumb on U, index on D:back) with the same hands on the same sides.
    python scripts/rotate_synth.py cache/synth70s_F.npz cache/synth70s_B.npz
"""
import sys
import numpy as np
import torch
sys.path.insert(0, "source/rubik_lab")
from rubik_lab.hand_kinematics import mat_to_quat, quat_to_mat  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
d = dict(np.load(src))
Ry = torch.diag(torch.tensor([-1.0, 1.0, -1.0]))
for s in ("left", "right"):
    p = torch.tensor(d[f"{s}_wrist_pos"])
    R = quat_to_mat(torch.tensor(d[f"{s}_wrist_quat"]))
    d[f"{s}_wrist_pos"] = (p @ Ry.T).numpy()
    d[f"{s}_wrist_quat"] = mat_to_quat(Ry.unsqueeze(0) @ R).numpy()
np.savez(dst, **d)
print(f"[rotate] {len(d['loss'])} grasps -> {dst}")
