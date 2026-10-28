"""Build a settled ready-pose cache without replacing the original grasp targets.

The old cache starts with the index on the layer; a complete primitive ends with
it released. This prepares both index fingers in that same released posture.
Each source grasp is checked twice and only one physical state per source is saved.
"""
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--out", default="cache/g70s_UDR_cycle_ready.npz")
p.add_argument("--hold_s", type=float, default=2.0)
p.add_argument("--copies", type=int, default=2)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app

import json
from pathlib import Path
import numpy as np
import torch
import isaaclab.utils.math as mu
from rubik_lab.tasks.bimanual_twist.cycle_cfg import CycleUEnvCfg
from rubik_lab.tasks.bimanual_twist.cycle_env import PrimitiveCycleEnv
from rubik_lab.moves import MOVE_INDEX

source_path = Path("cache/g70s_UDR.npz")
data = dict(np.load(source_path))
r = data["flick_reach"]
valid = ((data["move_id"] // 3) == 2) & (r[:, MOVE_INDEX["U"]] >= 40) & (r[:, MOVE_INDEX["U'"]] >= 40)
indices = np.flatnonzero(valid)
if not len(indices):
    raise RuntimeError("No shared U/U-prime source grasps")


class PreparationEnv(PrimitiveCycleEnv):
    def _get_dones(self):
        self._update_task_state()
        # This is a fixed-duration preparation/measurement, never an RL success.
        z = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return z, z


cfg = CycleUEnvCfg()
cfg.seed = 42
cfg.scene.num_envs = len(indices) * a.copies
cfg.episode_length_s = a.hold_s + 1
u = PreparationEnv(cfg)
u.forced_grasp_idx = torch.tensor(np.tile(indices, a.copies), device=u.device)
u.forced_move_id = torch.full((u.num_envs,), MOVE_INDEX["U"], dtype=torch.long, device=u.device)
u.reset()
ready_action = u._idle_offsets()[u.flick_cols[u.move_id]] / cfg.residual_finger_range
stable = torch.ones(u.num_envs, dtype=torch.bool, device=u.device)
hold_pos = hold_quat = None
for k in range(round(a.hold_s / u.step_dt)):
    # Initial-state construction only: once both indices are retracted, align
    # the free layer before validating the ready grasp. Never done in a policy
    # episode or a sequence evaluation.
    if k == round(0.75 / u.step_dt):
        zero = torch.zeros(u.num_envs, 1, device=u.device)
        u.cube.write_joint_state_to_sim(zero, zero, joint_ids=[u.twist_id])
        u.scene.write_data_to_sim()
        u.sim.forward()
    u.step(ready_action)
    pos, quat, twist, vel = u._cube_state()
    core = u.cube.data
    # Same pose-window stability definition as the full-cycle acceptance.
    if k >= round((a.hold_s - cfg.cycle_stable_s) / u.step_dt):
        if hold_pos is None:
            hold_pos, hold_quat = pos.clone(), quat.clone()
        quiet = (pos-hold_pos).norm(dim=-1) <= cfg.cycle_stable_pos
        quiet &= mu.quat_error_magnitude(quat,hold_quat) <= cfg.cycle_stable_rot
        quiet &= mu.wrap_to_pi(twist).abs() < np.radians(5)
        quiet &= vel.abs() < cfg.move_success_vel
        quiet &= pos[:, 2] > cfg.cube_drop_height
        pads = (u.layer_f.sum(-1) > cfg.tactile_contact_threshold).sum(-1)
        quiet &= (pads >= cfg.home_min_pads).all(-1)
        for s in ("left", "right"):
            cols = [i for i, name in enumerate(__import__('rubik_lab.constants', fromlist=['FINGER_JOINTS']).FINGER_JOINTS) if name.startswith("index")]
            q = u.hands[s].data.joint_pos[:, u.finger_ids[s]]
            tgt = (u.finger_grasp[s] + u._idle_offsets()).clamp(min=u.finger_lower[s], max=u.finger_upper[s])
            quiet &= (q[:, cols] - tgt[:, cols]).abs().amax(-1) < cfg.cycle_finger_tol
        stable &= quiet
finite = torch.isfinite(pos).all(-1) & torch.isfinite(quat).all(-1)
keep_each = stable & finite & (u.cycle_wrist_pos_peak <= cfg.cycle_wrist_pos)
keep_each &= u.peak_pos < 0.020
keep_each &= u.peak_rot < np.radians(15)
keep = keep_each.cpu().numpy().reshape(a.copies, len(indices)).all(0)
selected = indices[keep]
report = {"source_count": len(indices), "copies": a.copies, "kept": len(selected),
          "source_indices": selected.tolist(), "preparation_seconds": a.hold_s,
          "per_copy_pass": keep_each.cpu().numpy().reshape(a.copies, -1).sum(1).tolist(),
          "purpose": "ready-pose initial-state construction with aligned layer, not manipulation success",
          "alignment_initialization_at_s": 0.75, "contract_version": cfg.cycle_contract_version,
          "hold_pos_excursion_mm": np.percentile(((pos-hold_pos).norm(dim=-1)*1000).cpu().numpy(),[50,90]).tolist(),
          "hold_rot_excursion_deg": np.percentile(torch.rad2deg(mu.quat_error_magnitude(quat,hold_quat)).cpu().numpy(),[50,90]).tolist(),
          "final_angle_deg_p50_p90": np.percentile(np.abs(np.rad2deg(twist.cpu().numpy())), [50,90]).tolist(),
          "final_core_speed_p50_p90": np.percentile(core.body_lin_vel_w[:,u.core_id].norm(dim=-1).cpu().numpy(), [50,90]).tolist(),
          "quiet_pass": int(stable.sum())}
report["final_checks"] = {
    "linear_speed": int((core.body_lin_vel_w[:,u.core_id].norm(dim=-1) < 0.01).sum()),
    "angular_speed": int((core.body_ang_vel_w[:,u.core_id].norm(dim=-1) < 0.15).sum()),
    "twist_aligned": int((mu.wrap_to_pi(twist).abs() < np.radians(5)).sum()),
    "twist_still": int((vel.abs() < 0.15).sum()),
    "supported": int(((pads >= cfg.home_min_pads).all(-1)).sum()),
    "wrist_motion": int((u.cycle_wrist_pos_peak <= cfg.cycle_wrist_pos).sum()),
    "cube_pos": int((u.peak_pos < 0.020).sum()),
    "cube_rot": int((u.peak_rot < np.radians(15)).sum()),
}
for s in ("left", "right"):
    q = u.hands[s].data.joint_pos[:,u.finger_ids[s]]
    tgt = (u.finger_grasp[s] + u._idle_offsets()).clamp(min=u.finger_lower[s], max=u.finger_upper[s])
    report["final_checks"][s+"_finger"] = int(((q[:,cols]-tgt[:,cols]).abs().amax(-1)<cfg.cycle_finger_tol).sum())
report_path = Path("outputs/primitive_cycle") / (Path(a.out).stem + ".json")
report_path.write_text(json.dumps(report, indent=2)+"\n")
print("[ready_cache] diagnostics " + json.dumps(report), flush=True)
if not len(selected):
    raise RuntimeError("No ready grasp passed; inspect release posture before training")
rows = torch.tensor(np.flatnonzero(keep), device=u.device)
out = {k: v[selected].copy() for k, v in data.items()}
out["source_grasp_idx"] = selected.astype(np.int64)
out["cycle_ready"] = np.ones(len(selected), np.float32)
out["cube_pos"] = pos[rows].cpu().numpy()
out["cube_quat"] = quat[rows].cpu().numpy()
out["cube_twist"] = twist[rows].cpu().numpy()
for s in ("left", "right"):
    p, q = u.ctrl[s].ee_pose_w()
    out[f"{s}_ee_pos_act"] = (p[rows] - u.scene.env_origins[rows]).cpu().numpy()
    out[f"{s}_ee_quat_act"] = q[rows].cpu().numpy()
    out[f"{s}_fingers_act"] = u.hands[s].data.joint_pos[rows][:, u.finger_ids[s]].cpu().numpy()
    out[f"{s}_joint_pos_act"] = u.hands[s].data.joint_pos[rows].cpu().numpy()
# Contact/compatibility labels must describe the stored state.
out["cells"] = u.hand_cells()[rows].cpu().numpy()
out["link_touch"] = (u.cube_contacts.forces().sum(1) > 0.05)[rows].cpu().numpy()
np.savez(a.out, **out)
print(f"[ready_cache] {len(selected)}/{len(indices)} shared grasps passed both preparations -> {a.out}", flush=True)
u.close()
app.close()
