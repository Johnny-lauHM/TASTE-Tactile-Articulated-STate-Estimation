"""Restore serializable physical/control settings saved beside a cycle checkpoint.

Legacy checkpoints without env.json retain the caller's explicitly selected task.
Never deserialize pickle/YAML objects just to recover evaluation settings.
"""
import json
from pathlib import Path

CONTROL_KEYS = {
    "grasp_cache_path", "contact_model", "notch_torque", "notch_width_deg",
    "detent_torque", "twist_friction_torque", "flick_hold_press", "flick_lift",
    "action_ema", "residual_finger_range", "ee_init_pos_noise", "wrist_stiffness",
    "move_success_tol", "move_success_vel", "move_core_ang_speed", "home_min_pads", "home_pos_tol", "home_rot_tol",
    "episode_length_s", "flick_reach_min",
}


def restore_checkpoint_settings(cfg, checkpoint):
    if not checkpoint:
        return None
    path = Path(checkpoint).resolve().parent / "env.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if hasattr(cfg, "cycle_stability_mode") and "cycle_contract_version" not in data:
        cfg.cycle_contract_version = 2
        cfg.cycle_stability_mode = "velocity"
    for key, value in data.items():
        if key in CONTROL_KEYS or key.startswith("cycle_"):
            if not hasattr(cfg, key):
                raise ValueError(f"Checkpoint requires unsupported setting {key}")
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise ValueError(f"Unexpected non-scalar config value: {key}")
            setattr(cfg, key, value)
    return str(path)


def validate_sequence_policy_settings(cfg, checkpoint):
    """A secondary policy shares the live primary environment, never its own reset."""
    if not checkpoint:
        return None
    path=Path(checkpoint).resolve().parent/'env.json'
    if not path.exists():
        return None  # legacy teacher transfer: metadata unavailable, report that fact
    data=json.loads(path.read_text())
    excluded={'grasp_cache_path','episode_length_s','cycle_terminate_on_success'}
    mismatch={}
    data={'cycle_command_slew_rad_s':0.0,**data}
    for key,value in data.items():
        if key in excluded or not (key in CONTROL_KEYS or key.startswith('cycle_')):
            continue
        current=getattr(cfg,key,0.0 if key=="cycle_command_slew_rad_s" else None)
        if current!=value:mismatch[key]={'live':current,'secondary_checkpoint':value}
    if mismatch:
        raise ValueError('Incompatible sequence policy settings: '+json.dumps(mismatch,sort_keys=True))
    return str(path)
