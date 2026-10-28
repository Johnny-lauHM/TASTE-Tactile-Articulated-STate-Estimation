"""Versioned complete-cycle tasks; historical Flick tasks remain unchanged."""
import math
from isaaclab.utils import configclass
from .env_cfg import FlickUDRsEnvCfg
from .agents.rsl_rl_ppo_cfg import FlickPPORunnerCfg


@configclass
class CycleUEnvCfg(FlickUDRsEnvCfg):
    moves = ("U",)
    # Both directions have tables so no-reset same-face evaluation can switch them.
    flick_moves = ("U", "U'")
    episode_length_s = 6.0
    notch_torque = 0.0
    detent_torque = 0.0
    wrist_prior = False
    gravity_curriculum = False
    move_success_tol = math.radians(5.0)
    move_success_vel = 0.25
    move_core_ang_speed = 1.0  # existing layer-arrival guard, now explicit in saved config
    cycle_contract_version = 3
    cycle_stability_mode = "pose_window"
    cycle_stable_pos = 0.002
    cycle_stable_rot = math.radians(2.0)
    cycle_stable_s = 0.25
    cycle_command_slew_rad_s = 0.0  # optional encoder-target slew limit; zero preserves historical control
    cycle_idle_lift = 0.30  # extension of idle index/pinky, rad
    cycle_finger_tol = 0.20  # active finger max joint deviation from ready target, rad
    cycle_finger_speed = 1.0  # rad/s
    cycle_release_force = 0.5  # N, sum of NORMAL contact-force norms over active finger links; friction excluded
    cycle_wrist_pos = 0.010  # actual wrist, relative to original settled grasp
    cycle_wrist_rot = math.radians(10.0)
    cycle_cube_pos = 0.015
    cycle_cube_rot = math.radians(15.0)
    cycle_cube_speed = 0.02
    cycle_core_ang_speed = 0.25
    cycle_return_reward = -0.5
    cycle_return_progress = 10.0
    cycle_violation_penalty = -100.0
    cycle_hold_reward = 1.0
    # Evaluation alone can suppress success auto-reset for a same-face sequence.
    cycle_terminate_on_success = True
    # No whole-hand layer-contact reward: thumb/middle must hold the core while
    # the index turns the layer. The legacy term confuses these two roles.
    rew_move_done = 0.0  # no perpetual payment for holding the layer without retracting
    rew_pad_progress = 0.0  # optional training-only reward; does not change acceptance
    rew_contact_selective = 0.0
    rew_tip_dist = 0.0
    rew_drop = -100.0


@configclass
class CycleUpEnvCfg(CycleUEnvCfg):
    moves = ("U'",)


@configclass
class CyclePPORunnerCfg(FlickPPORunnerCfg):
    experiment_name = "rubik_primitive_cycle"
    max_iterations = 2000
    save_interval = 100
