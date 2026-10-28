"""Config of the full-solve skill environment (two float-base Sharpa Wave hands + a face-rebinding two-body cube)."""

from __future__ import annotations

import math

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from rubik_lab.assets import RUBIK_CUBE_CFG, sharpa_hand_cfg
from rubik_lab.constants import CUBE_SIZE
from rubik_lab.wave_se_tactile import WaveSETactileCfg


@configclass
class SolveEnvCfg(DirectRLEnvCfg):
    # timing
    physics_hz = 240
    decimation = 6  # 40 Hz policy
    episode_length_s = 3.0
    phase_horizon_s = 3.0  # time scale of the phase observation: the training episode length (train.py --episode_s)
    # spaces (recomputed in the env)
    action_space = 56
    observation_space = 0
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 240,
        gravity=(0.0, 0.0, -9.81),
        render_interval=6,
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=8,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=5 * 2**18,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=1.0, replicate_physics=True)

    contact_model = "rigid"
    wrist_stiffness = 10000.0  # [N/m] wrist translation drives (None = USD 3000); stands in for a stiff arm
    wrist_vel_ff = True  # feed the nominal wrist velocity to the drives
    # "float": the hands on 6 virtual wrist joints; "fr3": on FR3 arms + printed adapters (~/workspace/fr3-sharpa-rubik)
    # placed by fr3_layout (scripts/franka/fr3_layout.py), the same wrist pose targets tracked by arm IK (rubik_lab.fr3_arm)
    arm = "float"
    fr3_layout = "assets_franka/fr3_layout.json"
    fr3_arm_stiffness = (4000.0, 1000.0)  # [Nm/rad] joints 1-4, 5-7
    fr3_arm_collision = True  # False: arm links and adapters without colliders (diagnostic)
    # the robot: URDF (<dir>/<side>/robot.urdf, for the arm IK; "" = the fr3-sharpa-rubik assets) and its physics overlay
    # USD (<dir>/<side>/fr3_sharpa_<side>.usda, scripts/franka/make_fr3_hand_usd.py), both relative to the project
    fr3_urdf_dir = ""
    fr3_usd_dir = "assets_franka"
    wrist_lin_vel_limit = 1.7  # m/s (Franka end-effector)
    wrist_ang_vel_limit = 2.6  # rad/s (Franka joint 7)
    left_hand_cfg: ArticulationCfg = sharpa_hand_cfg("left", contact_model="rigid")
    right_hand_cfg: ArticulationCfg = sharpa_hand_cfg("right", contact_model="rigid")
    cube_cfg: ArticulationCfg = RUBIK_CUBE_CFG.replace(prim_path="/World/envs/env_.*/cube")
    cube_size = CUBE_SIZE
    tactile: WaveSETactileCfg = WaveSETactileCfg()
    link_contacts = True  # per-link cube contact sensors (release of all trick-finger links, grasp cells); costly

    # cube turning resistance, PhysX joint friction (implicit, no low-speed limit cycle): static / dynamic [Nm],
    # viscous [Nm s/rad]. Speedcubes need roughly 0.02-0.1 Nm to start a layer.
    twist_static_friction = 0.03
    twist_dynamic_friction = 0.02
    twist_viscous_friction = 0.0
    use_joint_friction = True  # False: legacy explicit -0.01 tanh(w / 0.05) torque (kept for comparison only)
    detent_torque = 0.0  # Nm peak, magnetic positioning (restoring torque near quarter turns); 0 = off
    detent_width = math.radians(10.0)  # active within this of a quarter turn (peak torque at half the width)

    # ready grip: npz in the grasp-cache format (rubik_lab.tasks.bimanual_twist), relative to the project root
    ready_cache = "cache/g70s_cycle_demo1_v3.npz"
    ready_index = -1  # -1: sample all entries
    ready_sequential = False
    # alternative grasp per hand (grip mode 1, e.g. edge grips for the wrist rolls): {side: npz path}, settled caches
    grip_alt_cache = {}
    # extension [rad] of idle trick fingers in the ready grip (the cache stores the squeezing grasp)
    idle_lift = {"index": 0.30, "pinky": 0.30}
    regrip_servo = True  # sighted: the regripping hand's nominal follows the cube's actual pose
    regrip_power_grip = 0.0  # fraction of the idle lift the holding hand closes again during the other hand's regrip (1.0: the
    # closing index / pinky push the cube ~3 mm down and tip it ~5 deg in the grip; the next flick fails)
    hold_press = 0.15  # extra flexion of the holding thumb / middle (squeeze headroom)

    # skills trained in this env instance and their sampling weights
    skills = ("U", "U'")
    skill_weights = None

    # chaining: successful terminal states feed later episodes' starts
    buffer_size = 0
    buffer_prob = 0.5
    regrip_boost = 4.0
    solo_roll_other_unrolled = True  # R / R' / L / L' start only with the other hand at roll 0 (cube_logic.SkillCompiler)
    flick_both_unrolled = True  # flicks start only with both hands at roll 0 (cube_logic.SkillCompiler)
    # handoff starts: capture_full states recorded at skill starts of real skill chains (scripts/solve/handoff_matrix.py
    # --dump), sampled with start_pool_prob before the chaining buffer; "" = off
    start_pool = ""
    start_pool_prob = 0.4
    tilt_tol = math.radians(10.0)  # completion: cube within this of an axis-aligned orientation

    # residual action ranges around the nominal targets
    act_finger = 0.6  # rad
    # residual ranges (wrist pos m, wrist rot rad, finger rad) of the wrist family (rolls, x, regrips), whatever the
    # global ones above: the wrist policies are trained with these, and a run with other global ranges (flick training,
    # chained execution) must not rescale their actions. None: the global ranges
    wrist_act = (0.005, 0.06, 0.4)
    act_wrist_pos = 0.012  # m
    act_wrist_rot = 0.15  # rad
    action_ema = 0.6
    flick_nominal = "cache/solve_flick_nominal.npz"  # demonstrated trick-finger offsets (scripts: see MEMORY)
    flick_nominal_s = 0.5
    # old bimanual_twist flick checkpoint (Rubik-Flick-*-Sharpa-v0) as the nominal finger targets of U / U' / D / D'
    # (solve.policies.LegacyFlickPolicy: push, then a scripted release); the residual policy acts on top. "" = off
    flick_legacy_ckpt = ""
    flick_legacy_kw = {}
    flick_trick_scale = 2.0  # trick finger joints of a flick: 2 x act_finger
    flick_other_scale = 0.3  # every other dimension during a flick

    # success (frozen engineering thresholds of the cycle contract, plus whole-cube goal for x)
    layer_tol = math.radians(5.0)
    layer_speed_tol = 0.25  # rad/s
    cube_rot_goal_tol = math.radians(8.0)  # x / x': cube orientation vs goal
    cube_pos_bound = 0.015  # m, cube center from nominal over the skill
    cube_rot_bound = math.radians(15.0)  # deviation from the logical goal orientation (beyond the intended motion)
    regrip_pos_bound = 0.03  # m, the same during a regrip (single-hand sway, restored by the close)
    regrip_rot_bound = math.radians(30.0)
    wrist_track_tol = (0.008, math.radians(8.0))  # actual wrist vs nominal end pose at completion
    finger_ready_tol = 0.2  # rad, max joint error of the active hand(s) to the settled ready posture
    regrip_finger_tol = 0.3  # rad: a re-closed hand settles on slightly different contacts
    finger_speed_tol = 1.0  # rad/s
    release_force = 0.5  # N, flick finger normal force on the layer at completion
    min_pads = 1  # elastomer pads touching the cube, per holding hand
    pad_threshold = 0.05  # N
    # handoff-ready end state, the precondition every skill may assume: success also needs the cube at its ready-grip
    # pose (modulo the cube symmetry) within handoff_pos_tol / handoff_rot_tol, the layer aligned, both wrists on their
    # nominal end pose and every finger of both hands near the ready posture. The cube tolerances anneal from
    # handoff_start to the final values over handoff_anneal_steps env steps (0: final values from the start)
    handoff_required = False
    handoff_pos_tol = 0.002  # m
    handoff_rot_tol = math.radians(3.0)
    handoff_layer_tol = math.radians(3.0)
    handoff_wrist_tol = (0.003, math.radians(3.0))
    handoff_finger_tol = 0.25  # rad, max joint error to the settled ready posture, both hands
    handoff_start = (0.006, math.radians(8.0))
    handoff_anneal_steps = 0
    stable_s = 0.25  # consecutive completion window
    stable_pos = 0.002
    stable_rot = math.radians(2.0)
    stable_layer = math.radians(1.5)  # layer angle excursion over the window
    drop_height = 0.10  # below the nominal center: dropped

    # rewards: per-step penalties stay far below the termination penalties (else ending early pays)
    rew_progress = 20.0
    rew_success = 100.0
    rew_hold_ready = 0.5
    rew_cube_pos = -5.0
    rew_cube_rot = -1.0
    rew_finger_return = -0.5
    rew_wrist_track = -2.0
    rew_action_rate = -0.002
    rew_time = -0.02
    rew_handoff = 0.0  # per step, times (cube offset / pos tol + tilt / rot tol) at the final tolerances (each ratio <= 5)
    rew_trick_contact = 0.3  # flicks: the trick pad touches the moving layer while it is still far from the target
    rew_drop = -100.0
    rew_violation = -100.0
    terminate_on_success = True
    success_linger_s = 0.0  # with terminate_on_success=False: end the episode this long after success

    # chain training: an episode runs random skill chains (chain_moves compiled by cube_logic.SkillCompiler, regrips
    # included) back to back, the next skill starting on the step the last one succeeds (no settle, no reset), so every
    # skill trains from the end states its predecessors really leave. A failure (drop, violation, skill time-out) ends
    # the episode (restart from the ready grip); an episode time-out continues the chain (truncation)
    chain_train = False
    chain_moves = ("U", "U'", "R", "R'", "L", "L'", "x", "x'")
    chain_bank = 2048  # precompiled sequences of chain_seq_moves random moves (both hands unrolled at both ends)
    chain_seq_moves = 100
    # skill sequences of real solves instead (scripts/solve/solve_bank.py, json {"seqs": [[skill names]]}), relative to
    # the project root; "" = random chain_moves
    chain_bank_file = ""
    chain_skill_timeout = 3.5  # s
    # fraction of chain skills that advance as soon as their completion window opens, before the success (the states after
    # a premature completion detector, as in blind execution with --blind_continue)
    chain_early_advance_p = 0.0
    chain_timeout_advance_p = 0.0  # fraction of skill time-outs that start the next skill instead of failing (as execution's --timeout_continue)
    chain_restart_prob = 0.0  # fraction of episode time-outs that restart from the ready grip instead
    # s of holding the freshly written ready grip (zero action) before a restarted chain's first skill: right after the
    # reset the contacts are not settled yet (in eval chains the first skill succeeded 57% without, mid-chain 96%)
    chain_linger_s = 0.0  # hold a completed skill this long before the next one starts
    chain_warmup_s = 0.5
    obs_next_skill = False  # append the next skill's one-hot (zeros when unknown) to both observation groups
    # blind student control (scripts/solve/distill_chain.py): the action sets the targets directly around the BLIND
    # nominal (no sighted regrip servo, no legacy flick policy - both use the true cube state): wrist position / rotation
    # / finger = blind nominal + action x student_scale (m, rad, rad), no smoothing. Every step the env also computes the
    # sighted teacher's targets (env.teacher_raw: the teacher's raw action) in that action space (env.teacher_free, the
    # distillation label) and executes it where env.teacher_mask (DAgger mixing)
    student_mode = False
    # blind state estimation: every controller (teacher observation, regrip servo, legacy flick) acts on env.est_cube
    # (set each step by solve.estimator) instead of the simulator's cube state, which then only scores
    cube_est = False
    # the skill's reference orientation (cube_ref / cube_goal; the estimate's rotation is relative to it) from the estimate
    # instead of the simulator: by default every skill start re-anchors the blind orientation on the truth (a leak)
    goal_from_perceived = False
    cube_est_ckpt = ""  # estimator checkpoint run inside the env (sets cube_est; obs "critic": the simulator's state)
    # sim2real (all off by default: older checkpoints and results unchanged; the observation layout never changes).
    # Observations as the real robot measures them: finger joint angles with white noise and a per-episode encoder
    # offset, finger joint velocities by finite difference of those angles over the control step (no simulator
    # velocities), per-pad F/T gain / offset and TacMap intensity gain per episode.
    obs_real = False
    joint_pos_noise = 0.005  # rad, white
    joint_pos_bias = 0.02  # rad, per episode, uniform +-
    tac_gain = (0.8, 1.2)  # per pad and episode: F/T scale
    tac_bias = 0.1  # N, per pad and episode, uniform +- on the forces
    tac_img_gain = (0.8, 1.2)  # per pad and episode: TacMap intensity scale
    # dynamics randomization, sampled per env at every full reset (pad friction / cube mass: once at startup) and the
    # command latency: new joint targets reach the drives after a per-env delay of physics steps (1/240 s each)
    dr = False
    dr_pad_friction = (0.6, 1.4)  # pad material friction (nominal 1.0)
    dr_cube_mass = (0.8, 1.25)  # scale of both cube bodies' masses / inertias
    dr_finger_gain = (0.75, 1.3)  # scale of the finger actuator stiffness, damping sampled independently (inert before 10-27)
    dr_arm_gain = (0.8, 1.25)  # scale of FR3 joint stiffness / damping
    dr_twist_friction = (0.5, 2.0)  # scale of the layer joint friction
    dr_detent = (0.6, 1.5)  # scale of the magnetic detent torque
    dr_act_delay = (0, 5)  # physics steps, inclusive (0 - 21 ms)
    # hand command latency of any length (the real Wave SE in POSITION mode with interpolation lags ~100 ms): finger
    # targets pass through a per-env ring buffer of physics steps; dr_act_delay then delays only the wrist / arm targets.
    # dr_hand_delay_ramp: control steps over which the sampled range grows from 0 to the full one (a curriculum)
    dr_hand_delay = None  # (lo, hi) physics steps, inclusive, e.g. (14, 29) = 58 - 121 ms
    dr_hand_delay_ramp = 0
    # fraction of envs (fixed per env at startup) that get obs_real / dr; the others run nominal (no noise, offsets,
    # gains or latency): one estimator trained on both
    dr_frac = 1.0
    # random external wrench pulses on the cube core during skills (bumps and slips the finger joints barely see):
    # (pulses per s per env, force max N, torque max Nm, (min, max) duration s); None = off
    dr_slip = None  # e.g. (0.3, 0.8, 0.015, (0.03, 0.1))
    # delivered tactile (pad F/T and images; needs obs_real): a new sample each control step with probability tac_rate_p
    # (0.75 ~ the real 30 Hz at the 40 Hz control rate; held otherwise), tac_lag control steps (25 ms each) old
    tac_rate_p = 1.0
    tac_lag = 0
    # camera cube perception (a measurement model, no rendering): every controller acts on env.perceived() = the cube
    # pose / free-layer angle a vision tracker would report: frames refreshed with probability vis_rate_p per control
    # step (0.75 ~ 30 Hz), vis_lag control steps old, white noise per frame and a per-episode bias (pos m, rot deg,
    # layer deg), occlusion bursts (two-state chain per control step: p enter, p exit), tracked by an alpha-beta filter.
    # env.success_vis: the completion criteria on measurements (execute --trigger vision): the cube ones on the tracker,
    # holding / release on the delivered pad forces
    cube_vis = False
    vis_rate_p = 0.75
    vis_lag = 2
    vis_noise = (0.0015, 1.5, 2.0)
    vis_bias = (0.001, 1.0, 1.0)
    vis_occlusion = (0.04, 0.17)  # ~20% of the control steps occluded, bursts ~150 ms
    vis_filter = (0.5, 0.1)  # tracker alpha / beta (position, orientation, layer angle and their rates)
    vis_quiet = (0.004, 4.0, 4.0)  # camera completion: excursion over the window (m, deg, deg); raw noise exceeds stable_*
    vis_max_age = 0.1  # s, camera completion needs a frame at most this old
    vis_touch = (0.5, 0.8)  # N, delivered pad force: holding above / flick finger released below
    vis_control = True  # False: the controllers act on the truth, only the completion (success_vis) uses the camera
    # real-cube misalignment (None = the idealized cube: every re-binding snaps the free layer to the nearest quarter, as
    # if the corners always cut). Set [deg]: a re-binding to the same face keeps the residual; a layer left more than this
    # off a quarter stays out of line (one slot) and locks the free layer of any orthogonal face until the skill binds that
    # layer again; a cube with a layer out of line is not solved (env.misaligned)
    cube_jam_deg = None
    # separate middle layer (assets/rubik_cube/rubik_cube_3body.usda: core = bottom layer, mid on mid_joint, top on
    # twist_joint relative to mid, same friction / detents): the top layer can drag the middle one along. A middle layer
    # turned more than cube_jam_deg (default 15) off its binding stays out of line (env.misaligned, jams orthogonal turns)
    cube_3body = False
    mid_friction_scale = 1.0  # mid_joint friction relative to the twist joint's
    # K earlier actions (beyond the last one) appended at the very end of the teacher observation, so a policy can reason
    # about the commands still in flight under a long latency; older checkpoints extend by zero columns (pad_ckpt.py)
    obs_act_hist = 0
    student_scale = (0.05, 0.4, 2.0)  # 0.02 m clipped 1.85% of the wrist labels (regrip servo) and halved the chain length
