"""Bimanual Rubik's cube move primitives with two float-base Sharpa Wave hands, EE-pose actions.

Home grip (human-like, env frame z up): the hands hold the cube from the sides, left hand on +y and right hand on -y.
The palms face the cube, the thumbs are on the front face (-x), and the fingers point up-and-back so they can wrap
the back face (+x). Each episode executes one primitive move (see rubik_lab.moves), conditioned on a one-hot move id.

Wrist frame of both hands: +z along fingers, +x palm normal (fingers curl towards +x),
right thumb on wrist +y, left thumb on wrist -y.
"""

import math

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from rubik_lab.assets import FINGER_JOINTS, RUBIK_CUBE_CFG, sharpa_hand_cfg
from rubik_lab.wave_se_tactile import WaveSETactileCfg

from rubik_lab.constants import CUBE_SIZE  # noqa: E402  (0.070; see constants.py)
CUBE_CENTER = (0.0, 0.0, 0.30)
# palm collision (<side>_hand_C_MC) in the wrist frame: surface at x=+0.025, spans z in [0.029, 0.115]
PALM_SURFACE_X = 0.025


def _quat_from_axes(x, y, z):
    """(w, x, y, z) of the rotation whose matrix columns are the given orthonormal axes."""
    m = [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]]
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = 2.0 * math.sqrt(tr + 1.0)
        return (0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
    i = max(range(3), key=lambda k: m[k][k])
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * math.sqrt(1.0 + m[i][i] - m[j][j] - m[k][k])
    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = (m[k][j] - m[j][k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (m[j][i] + m[i][j]) / s
    q[1 + k] = (m[k][i] + m[i][k]) / s
    return tuple(q)


def home_grip_pose(side: str, tilt_deg: float, palm_gap: float, cube_along_fingers: float):
    """Wrist pose (pos, quat) of the side grip.

    tilt_deg: finger direction tilted from +z towards +x (back face); thumbs end up on the front face.
    palm_gap: gap between palm surface and the cube side face.
    cube_along_fingers: cube center position along the finger axis, measured from the wrist.
    """
    a = math.radians(tilt_deg)
    z = (math.sin(a), 0.0, math.cos(a))  # fingers
    x = (0.0, -1.0, 0.0) if side == "left" else (0.0, 1.0, 0.0)  # palm normal towards the cube
    y = (z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0])  # y = z cross x
    d = CUBE_SIZE / 2 + PALM_SURFACE_X + palm_gap
    pos = tuple(CUBE_CENTER[i] - d * x[i] - cube_along_fingers * z[i] for i in range(3))
    return pos, _quat_from_axes(x, y, z)


@configclass
class BimanualTwistEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0  # twist + return to the home grip
    decimation = 6  # 240 Hz physics -> 40 Hz policy
    # action mode, per hand:
    #   "residual": [dpos(3), drot axis-angle(3), dfinger(22)] in [-1, 1] -> 28, an OFFSET from this episode's grasp
    #               (action 0 = hold the grasp; return-to-grasp = action back to 0). The wrist rotation pivots about
    #               the cube center. Actions are EMA-smoothed (action_ema). Wrist part masked per move (wrist_faces).
    #   "relative": same layout, accumulated on the previous target (exploration noise random-walks the fingers open)
    #   "fingers" : [dfinger(22)] -> 22, wrists never move
    #   "absolute": [pos(3), quat wxyz(4), finger(22) in [-1, 1] mapped to joint limits] -> 29 (scripts / teleop)
    ee_action_mode = "residual"
    # faces whose primitive is a wrist roll of the own-side hand (palms cover R/L, so fingers can't flick them);
    # for every other move both wrists are held at the grasp pose (wrist actions masked to 0)
    wrist_faces = ("R", "L")
    # moves executed as human finger tricks from the human home grip (rubik_lab.flick); empty = old U orbit
    flick_moves = ()
    flick_lift = 0.45  # [rad] "flick" action mode: extension of the fingers lifted off the moving layer
    rew_calm_done = 0.0  # bonus at primitive_done scaled by (1 - max(peak_rot/calm_rot_deg, peak_pos/calm_pos_mm)): rewards a cube that stays put
    calm_rot_deg = 10.0
    calm_pos_mm = 10.0
    rew_push_force = 0.0  # [1/N] flick mode: penalty on the trick pad's force on the moving layer above push_force_free
    push_force_free = 4.0  # [N]
    penalty_warmup_steps = 0  # env steps over which the cube_pos/cube_rot penalties ramp from penalty_warmup_min to 1 (0 = off)
    penalty_warmup_min = 0.2
    flick_hold_press = 0.0  # [rad] "flick" action mode: extra flexion of both thumbs and middles (grip force headroom)
    flick_reach_min = 40.0  # [deg] finger tricks sample grasps whose offline planner reach is at least this (cache field flick_reach)
    ee_pos_limit = 0.02  # EE target may move at most this far from its grasp target [m] (roll is unrestricted)
    action_space = 56  # recomputed in the env from ee_action_mode
    observation_space = 0  # recomputed in the env
    state_space = 0
    clip_actions = 1.0
    clip_obs = 5.0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 240,
        gravity=(0.0, 0.0, -9.81),  # overwritten by the gravity curriculum while training (see below)
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=1.0, replicate_physics=True)

    # assets. Each float base is placed at its default wrist pose (see __post_init__), so the virtual joints start
    # at 0 and the roll-pitch-yaw chain stays away from its pitch=+-90deg singularity. A layer twist by the own-side
    # hand is a rotation about the palm normal = the first (roll) joint.
    left_hand_cfg: ArticulationCfg = sharpa_hand_cfg("left")
    right_hand_cfg: ArticulationCfg = sharpa_hand_cfg("right")
    cube_cfg: ArticulationCfg = RUBIK_CUBE_CFG.replace(
        init_state=ArticulationCfg.InitialStateCfg(pos=CUBE_CENTER, joint_pos={"twist_joint": 0.0})
    )
    cube_size = CUBE_SIZE

    # contact model: "tacmap" (official sharpa-tacmap: compliant pads, needed for TacMap images) or "rigid" (sharpa-rl-lab)
    wrist_stiffness: float | None = None  # [N/m] float-base x/y/z drive stiffness (None = USD 3000); cached squeeze is preserved
    contact_model = "rigid"  # the real Wave SE pads are nearly rigid (user); tactile emulation still uses the contact forces
    # Sharpa Wave SE tactile, on-board mode: 60x60 image @ 30 fps + 6-D F/T per pad, 20 ms latency (rubik_lab.wave_se_tactile)
    # images are off by default (ray casting is expensive); turn on for completion-detector data collection
    tactile: WaveSETactileCfg = WaveSETactileCfg()
    # ground-truth link-level contacts with the cube (rubik_lab.cube_contacts): grasp selection, rewards, labels
    link_contacts = True
    tactile_force_scale = 0.2  # obs = F [N] * scale
    tactile_moment_scale = 20.0  # obs = M [Nm] * scale
    tactile_contact_threshold = 0.05  # N

    # EE = wrist link (later: the arm flange). Home grip parameters -> default EE poses (env frame), computed by
    # home_grip_pose() when the env is created; the float bases are placed there too.
    ee_body = None  # None -> "<side>_hand_wrist"
    grip_tilt_deg = 40.0
    grip_palm_gap = 0.003
    grip_cube_along_fingers = 0.12  # cube center along the fingers from the wrist; swept 0.10-0.14 in gen_grasp_cache
    ee_init_pos_noise = 0.003

    # initial state: sampled from a grasp cache (scripts/gen_grasp_cache.py) if it exists, else open hands
    grasp_cache_path = "cache/home_grip_AB_teach.npz"  # relative to the project root (A/B role grasps, teacher-curated)
    teach_rate_min = 0.5  # per primitive, sample grasps the scripted teacher solved at least this often (0 = off)
    finger_init = [0.0] * len(FINGER_JOINTS)
    finger_init_noise = 0.05
    cube_init_pos_noise = 0.002

    # action scaling
    residual_pos_range = 0.06  # m   (residual: |dpos| <= this; U grip shift, following the cube, x regrip to the paired grasp)
    residual_rot_range = 2.0  # rad (residual: wrist roll about the cube center, 90 deg twist = 0.79)
    residual_finger_range = 1.0  # rad (residual: finger offset from the grasp, clamped to the joint limits)
    action_ema = 0.5  # residual: a_smooth = ema * a + (1 - ema) * a_smooth_prev
    # nominal wrist-roll prior for wrist moves (residual mode): the twisting hand rolls about the face axis towards the
    # move angle (start after roll_delay, over roll_time), and back to 0 once the layer reached the target; the policy
    # adds its residual on top (finger placement, release / regrip). Exploration alone never found the 90 deg roll (v2).
    wrist_prior = True
    roll_delay = 0.5  # s
    roll_time = 1.5  # s
    ee_pos_scale = 0.005  # m per policy step (relative)
    ee_rot_scale = 0.05  # rad per policy step (relative)
    finger_scale = 1 / 24  # rad per policy step (relative), same as the official Sharpa in-hand task
    ee_workspace = ((-0.30, -0.30, 0.05), (0.30, 0.30, 0.60))  # clamp for EE target positions (env frame)
    dof_limits_scale = 0.95

    # task: one primitive per episode, sampled from `moves` (names from rubik_lab.moves.ALL_MOVES)
    # U/D/F/B: finger flicks with fixed wrists; R/L: wrist roll. <U,D,F,B> alone reaches 1/2048 of the cube states,
    # adding R makes it complete. Old Pochmann's T/Y/Ra-perms only use R, U, F.
    moves = ("U", "U'", "R", "R'")
    move_success_tol = math.radians(8.0)  # layer within this of the target ...
    rot_success_tol = math.radians(12.0)  # whole-cube rotations (x): the compliant edge grip lags ~10 deg, no detent
    move_success_vel = 0.5  # ... and slower than this [rad/s] -> move "done" (the completion label)
    # primitive done (= ready for the next move): layer done, both EEs back at their grasp pose, both hands touching
    home_pos_tol = 0.01  # m
    home_rot_tol = math.radians(10.0)
    home_min_pads = 1  # elastomer pads in contact with the cube, per hand
    # passive cube mechanics: detent torque -k*sin(4*theta) pulls the layer to the nearest 90 deg ("click")
    detent_torque = 0.0  # Nm; a real cube has no snap-to-90 (user): the finger has to turn the layer all the way
    notch_torque = 0.03  # [Nm] alignment notch (peak) within +-notch_width_deg of 90 deg multiples; 0 = off. A real cube's click: it
    notch_width_deg = 8.0  # holds the layer during release/regrip but never pulls a half-turned layer
    # Coulomb turning resistance: -twist_friction_torque * tanh(omega / 0.05). A real 3x3 needs ~0.02-0.1 Nm to turn; with
    # less, a cube held by one layer only (the right hand on the R layer) flops under its own weight
    twist_friction_torque = 0.01  # Nm (speedcube; finger tricks produce ~0.05 Nm at most)
    cube_drop_height = CUBE_CENTER[2] - 0.10

    # reward scales
    # gravity curriculum (as in the official sharpa-rl-lab): start low, +gravity_step whenever the moving-average drop
    # rate per step is below gravity_drop_thresh, up to 9.81. Turn off for play / data collection (real gravity).
    gravity_curriculum = True
    gravity_start = 0.5  # m/s^2
    gravity_step = 0.05
    gravity_drop_thresh = 0.002
    gravity_warmup_steps = 1000  # policy steps before the curriculum starts

    rew_twist_progress = 20.0
    rew_move_done = 5.0
    # shaping is kept small (~-0.15/step at the start): with gamma 0.99 (~100-step horizon) a per-step cost of ~-0.9
    # made dropping the cube (-20, episode ends) the best strategy in v1 (episode length collapsed to ~25 steps)
    rew_tip_dist = -0.3
    rew_cube_pos = -1.0
    rew_cube_rot = -0.2
    rew_action_rate = -0.002
    rew_drop = -50.0
    rew_return = -1.0  # after the layer is done: EE distance to its grasp pose (m * 10 + rad)
    rew_primitive_done = 100.0  # terminal bonus, ends the episode
    rew_layer_reached = 30.0  # one-time bonus the first time the layer reaches the target (before returning home)
    # layer-selective fingertip contacts (from the per-layer contact ground truth): twisting hand's pads on the moving
    # layer only, holding hand's pads on the core only; finger-trick moves: pads on the core + any pad on the layer.
    # The wrist-roll prior alone drags the whole cube (layer moved ~10 deg relative to the core): hands span all layers.
    rew_contact_selective = 0.1  # per correct pad, -same per wrong pad (10 pads)


# ---------------------------------------------------------------------- single-finger RL ("flick" action mode)
# The grip is frozen (per-move human grips, 70 mm cube), the policy drives the trick finger's 4 joints only; one task
# per grip family (the grasp cache is per family). Success = the layer turned 90 deg and settled, cube in place.
def _flick_cfg(name, moves, cache):
    _moves = tuple(moves)

    @configclass
    class _Cfg(BimanualTwistEnvCfg):
        ee_action_mode = "flick"
        flick_moves = _moves
        moves = _moves
        grasp_cache_path = cache
        wrist_prior = False
        gravity_curriculum = False
        ee_init_pos_noise = 0.0
        episode_length_s = 6.0
        rew_cube_pos = -30.0  # keep the cube in place (user): stronger than the twist task's default
        rew_cube_rot = -4.0
        rew_calm_done = 100.0  # as much as the completion bonus itself when the cube does not move at all
        rew_push_force = -0.03  # 25 N shove -> -0.63/step (the old policies pushed with 10-25 N and turned the cube 14 deg)
        penalty_warmup_steps = 7200  # ~300 PPO iterations (24 steps each): exploration pushes rotate the cube ~12 deg
        wrist_stiffness = None  # 20000 N/m made the hold itself unstable (45% drops in the static hold test); see MEMORY
        flick_hold_press = 0.15  # thumbs/middles squeeze ~2x the cached grip force (hold test: 0.8 mm drift, 0 drops)

    _Cfg.__name__ = name
    return _Cfg


FlickUEnvCfg = _flick_cfg("FlickUEnvCfg", ("U", "U'"), "cache/g70_U.npz")
FlickDEnvCfg = _flick_cfg("FlickDEnvCfg", ("D", "D'"), "cache/g70_D.npz")
FlickFBEnvCfg = _flick_cfg("FlickFBEnvCfg", ("F", "F'", "B", "B'"), "cache/g70_FB.npz")
# swept-volume-clean caches (scripts/build_flick_caches.sh): holding links clear the turning layer's corners
FlickUsEnvCfg = _flick_cfg("FlickUsEnvCfg", ("U", "U'"), "cache/g70s_U.npz")
FlickDsEnvCfg = _flick_cfg("FlickDsEnvCfg", ("D", "D'"), "cache/g70s_D.npz")
FlickFsEnvCfg = _flick_cfg("FlickFsEnvCfg", ("F", "F'"), "cache/g70s_F.npz")
FlickBsEnvCfg = _flick_cfg("FlickBsEnvCfg", ("B", "B'"), "cache/g70s_F.npz")  # same grip as F: ring finger does B
# one grip per face pair (user: wrists stay put between moves): holding pads on the middle row / column only
FlickUDsEnvCfg = _flick_cfg("FlickUDsEnvCfg", ("U", "U'", "D", "D'"), "cache/g70s_UD.npz")
FlickFBsEnvCfg = _flick_cfg("FlickFBsEnvCfg", ("F", "F'", "B", "B'"), "cache/g70s_FB.npz")
# side pinch with the right hand on the R column only: U/D flicks here, R by the right wrist (scripted roll / later RL)
FlickUDRsEnvCfg = _flick_cfg("FlickUDRsEnvCfg", ("U", "U'", "D", "D'"), "cache/g70s_UDR.npz")
