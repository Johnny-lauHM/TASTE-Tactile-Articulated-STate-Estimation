"""Physical skill table and the scripted nominal motion each skill's policy acts on (pure python / torch).

A skill policy outputs a residual on top of the nominal wrist / finger targets below. Nominals are deliberately
simple (they encode WHAT moves, not how to keep contact): rolls rotate a wrist about the cube's roll axis through the
nominal cube center, regrips open, back off, unroll, approach and close; flicks hold the ready grip and leave the
trick finger to the policy.
"""

from __future__ import annotations

import math

SKILLS = ["hold", "U", "U'", "D", "D'", "R", "R'", "L", "L'", "x", "x'", "regrip_left", "regrip_right"]
SKILL_INDEX = {s: i for i, s in enumerate(SKILLS)}
HANDS = ("left", "right")

# face whose layer is free during the skill (two-body binding). Rolls and regrips bind the slab of the rolling hand.
BIND_FACE = {"hold": "U", "U": "U", "U'": "U", "D": "D", "D'": "D", "R": "R", "R'": "R", "L": "L", "L'": "L",
             "x": "R", "x'": "R", "regrip_left": "L", "regrip_right": "R"}  # fmt: skip
# layer target [rad] about the bound face's outward normal (clockwise X = -90 deg)
LAYER_TARGET = {"U": -0.5, "U'": 0.5, "D": -0.5, "D'": 0.5, "R": -0.5, "R'": 0.5, "L": -0.5, "L'": 0.5}
LAYER_TARGET = {k: v * math.pi for k, v in LAYER_TARGET.items()}
# hand roll change in quarter turns about the hand's own face normal (see cube_logic.SkillCompiler)
ROLL_DELTA = {"R": {"right": -1}, "R'": {"right": 1}, "L": {"left": -1}, "L'": {"left": 1},
              "x": {"right": -1, "left": 1}, "x'": {"right": 1, "left": -1}}  # fmt: skip
# whole-cube rotation: quarter turns about the R normal (x turns like R)
CUBE_QUARTERS_ABOUT_R = {"x": -1, "x'": 1}
# finger tricks: (hand, finger) as in rubik_lab.flick.FLICKS
FLICK = {"U": ("right", "index"), "U'": ("left", "index"), "D": ("left", "ring"), "D'": ("right", "ring")}
REGRIP_HAND = {"regrip_left": "left", "regrip_right": "right"}
# digits of a rolled hand resting on the U layer (scripts/solve/grip_geometry.py): a quarter roll carries the thumb
# (front) or the middle / ring (back) onto the top face, where they block the other hand's flick
ROLLED_ON_U = {("right", -1): ("thumb",), ("right", 1): ("middle", "ring"), ("left", -1): ("middle", "ring"), ("left", 1): ("thumb",)}
# rad of extension of those digits while the other hand flicks (0: flicks only with both hands unrolled); they lift
# over FLICK_PARTNER_RAMP s and re-close as the layer nears its target: fully lifted until CLOSE[0] + CLOSE[1] rad
# left, closed at CLOSE[0]
FLICK_PARTNER_LIFT = 0.0
# (partner hand, roll) states a flick may start from with that partner rolled (the rest regrip it first): in chains
# R -> U' 94%, L -> U 87%, R' -> U' 80%, L' -> U 44% (the lifted left thumb)
FLICK_PARTNER_OK = {("right", -1), ("left", -1), ("right", 1)}
FLICK_PARTNER_RAMP = 0.15
FLICK_PARTNER_CLOSE = (0.03, 0.1)  # (0.1, 0.3): L' -> U 23%, the thumb re-closes onto the turning layer; 67% here

# D / D' from the ready grip: these digits of both (unrolled) hands wrap onto the D row and block the layer (the right
# middle's middle / distal phalanges press ~4 N on it, scripts/solve/diag_flick.py): rad of extension during a D flick,
# same ramp / re-close as the partner lift. 0: off
FLICK_D_LIFT = 0.0
ON_D = ("middle",)

# hand face normals in the env frame: the right hand sits on R (-y), the left on L (+y)
HAND_NORMAL = {"left": (0.0, 1.0, 0.0), "right": (0.0, -1.0, 0.0)}


def smoothstep(x):
    x = x.clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# nominal timing [s]
# a quarter roll with a smoothstep profile peaks at 1.5 x its mean speed: 0.95 s keeps it under the 2.6 rad/s wrist
# (Franka joint 7) limit of the virtual wrist joints
ROLL_DELAY, ROLL_TIME = 0.05, 0.95
# a single-hand roll shifts the rolling wrist outward along its face normal (m) while the layer turns, so its pads
# slide away from the column boundary instead of dragging the core; ramped in/out over ROLL_SHIFT_RAMP (s)
ROLL_SHIFT, ROLL_SHIFT_RAMP = 0.0, 0.15
ROLL_RING_LIFT = 0.0  # rad extension of the rolling hand's ring flexion joints (same ramp as the shift)
# rad of extra thumb / middle / ring flexion of the HOLDING hand during a single-hand roll (same ramp as the shift): in
# chains R / R' right after regrip_left (the left hand just re-closed, holding) succeed 89-92% vs 98% otherwise, with the
# left thumb pressing ~2 N less at the start of the failures
ROLL_HOLD_PRESS = 0.0
REGRIP_PHASES = (0.15, 0.15, 0.95, 0.2, 0.45)  # open, back off, unroll, approach, close (slow: soft contact)
REGRIP_HOLD_PRESS = 0.4  # rad extra flexion of the holding hand's thumb / middle / ring while the other hand is off (0.15: the cube slips
# ~5 mm in the grip and the next flick fails; 0.2-0.5 swept with the power grip off, 0.4 best for the flicks after it)
REGRIP_BACKOFF = 0.022  # m, along the hand's face normal (away from the cube)
REGRIP_OPEN = 0.6  # rad of extension of every flexion joint of the regripping hand
REGRIP_RESTORE = 0.35  # s after the close: the servoed hand returns to its ready pose, carrying the cube back upright
# the holding hand loosens (rad of extension of its flexion joints) while the regripped hand carries the cube back, so
# the cube re-seats in both grips instead of staying rotated inside the holding hand (in-grip error ~4 deg otherwise)
REGRIP_RESTORE_LOOSEN = 0.0
# the regripped hand squeezes (rad of extra thumb / middle / ring flexion) while it carries the cube back, so the cube
# follows it instead of staying ~2 mm toward the holding hand (same bump as the loosening)
REGRIP_RESTORE_SQUEEZE = 0.0
# recentering regrip: the HOLDING hand servos the cube back to its ready-grip pose (integral servo on the deviation,
# moving about the nominal cube center) while the other hand is off, the regripping hand closes at its plain nominal
# (no servo), and over the restore the holding hand returns to its nominal (with REGRIP_RESTORE_LOOSEN, sliding on the
# cube the regripped hand now holds): the cube ends at its ready-grip pose in both grips. Off: the regripping hand
# servos to the cube where it is (it stays offset)
REGRIP_RECENTER = False
REGRIP_RECENTER_GAIN = 0.1  # per control step


def nominal_duration(skill: str) -> float:
    if skill in ROLL_DELTA:
        return ROLL_DELAY + ROLL_TIME
    if skill in REGRIP_HAND:
        return sum(REGRIP_PHASES) + REGRIP_RESTORE
    return 0.0  # flicks / hold: open ended (policy driven)


def roll_progress(t):
    """fraction of the quarter roll done at time t [s] (tensor)."""
    return smoothstep((t - ROLL_DELAY) / ROLL_TIME)


def roll_shift(t):
    """outward wrist shift [m] of a single-hand roll at time t [s] (tensor)."""
    end = ROLL_DELAY + ROLL_TIME
    return ROLL_SHIFT * (smoothstep(t / ROLL_SHIFT_RAMP) - smoothstep((t - end + ROLL_SHIFT_RAMP) / ROLL_SHIFT_RAMP))


def regrip_profile(t):
    """(open, backoff, unroll) fractions in [0, 1] at time t [s] (tensors) for a regrip."""
    t_open, t_back, t_unroll, t_appr, t_close = REGRIP_PHASES
    a = t_open
    b = a + t_back
    c = b + t_unroll
    d = c + t_appr
    e = d + t_close
    opening = smoothstep(t / t_open) - smoothstep((t - d) / t_close)
    backoff = smoothstep((t - a) / t_back) - smoothstep((t - c) / t_appr)
    unroll = smoothstep((t - b) / t_unroll)
    del e
    return opening, backoff, unroll


def regrip_loosen(t):
    """holding-hand loosening fraction in [0, 1] at time t [s]: opens over the first, re-closes over the last third of
    the restore."""
    t0, r = sum(REGRIP_PHASES), REGRIP_RESTORE / 3.0
    return smoothstep((t - t0) / r) - smoothstep((t - t0 - 2.0 * r) / r)


def regrip_servo_weight(t):
    """(weight, live) of the sighted regrip servo at time t [s] (tensors): the cube-pose correction blends in while the
    opened hand backs off and is frozen from the approach on. Following the cube while the hand touches it makes the
    hand chase the cube it pushes (runaway drag, ~90 deg layer turns). After the close it blends out again: both hands
    hold the cube by then and the servoed hand's return to the ready pose carries the cube back (no lasting tilt)."""
    t_open, t_back, t_unroll, _, _ = REGRIP_PHASES
    weight = smoothstep((t - t_open) / t_back) - smoothstep((t - sum(REGRIP_PHASES)) / REGRIP_RESTORE)
    live = t < t_open + t_back + t_unroll
    return weight, live
