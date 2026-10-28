"""Rubik's cube move primitives and formula parsing (pure Python, no Isaac imports).

Cube faces in the env frame of the bimanual home grip (thumbs on the front face):
    R = -y (right hand)   L = +y (left hand)   U = +z   D = -z   F = -x (thumbs)   B = +x (fingers)

Physics uses one 2-body cube (core + "top" layer twisting about its local +z). Because the outer shape is
the same under all 24 cube rotations, any single-layer move is simulated by spawning the cube so that its
local +z points along that face's outward normal for independent episodes. This does not
provide a physical multi-face executor: continuous two-body execution supports only one face.
Formula compilation below is symbolic and does not certify trained skill availability.

Angle convention: positive = counter-clockwise about the face's outward normal (right-hand rule), so the
standard clockwise move X is -90 deg, X' is +90 deg, and X2 is 180 deg.
"""

from __future__ import annotations

import math
import re

FACE_NORMALS = {
    "R": (0.0, -1.0, 0.0),
    "L": (0.0, 1.0, 0.0),
    "U": (0.0, 0.0, 1.0),
    "D": (0.0, 0.0, -1.0),
    "F": (-1.0, 0.0, 0.0),
    "B": (1.0, 0.0, 0.0),
}
# rotation (w, x, y, z) taking the cube's local +z (twist axis of the "top" layer) onto the face normal
_S = math.sqrt(0.5)
FACE_QUATS = {
    "U": (1.0, 0.0, 0.0, 0.0),
    "D": (0.0, 1.0, 0.0, 0.0),
    "R": (_S, _S, 0.0, 0.0),
    "L": (_S, -_S, 0.0, 0.0),
    "F": (_S, 0.0, -_S, 0.0),
    "B": (_S, 0.0, _S, 0.0),
}
# hand whose own side is the moving layer (twists it with a wrist roll); None = finger-trick move
FACE_TWIST_HAND = {"R": "right", "L": "left", "U": None, "D": None, "F": None, "B": None}

_SUFFIX_ANGLE = {"": -math.pi / 2, "'": math.pi / 2, "2": math.pi}
# face turns, then whole-cube rotations x (turns the whole cube like R). Appended so the face-move indices stay fixed.
ALL_MOVES = [f + s for f in "RLUDFB" for s in ("", "'", "2")] + ["x", "x'", "x2"]  # fixed order -> one-hot index
MOVE_INDEX = {m: i for i, m in enumerate(ALL_MOVES)}
from .constants import CUBE_SIZE

ROW = CUBE_SIZE / 3  # cubie pitch [m]

# whole-cube rotation x: about the R axis, same direction as R
FACE_NORMALS["x"] = FACE_NORMALS["R"]
FACE_QUATS["x"] = FACE_QUATS["R"]  # internal twist axis for x episodes (the layer must simply stay put)
FACE_TWIST_HAND["x"] = None


def primitive_spec(move: str) -> dict | None:
    """How a primitive is executed with the edge grasp (None = not a primitive; compose it, see compile_formula).

    hands  : (left, right) which wrists rotate
    axis   : logical rotation axis (unit), angle: signed angle about it [rad]
    layer  : target angle of the internal twist joint (0 for whole-cube rotations)
    shift  : (left, right) logical offset of the twisting grip before the twist (U: grip the top row first)
    """
    f, sfx = move[0], move[1:]
    if sfx == "2":
        return None
    ang = _SUFFIX_ANGLE[sfx]
    z = (0.0, 0.0, 0.0)
    if f == "R":
        return dict(hands=(0, 1), axis=FACE_NORMALS["R"], angle=ang, layer=ang, shift=(z, z))
    if f == "L":
        return dict(hands=(1, 0), axis=FACE_NORMALS["L"], angle=ang, layer=ang, shift=(z, z))
    if f == "U":  # U: left hand orbits towards the back, U': right hand orbits towards the back (fingers side)
        hands = (1, 0) if sfx == "" else (0, 1)
        up = (0.0, 0.0, ROW)
        return dict(hands=hands, axis=FACE_NORMALS["U"], angle=ang, layer=ang, shift=(up, z) if hands[0] else (z, up))
    if f == "x":
        return dict(hands=(1, 1), axis=FACE_NORMALS["R"], angle=ang, layer=0.0, shift=(z, z))
    return None


PRIMITIVES = [m for m in ALL_MOVES if primitive_spec(m) is not None]  # R R' L L' U U' x x'
# face moves the edge grasp cannot turn directly, as whole-cube rotations + U (verified with facelet permutations)
CONJUGATES = {"F": "x U x'", "F'": "x U' x'", "B": "x' U x", "B'": "x' U' x", "D": "x2 U x2", "D'": "x2 U' x2"}


def compile_formula(formula: str) -> list[str]:
    """Expand symbolic primitives; this does not validate physical execution support."""
    out: list[str] = []
    for m in parse_formula(formula):
        if m in PRIMITIVES:
            out.append(m)
        elif m.endswith("2"):
            out += compile_formula(m[0]) * 2
        else:
            out += compile_formula(CONJUGATES[m])
    return out


def move_face(move: str) -> str:
    return move[0]


def move_angle(move: str) -> float:
    return _SUFFIX_ANGLE[move[1:]]


def parse_formula(formula: str) -> list[str]:
    """Parse moves and repeated groups, rejecting every unrecognized character.

    Supports nested parentheses and Unicode prime marks. Invalid text must never
    silently become a shorter robot instruction sequence.
    """
    s = formula.replace("’", "'").replace("′", "'")
    pos = 0

    def group(nested=False):
        nonlocal pos
        out = []
        while pos < len(s):
            if s[pos].isspace():
                pos += 1
                continue
            if s[pos] == ")":
                if not nested:
                    raise ValueError(f"Unexpected ')' at offset {pos}")
                pos += 1
                return out
            if s[pos] == "(":
                pos += 1
                part = group(True)
                if not part:
                    raise ValueError("Empty formula group")
                start = pos
                while pos < len(s) and s[pos].isascii() and s[pos].isdigit():
                    pos += 1
                count = int(s[start:pos]) if pos > start else 1
                if count < 1:
                    raise ValueError("Group repetition must be positive")
                out.extend(part * count)
                continue
            match = re.match(r"[RLUDFBx](?:2'|2|')?", s[pos:])
            if match is None:
                raise ValueError(f"Unsupported formula text at offset {pos}: {s[pos:]!r}")
            out.append(match.group().replace("2'", "2"))
            pos += len(match.group())
        if nested:
            raise ValueError("Unclosed formula group")
        return out

    return group()


def invert(moves: list[str]) -> list[str]:
    inv = {"": "'", "'": "", "2": "2"}
    return [m[0] + inv[m[1:]] for m in reversed(moves)]
