"""Logical 3x3x3 cube in the HAND frame and compilation of a solve into physical skills (pure python + numpy).

Frame (same as rubik_lab.moves): R=-y, L=+y, U=+z, D=-z, F=-x, B=+x. Face positions are positions relative to the
hands, not sticker colors: after a whole-cube rotation x the face that was at F sits at U.

Each sticker is (cubie position in {-1,0,1}^3, outward normal). A face turn rotates the stickers whose position lies in
that face's slab; a cube rotation rotates all of them. Turn convention: X = clockwise looking at face X = -90 deg
about the outward normal; X' = +90 deg; X2 = 180 deg. x turns the whole cube like R (y like U, z like F).

Physical skill alphabet (see rubik_lab.solve.skills):
    U U' D D'  : finger flicks (layer about z)
    R R' L L'  : wrist roll of the own-side hand while the other hand holds the rest (layer about y)
    x x'       : both wrists roll together (whole cube about y)
    regrip_left / regrip_right : a rolled hand returns to its ready pose while the other hand holds the cube
F/B are reached by x rotations. Rolls are bounded: a hand roll state is in {-1, 0, +1} quarter turns, a roll that
would leave that range first regrips the hand. Flicks need the flicking hand at roll 0.
"""

from __future__ import annotations

import itertools

import numpy as np

from . import skills as _S

FACES = "URFDLB"  # kociemba facelet order
NORMAL = {"R": (0, -1, 0), "L": (0, 1, 0), "U": (0, 0, 1), "D": (0, 0, -1), "F": (-1, 0, 0), "B": (1, 0, 0)}
ROT_AXIS = {"x": "R", "y": "U", "z": "F"}  # whole-cube rotations turn like these faces
SUFFIX_QUARTERS = {"": -1, "'": 1, "2": 2}  # quarter turns about the outward normal (counter-clockwise positive)


def _rot(axis, quarters):
    """integer rotation matrix: `quarters` * 90 deg about the unit axis (right-hand rule)."""
    a = np.array(axis)
    k = quarters % 4
    c, s = [1, 0, -1, 0][k], [0, 1, 0, -1][k]
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return (np.eye(3, dtype=int) * c + s * K + (1 - c) * np.outer(a, a)).astype(int)


def _facelet_frame(face):
    """(up, right) in-face directions of the kociemba facelet grid for `face` (row 0 = top, col 0 = left)."""
    return {
        "U": ((1, 0, 0), (0, -1, 0)),  # looking down at U with F at the bottom: up = B(+x), right = R(-y)
        "D": ((-1, 0, 0), (0, -1, 0)),  # looking up at D with F at the top: up = F, right = R
        "F": ((0, 0, 1), (0, -1, 0)),
        "B": ((0, 0, 1), (0, 1, 0)),
        "R": ((0, 0, 1), (1, 0, 0)),
        "L": ((0, 0, 1), (-1, 0, 0)),
    }[face]


# canonical sticker slots: index i in kociemba facelet order -> (pos, normal)
SLOTS = []
for _f in FACES:
    _n = np.array(NORMAL[_f])
    _up, _right = (np.array(v) for v in _facelet_frame(_f))
    for _r, _c in itertools.product(range(3), range(3)):
        SLOTS.append((tuple(_n + _up * (1 - _r) + _right * (_c - 1)), tuple(_n)))
SLOT_INDEX = {s: i for i, s in enumerate(SLOTS)}


def _perm(R, mask=None):
    """sticker permutation: new[j] = old[perm[j]] for a rotation R applied to the stickers selected by mask."""
    perm = list(range(54))
    for i, (p, n) in enumerate(SLOTS):
        if mask is not None and not mask(np.array(p)):
            continue
        j = SLOT_INDEX[(tuple(R @ np.array(p)), tuple(R @ np.array(n)))]
        perm[j] = i
    return perm


MOVE_PERM = {}
for _f in FACES:
    _n = np.array(NORMAL[_f])
    for _sfx, _q in SUFFIX_QUARTERS.items():
        MOVE_PERM[_f + _sfx] = _perm(_rot(_n, _q), lambda p, n=_n: int(p @ n) == 1)
for _r, _f in ROT_AXIS.items():
    for _sfx, _q in SUFFIX_QUARTERS.items():
        MOVE_PERM[_r + _sfx] = _perm(_rot(NORMAL[_f], _q))
# middle slices (3-body cube): s<X> turns the slab between X and its opposite face like X (sU == sD')
for _f in FACES:
    _n = np.array(NORMAL[_f])
    for _sfx, _q in SUFFIX_QUARTERS.items():
        MOVE_PERM["s" + _f + _sfx] = _perm(_rot(_n, _q), lambda p, n=_n: int(p @ n) == 0)


def bound_layer_moves(face: str, twist: float, mid: float = 0.0) -> list[str]:
    """hand-frame moves for whole quarter turns of the bound layer: twist = free layer relative to the middle slab (the core
    for the 2-body cube), mid = middle slab relative to the core (3-body cube; the core is the face opposite `face`)."""
    qt, qm = (int(round(a / (0.5 * np.pi))) for a in (twist, mid))
    name = {1: "'", 2: "2", 3: ""}
    out = [face + name[(qt + qm) % 4]] if (qt + qm) % 4 else []
    return out + (["s" + face + name[qm % 4]] if qm % 4 else [])

SOLVED = "".join(f * 9 for f in FACES)


class Cube:
    """facelet string in kociemba order; letters are the colors named after the face they belong to when solved."""

    def __init__(self, facelets: str = SOLVED):
        assert len(facelets) == 54
        self.s = facelets

    def apply(self, moves):
        if isinstance(moves, str):
            moves = moves.split()
        s = self.s
        for m in moves:
            perm = MOVE_PERM[m.replace("’", "'")]
            s = "".join(s[perm[j]] for j in range(54))
        self.s = s
        return self

    def copy(self):
        return Cube(self.s)

    def solved(self) -> bool:
        return all(len(set(self.s[9 * i : 9 * i + 9])) == 1 for i in range(6))

    def kociemba_string(self) -> str:
        """relabel colors by the current center at each face position (kociemba needs centers on their faces)."""
        center = {self.s[9 * i + 4]: FACES[i] for i in range(6)}
        return "".join(center[c] for c in self.s)

    def __eq__(self, other):
        return isinstance(other, Cube) and self.s == other.s

    def __repr__(self):
        return f"Cube({self.s})"


def solve(cube: Cube) -> list[str]:
    """face-turn solution (positions in the hand frame) for the current state."""
    import kociemba

    if cube.solved():
        return []
    return kociemba.solve(cube.kociemba_string()).split()


def random_scramble(n: int, rng: np.random.Generator) -> list[str]:
    out, last = [], None
    while len(out) < n:
        f = FACES[rng.integers(6)]
        if f == last:
            continue
        out.append(f + ["", "'", "2"][rng.integers(3)])
        last = f
    return out


# --------------------------------------------------------------------------------------------- skill compilation
X_CYCLE = {"x": {"F": "U", "U": "B", "B": "D", "D": "F", "R": "R", "L": "L"}}  # x: the face at F goes to U, ...
X_CYCLE["x'"] = {v: k for k, v in X_CYCLE["x"].items()}
DIRECT = set("UDRL")
ROLL_HAND = {"R": "right", "L": "left"}


def face_after(rotation: str, face: str) -> str:
    """position a face moves to under a whole-cube rotation x / x'."""
    return X_CYCLE[rotation][face]


class SkillCompiler:
    """Emits physical skills, tracking each hand's roll state in quarter turns about its own face's outward normal.

    R (clockwise seen from R) rolls the right hand by -1, R' by +1, L / L' the left hand by -1 / +1. x turns like R:
    right -1, left +1 (counter-clockwise seen from L). States stay in [-1, 1]; flicks need roll 0.
    """

    ROLL_DELTA = {"R": {"right": -1}, "R'": {"right": 1}, "L": {"left": -1}, "L'": {"left": 1},
                  "x": {"right": -1, "left": 1}, "x'": {"right": 1, "left": -1}}  # fmt: skip
    FLICK_HAND = {"U": "right", "U'": "left", "D": "left", "D'": "right"}  # rubik_lab.flick.FLICKS
    # flicks need both hands at roll 0: a rolled holding hand keeps fingers on the flicked layer's row and blocks it
    # (in-chain U after L / L': 0/16)
    FLICK_BOTH_UNROLLED = True
    # the last regrip before a flick is the OTHER hand's: after a regrip the cube sits ~2 mm toward the other hand and the
    # regripped hand's own flick fails half the time (regrip_right -> U 42-51%, regrip_left -> U' 48%; the other hand's
    # flick 86-96%). With the handoff-ready policies (2026-10-20, hr2) it hurts: R / R' / U' -> regrip_right -> U 67 / 93 /
    # 89% directly, 53 / 58 / 68% with the extra regrip_left in between
    FLICK_REGRIP_OTHER_LAST = False
    # a single-hand roll (R / R' / L / L') needs the other hand at roll 0: rolling while the other hand holds in a rolled
    # grip fails a third to half of the time (L -> R 47%, R' -> L 60%, x -> R 73%); planning cost ~0 (DP reorders)
    SOLO_ROLL_OTHER_UNROLLED = True

    def __init__(self):
        self.roll = {"left": 0, "right": 0}
        self.skills: list[str] = []

    # when both hands need a regrip, the right one goes first: with both rolled (after x / x') the hand that stays holds
    # the cube alone in its rolled grip, and the rolled LEFT hand does that far better (chain5 m400: x -> regrip_right
    # 98.3%, x -> regrip_left 92.9%, the cube falls ~30 mm in the first 0.15 s of the opening)
    REGRIP_RIGHT_FIRST = True

    def _regrip(self, hand):
        if self.roll[hand] != 0:
            self.skills.append(f"regrip_{hand}")
            self.roll[hand] = 0

    # the left hand never regrips while the right one is rolled (it would hold alone in its rolled grip): the right
    # regrips first, also when it would not need to (one extra regrip ~99.6% vs 92.9%)
    LEFT_REGRIP_RIGHT_UNROLLED = True

    def _regrip_all(self, hands):
        if self.LEFT_REGRIP_RIGHT_UNROLLED and "left" in hands and self.roll["left"] != 0 and self.roll["right"] != 0:
            hands = list(hands) + ["right"]
        if self.REGRIP_RIGHT_FIRST:
            hands = sorted(set(hands), key=lambda h: h != "right")
        for h in hands:
            self._regrip(h)

    def _emit(self, skill):
        if skill in self.FLICK_HAND:
            fh = self.FLICK_HAND[skill]
            other = "left" if fh == "right" else "right"
            # (with skills.FLICK_PARTNER_LIFT the rolled other hand lifts its digits off the U layer: only fh unrolled)
            both = self.FLICK_BOTH_UNROLLED and _S.FLICK_PARTNER_LIFT <= 0
            need = [fh]
            if both or (self.roll[other] != 0 and (other, self.roll[other]) not in _S.FLICK_PARTNER_OK):
                need.append(other)
            self._regrip_all(need)
            if self.FLICK_REGRIP_OTHER_LAST and self.skills and self.skills[-1] == f"regrip_{fh}":
                self.skills.append(f"regrip_{other}")  # (at roll 0: open, back off, re-close)
                self.roll[other] = 0
        delta = self.ROLL_DELTA.get(skill, {})
        need = []
        if self.SOLO_ROLL_OTHER_UNROLLED and len(delta) == 1:
            need.append("left" if "right" in delta else "right")
        need += [hand for hand, d in delta.items() if abs(self.roll[hand] + d) > 1]
        self._regrip_all(need)
        for hand, d in delta.items():
            self.roll[hand] += d
        self.skills.append(skill)

    def finish(self):
        """return both hands to their ready pose (so the next solve or primitive starts canonical)."""
        self._regrip_all(["left", "right"])
        return self.skills


def compile_solution(solution: list[str]) -> tuple[list[str], list[str]]:
    """face turns (hand frame, kociemba output) -> (physical skills, hand-frame moves they perform).

    Faces at F/B are brought to U by x / x' (and stay rotated: later faces are renamed accordingly).
    D2/U2/R2/L2 are two quarter turns in the same direction.
    """
    comp = SkillCompiler()
    performed: list[str] = []
    # mapping original position -> current position, updated by every whole-cube rotation
    where = {f: f for f in "URFDLB"}
    for move in solution:
        face, sfx = move[0], move[1:]
        cur = where[face]
        if cur in "FB":
            rot = "x" if cur == "F" else "x'"
            comp._emit(rot)
            performed.append(rot)
            where = {k: face_after(rot, v) for k, v in where.items()}
            cur = where[face]
        assert cur in DIRECT, (move, cur)
        quarters = [sfx] if sfx in ("", "'") else [""] * 2
        for q in quarters:
            comp._emit(cur + q)
            performed.append(cur + q)
    comp.finish()
    return comp.skills, performed


# ------------------------------------------------------------------------------- regrip-aware compilation (DP)
SKILL_COST = {"U": 0.7, "U'": 0.7, "D": 0.7, "D'": 0.7, "R": 1.25, "R'": 1.25, "L": 1.25, "L'": 1.25,
              "x": 1.25, "x'": 1.25, "regrip_left": 1.8, "regrip_right": 1.8}  # fmt: skip  nominal seconds
DEFAULT_FACE_SKILLS = ("U", "U'", "R", "R'", "L", "L'")


def _emit_all(skills, roll, face_skills):
    """simulate SkillCompiler on a candidate segment: (emitted skills incl. regrips, new roll) or None if invalid."""
    comp = SkillCompiler()
    comp.roll = dict(roll)
    for s in skills:
        if s not in face_skills and s not in ("x", "x'"):
            return None
        comp._emit(s)
    return comp.skills, comp.roll


def seq_cost(skills, cost=None, trans_cost=None, prev="start"):
    """cost of a skill sequence: per-skill costs, or trans_cost(previous skill, skill) summed along it."""
    if trans_cost is None:
        cost = cost or SKILL_COST
        return sum(cost[s] for s in skills)
    c = 0.0
    for s in skills:
        c += trans_cost(prev, s)
        prev = s
    return c


def compile_solution_dp(solution: list[str], face_skills=DEFAULT_FACE_SKILLS, cost=None, finish=True, init_roll=(0, 0), trans_cost=None):
    """Like compile_solution, but chooses for every face turn the cheapest execution (whole-cube rotations x / x' /
    x2 first or not, direction of half turns) by dynamic programming over (x orientation, left roll, right roll).
    trans_cost(prev, skill): cost depending on the preceding skill too (e.g. -log success of that transition); the
    previous skill then joins the DP state. Returns (skills, hand-frame moves performed)."""
    cost = cost or SKILL_COST
    face_skills = set(face_skills)
    rots = {0: [], 1: ["x"], 3: ["x'"], 2: ["x", "x"], -2: ["x'", "x'"]}  # extra quarters of x before the turn

    def where_after(o, face):
        f = face
        for _ in range(o % 4):
            f = X_CYCLE["x"][f]
        return f

    # state: (o, roll_left, roll_right, last skill) -> (cost, skills, performed); the last skill only with trans_cost
    states = {(0, init_roll[0], init_roll[1], "start"): (0.0, [], [])}
    for move in solution:
        face, sfx = move[0], move[1:]
        new = {}
        for (o, rl, rr, last), (c0, sk0, pf0) in states.items():
            for dq, rseq in rots.items():
                o2 = (o + dq) % 4
                pos = where_after(o2, face)
                if sfx == "2":
                    variants = [[pos] * 2, [pos + "'"] * 2]
                else:
                    variants = [[pos + sfx]]
                for turns in variants:
                    r = _emit_all(rseq + turns, {"left": rl, "right": rr}, face_skills)
                    if r is None:
                        continue
                    emitted, roll = r
                    c = c0 + seq_cost(emitted, cost, trans_cost, last)
                    key = (o2, roll["left"], roll["right"], emitted[-1] if trans_cost is not None else "")
                    if key not in new or c < new[key][0]:
                        new[key] = (c, sk0 + emitted, pf0 + rseq + turns)
        states = new
    best = None
    for (o, rl, rr, last), (c, sk, pf) in states.items():
        sk = list(sk)
        if finish:
            comp = SkillCompiler()
            comp.roll = {"left": rl, "right": rr}
            tail = comp.finish()
            c += seq_cost(tail, cost, trans_cost, sk[-1] if sk else "start")
            sk += tail
        if best is None or c < best[0]:
            best = (c, sk, pf)
    return best[1], best[2]


def _orientations():
    """24 whole-cube rotation sequences (x/y/z words) reaching every orientation once."""
    seen, out = set(), []
    for a in ([], ["x"], ["x2"], ["x'"], ["z"], ["z'"]):
        for b in ([], ["y"], ["y2"], ["y'"]):
            seq = a + b
            key = Cube().apply(seq).s
            if key not in seen:
                seen.add(key)
                out.append(seq)
    return out


ORIENTATIONS = _orientations()


def invert(moves: list[str]) -> list[str]:
    return [m[0] + {"": "'", "'": "", "2": "2"}[m[1:]] for m in reversed(moves)]


def best_solution(cube: Cube, face_skills=DEFAULT_FACE_SKILLS, cost=None, init_roll=(0, 0), trans_cost=None):
    """Solve the cube conjugated by each of the 24 whole-cube orientations, map every solution back to the hand
    frame (also the inverse of the inverse state's solution) and keep the one whose regrip-aware execution is cheapest. Returns (solution, skills, performed, cost)."""
    cost = cost or SKILL_COST
    best = None
    for g in ORIENTATIONS:
        rot = Cube().apply(g)  # center letters show which original face sits at each position after g
        to_orig = {FACES[i]: rot.s[9 * i + 4] for i in range(6)}
        sol_g = solve(cube.copy().apply(g))
        # a second candidate: the inverse of the solution of the inverse state (another solution of similar length)
        for sg in (sol_g, invert(solve(Cube().apply(sol_g)))):
            sol = [to_orig[m[0]] + m[1:] for m in sg]
            skills, performed = compile_solution_dp(sol, face_skills, cost, init_roll=init_roll, trans_cost=trans_cost)
            c = seq_cost(skills, cost, trans_cost)
            if best is None or c < best[3]:
                best = (sol, skills, performed, c)
    return best


class ReliabilityCost:
    """trans_cost(prev, skill) = -log P(skill succeeds | previous skill), from measured transition counts
    {"prev -> skill": [ok, n]} (handoff_matrix json "trans", or executor records): each transition's rate is shrunk
    toward its skill's overall rate with `prior` pseudo-counts (rare transitions stay near the skill's mean)."""

    def __init__(self, trans: dict, prior: float = 30.0):
        import math

        sk = {}
        for key, (ok, n) in trans.items():
            cur = key.split(" -> ")[1]
            a = sk.setdefault(cur, [0, 0])
            a[0] += ok
            a[1] += n
        self.p_skill = {k: (v[0] + 1.0) / (v[1] + 2.0) for k, v in sk.items()}
        self.p = {}
        for key, (ok, n) in trans.items():
            prev, cur = key.split(" -> ")
            self.p[(prev, cur)] = (ok + prior * self.p_skill[cur]) / (n + prior)
        self._log = math.log

    def __call__(self, prev, skill):
        p = self.p.get((prev, skill), self.p_skill.get(skill, 0.99))
        return -self._log(max(p, 1e-3))

    @classmethod
    def from_files(cls, paths, prior: float = 30.0):
        """merge transition counts of handoff_matrix jsons and executor reports."""
        import json

        trans = {}
        for path in paths:
            d = json.load(open(path))
            if "trans" in d:
                items = d["trans"].items()
            else:  # executor report: rebuild transitions from every env's skill records
                items = []
                for r in d["results"]:
                    prev = "start"
                    for q in r["records"]:
                        items.append((f"{prev} -> {q['skill']}", [int(bool(q.get("truth_success"))), 1]))
                        prev = q["skill"]
            for k, (ok, n) in items:
                a = trans.setdefault(k, [0, 0])
                a[0] += ok
                a[1] += n
        return cls(trans, prior)

