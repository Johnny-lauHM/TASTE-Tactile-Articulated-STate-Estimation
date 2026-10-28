import numpy as np
import pytest

from rubik_lab.solve.cube_logic import Cube, SOLVED, compile_solution, random_scramble, solve


def test_move_orders_and_identities():
    for m in ["U", "R", "F", "D", "L", "B", "x", "y", "z"]:
        c = Cube().apply([m] * 4)
        assert c.s == SOLVED
        assert Cube().apply([m, m + "'"]).s == SOLVED
        assert Cube().apply([m, m]).s == Cube().apply([m + "2"]).s
    assert Cube().apply("R U R' U'".split() * 6).s == SOLVED  # sexy move has order 6


def test_conventions():
    # x turns the whole cube like R: R L' x' leaves the M slice displaced, x == R M' L' style identity via facelets
    c = Cube().apply(["x"])
    assert c.s[9 * 0 + 4] == "F"  # the F center is now on top
    # F == x' U x? With x, the face at F goes to U, so performing U there and undoing is F: x U x'
    assert Cube().apply("x U x'".split()).s == Cube().apply(["F"]).s
    assert Cube().apply("x' U x".split()).s == Cube().apply(["B"]).s
    assert Cube().apply("x2 U x2".split()).s == Cube().apply(["D"]).s
    # clockwise U moves the F face's top row to L
    c = Cube().apply(["U"])
    assert c.s[36:39] == "FFF"  # L face top row (kociemba L block starts at 36)


@pytest.mark.parametrize("seed", range(20))
def test_solve_and_compile(seed):
    rng = np.random.default_rng(seed)
    scramble = random_scramble(25, rng)
    cube = Cube().apply(scramble)
    sol = solve(cube)
    assert Cube(cube.s).apply(sol).solved()
    skills, performed = compile_solution(sol)
    # replay the hand-frame moves: the cube ends solved up to a whole-cube rotation
    c = Cube(cube.s).apply(performed)
    assert c.solved()
    roll = {"left": 0, "right": 0}
    flick = {"U": "right", "U'": "left", "D": "left", "D'": "right"}
    for s in skills:
        if s.startswith("regrip_"):
            roll[s[7:]] = 0
            continue
        if s in flick:
            assert roll[flick[s]] == 0
        for h, d in {"R": {"right": -1}, "R'": {"right": 1}, "L": {"left": -1}, "L'": {"left": 1},
                     "x": {"right": -1, "left": 1}, "x'": {"right": 1, "left": -1}}.get(s, {}).items():
            roll[h] += d
            assert abs(roll[h]) <= 1
    assert roll == {"left": 0, "right": 0}
    assert [s for s in skills if not s.startswith("regrip")] == performed


@pytest.mark.parametrize("seed", range(20))
def test_dp_compile(seed):
    from rubik_lab.solve.cube_logic import compile_solution_dp
    rng = np.random.default_rng(100 + seed)
    cube = Cube().apply(random_scramble(25, rng))
    sol = solve(cube)
    skills, performed = compile_solution_dp(sol)
    assert Cube(cube.s).apply(performed).solved()
    assert [s for s in skills if not s.startswith("regrip")] == performed
    assert not any(s in ("D", "D'") for s in skills)


@pytest.mark.parametrize("seed", range(5))
def test_best_solution(seed):
    from rubik_lab.solve.cube_logic import best_solution, ORIENTATIONS
    assert len(ORIENTATIONS) == 24
    rng = np.random.default_rng(200 + seed)
    cube = Cube().apply(random_scramble(25, rng))
    sol, skills, performed, c = best_solution(cube)
    assert Cube(cube.s).apply(sol).solved()
    assert Cube(cube.s).apply(performed).solved()


def test_slices_and_bound_layer_moves():
    from rubik_lab.solve.cube_logic import bound_layer_moves

    for f, o in zip("URF", "DLB"):
        assert Cube().apply(["s" + f] * 4).s == SOLVED
        assert Cube().apply(["s" + f]).s == Cube().apply(["s" + o + "'"]).s  # the same slab, opposite normal
        assert Cube().apply(["s" + f]).s != SOLVED
    # a whole-cube y (turns like U) = U layer + middle slab + D layer, all -90 deg about +z
    assert Cube().apply("U sU D'".split()).s == Cube().apply(["y"]).s
    q = 0.5 * np.pi
    assert bound_layer_moves("U", q, 0.0) == ["U'"]
    assert bound_layer_moves("U", 0.03, -0.02) == []
    # 3-body: twist is relative to the middle slab, so the free layer turned by mid + twist about the core
    assert bound_layer_moves("U", 0.0, q) == ["U'", "sU'"]
    assert bound_layer_moves("U", -q, q) == ["sU'"]
    assert bound_layer_moves("R", q, q) == ["R2", "sR'"]
    # the free layer and the slab both a quarter: the cube is the core face turned back (whole-cube rotation)
    assert Cube().apply(bound_layer_moves("U", 0.0, q)).s == Cube().apply("y' D'".split()).s
