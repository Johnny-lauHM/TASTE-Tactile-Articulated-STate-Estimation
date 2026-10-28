"""Skill sequences of real solves (random scramble -> best_solution, the executor's planner) for chain training:
each bank entry concatenates `per_seq` solves (both hands end unrolled after every solve, so they chain).

    python scripts/solve/solve_bank.py --n 6144 --per_seq 3 --out cache/chain_bank_solves.json
"""
import argparse
import json
import multiprocessing as mp

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=6144, help="solves")
p.add_argument("--per_seq", type=int, default=3)
p.add_argument("--scramble_len", type=int, default=25)
p.add_argument("--seed", type=int, default=1000, help="scrambles disjoint from the executor's eval seeds (0-99)")
p.add_argument("--workers", type=int, default=0)
p.add_argument("--out", default="cache/chain_bank_solves.json")
a = p.parse_args()


def solve(i):
    from rubik_lab.solve.cube_logic import Cube, best_solution, random_scramble
    rng = np.random.default_rng(a.seed * 100003 + i)
    return best_solution(Cube().apply(random_scramble(a.scramble_len, rng)))[1]


if __name__ == "__main__":
    with mp.Pool(a.workers or mp.cpu_count()) as pool:
        sols = pool.map(solve, range(a.n), chunksize=16)
    seqs = [sum(sols[k : k + a.per_seq], []) for k in range(0, len(sols) - a.per_seq + 1, a.per_seq)]
    json.dump(dict(seqs=seqs, args=vars(a)), open(a.out, "w"))
    print(f"[bank] {len(seqs)} sequences of {a.per_seq} solves, {np.mean([len(s) for s in seqs]):.0f} skills each -> {a.out}")
