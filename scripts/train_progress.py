"""Summarize an rsl_rl training log (stdout) as a table of key metrics.

    python scripts/train_progress.py logs/train_v2.log [--every 50]
"""

import argparse
import re

KEYS = ["Mean reward", "Mean episode length", "layer_done_latch_rate", "primitive_done_rate", "move_done_rate",
        "drop_rate", "gravity", "tips_on_layer", "twist_err_deg", "home_err"]  # fmt: skip
p = argparse.ArgumentParser()
p.add_argument("log")
p.add_argument("--every", type=int, default=50)
a = p.parse_args()
rows, cur, it = [], {}, None
for line in open(a.log, errors="ignore"):
    m = re.search(r"Learning iteration (\d+)/(\d+)", line)
    if m:
        if it is not None:
            rows.append((it, cur))
        it, cur = int(m.group(1)), {}
        continue
    for k in KEYS:
        m = re.search(rf"{re.escape(k)}:\s*(-?[\d.]+)", line)
        if m and k not in cur:
            cur[k] = float(m.group(1))
    m = re.search(r"ETA:\s*(\S+)", line)
    if m:
        cur["ETA"] = m.group(1)
if it is not None:
    rows.append((it, cur))
short = ["reward", "ep_len", "latch", "prim_done", "move_done", "drop", "g", "tips_layer", "twist_err", "home_err"]
print("iter  " + " ".join(f"{s:>10s}" for s in short) + "   ETA")
sel = [r for r in rows if r[0] % a.every == 0] + ([rows[-1]] if rows and rows[-1][0] % a.every else [])
for i, c in sel:
    print(f"{i:5d} " + " ".join(f"{c.get(k, float('nan')):10.4f}" for k in KEYS) + f"   {c.get('ETA', '')}")
