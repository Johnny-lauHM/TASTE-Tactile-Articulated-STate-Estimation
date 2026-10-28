"""All paper tables from outputs/solve/*/report.json (run from the project dir). Missing runs print as running/missing."""
import json, math, collections, os

def load(n):
    try:
        return json.load(open(f"outputs/solve/{n}/report.json"))["results"]
    except FileNotFoundError:
        return None

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)

def row(label, n):
    R = load(n)
    if R is None:
        st = "running" if os.path.exists(f"outputs/solve/{n}.log") else "missing"
        print(f"  {label:34s} {n:22s} {st}"); return
    k = sum(r["solved"] for r in R); N = len(R); lo, hi = wilson(k, N)
    st = collections.Counter(r["status"] for r in R)
    mis = sum(bool(r.get("misaligned")) for r in R)
    jam = sum(any(x.get("jam") for x in r["records"]) for r in R)
    fin = sum(sum(bool(x.get("finish")) for x in r["records"]) for r in R)
    extra = (f" misaligned {mis} jam-runs {jam}" if mis or jam else "") + (f" finishes {fin}" if fin else "")
    print(f"  {label:34s} {k:3d}/{N} = {100*k/N:5.1f}% [{lo:4.1f},{hi:4.1f}]  drop {st.get('dropped',0):3d} timeout {st.get('timeout',0):3d}{extra}")

T = {
 "A1 frozen system (fr3dr1 + est_3300), 256": [("sighted nominal", "s2r_dr1_sighted_nom"), ("sighted DR", "s2r_dr1_sighted"),
    ("blind nominal", "s2r_dr1_blind_nom"), ("blind DR", "s2r_dr1_blind"), ("blind nominal rerun (128)", "p_dg_e33_t1")],
 "A1' first candidate (fr3dr2 + p_mix), 256": [("sighted nominal", "p_sig_nom"), ("sighted DR", "p_sig_dr"),
    ("blind nominal", "p_blind_nom"), ("blind DR", "p_blind_dr")],
 "A1 held-out (seed 21, frozen system, 256)": [("sighted nominal", "p_h1_sig_nom"), ("sighted DR", "p_h1_sig_dr"),
    ("blind nominal", "p_h1_bl_nom"), ("blind DR", "p_h1_bl_dr")],
 "Cube realism, jam 15 deg v1 (frozen system, blind, 128)": [("nominal", "p_cj_bl_nom"), ("DR", "p_cj_bl_dr")],
 "Finish before re-grip, nominal blind, jam v1 15 deg, --retry_reset (128)": [("no finish", "p_fin2_base"),
    ("finish, oracle trigger", "p_fin2_or"), ("finish, estimate trigger", "p_fin2_es")],
 "Finish before re-grip, DR blind, jam v2 15 deg, --retry_reset (128)": [("no jam (control)", "p_fin3_nojam"), ("jam, no finish", "p_fin3_base"),
    ("jam, finish oracle", "p_fin3_or"), ("jam, finish estimate", "p_fin3_es")],
 "Finish before re-grip, DR blind, jam v2 15 deg, residual trigger (256)": [("no jam (control)", "p_fin4_nojam"),
    ("jam, no finish", "p_fin4_base"), ("jam, finish oracle (residual)", "p_fin4_or")],
 "Jam v2 15 deg, sighted DR (frozen system, 128)": [("no jam", "p_nj_sig_dr"), ("jam", "p_cj_sig_dr")],
 "3-body cube, nominal (frozen system, 128)": [("blind, jam 15", "p_cb_bl_nom2"), ("blind, no jam", "p_cb_bl_nojam"),
    ("sighted, jam 15", "p_cb_sig_nom"), ("sighted, no jam", "p_cb_sig_nojam")],
 "A2 downstream (fr3dr2, blind DR + slip, 128)": [("proprio + commands", "p_a2s_prop"), ("+ F/T", "p_a2s_ft"), ("p_mix", "p_a2s_mix")],
 "A4 recovery (sighted DR, fr3dr2, 256)": [("full", "p_sig_dr"), ("time-based plan", "p_a4_time"), ("no redo", "p_a4_noredo"),
    ("no timeout-continue", "p_a4_nocont"), ("neither", "p_a4_none")],
 "A5 tactile lag (blind DR, p_mix, 128)": [("0 ms", "p_a5_tac0"), ("50 ms", "p_a5_tac2"), ("100 ms", "p_a5_tac4"),
    ("hand delay 6", "p_a5_hd6"), ("hand delay 12", "p_a5_hd12")],
 "A6 one DR factor on (sighted, fr3dr2, 256; nominal = sighted nominal)": [("obs noise only", "p_ab_obs"), ("pad friction", "p_ab_pad"),
    ("finger gain", "p_ab_finger"), ("arm gain", "p_ab_arm"), ("cube mass", "p_ab_mass"), ("twist friction", "p_ab_twist"),
    ("detent", "p_ab_detent"), ("action delay", "p_ab_delay")],
}
for t, rows in T.items():
    print(t)
    for lab, n in rows:
        row(lab, n)
