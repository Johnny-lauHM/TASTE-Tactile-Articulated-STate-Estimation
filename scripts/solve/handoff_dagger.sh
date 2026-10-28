#!/usr/bin/env bash
# Handoff training rounds (DAgger style): train the flick and the wrist family in parallel from the cumulative start
# pool, then run random skill chains with the new policies (scripts/solve/handoff_matrix.py), which measures every
# skill / transition inside chains and records the start states they really meet; those join the pool for the next round.
#   [MODS="REGRIP_HOLD_PRESS=0.5 ..."] [EVAL0=1] scripts/solve/handoff_dagger.sh <tag> <rounds> <iters> <flick_ckpt> <wrist_ckpt> [pool0.pt...]
# MODS: rubik_lab.solve.skills constants for every run; EVAL0=1: a first evaluation / pool round with the given policies.
# TRAIN_FLICK=0: keep the flick policy fixed (train the wrist family only); WRIST_ENVS (4096); SETTLE: s between skills in
# the evaluation chains (as the executor pauses); FLICK_ENVS (4096)
# Progress: logs/dagger_<tag>/summary.log
set -uo pipefail
cd "$(dirname "$0")/../.."
tag=$1 rounds=$2 iters=$3 flick=$4 wrist=$5; shift 5
out=logs/dagger_$tag; mkdir -p $out cache/pools
pool=cache/pools/${tag}_pool.pt
source ~/miniconda3/etc/profile.d/conda.sh && conda activate sharpa
merge() {  # merge pool files into $pool (newest states kept, at most 200k)
    python - "$pool" "$@" <<'EOF'
import sys, torch
parts = [torch.load(f) for f in sys.argv[2:]]
pool = {k: torch.cat([p[k] for p in parts])[-200000:] for k in parts[0]}
torch.save(pool, sys.argv[1]); print(f"pool {sys.argv[1]}: {len(pool['roll'])} states")
EOF
}
COMMON=(--set link_contacts=False --set detent_torque=0.08 --set detent_width=0.2618 --mod ROLL_SHIFT=0.016 --mod ROLL_RING_LIFT=0.6)
for m in ${MODS:-}; do COMMON+=(--mod "$m"); done
FLICK_NOM="flick_legacy_ckpt='logs/rsl_rl/rubik_flick/2026-10-17_16-28-11/model_1900.pt'"
CACHE=cache/solve_ready_ud_all4_flick.npz
latest() { ls -t "$1"/model_*.pt | head -1; }
evaluate() {  # evaluate the current policies in random chains, add the start states they meet to the pool
    local r=$1
    MEM_MAX=20G scripts/run_isolated.sh $out/eval_r$r.log python -u scripts/solve/handoff_matrix.py --headless --num_envs 256 \
        --length 40 --stop_alive 0.03 --seed $((100 + r)) --settle ${SETTLE:-0} --set "$FLICK_NOM" --policy flick=$flick --policy roll=$wrist --policy regrip=$wrist \
        "${COMMON[@]}" --tag ${tag}_r$r --dump cache/pools/${tag}_r$r.pt
    grep "^\[handoff\]" $out/eval_r$r.log | grep -v "t=\|dev " >> $out/summary.log
    # keep each family's checkpoint only if its in-chain skill success improved (else go back to the best so far)
    read flick wrist < <(python - outputs/solve/handoff/${tag}_r$r.json $out/best.json $flick $wrist <<'PY'
import json, os, sys
res, best_f, fl, wr = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sk = json.load(open(res))["skill"]
def score(names):
    v = [sk[k] for k in names if k in sk]
    return sum(a / b for a, b in v) / max(len(v), 1)
cur = {"flick": score(["U", "U'"]), "wrist": score(["x", "x'", "regrip_left", "regrip_right", "R", "R'", "L", "L'"])}
best = json.load(open(best_f)) if os.path.exists(best_f) else {}
out = {}
for fam, ck in (("flick", fl), ("wrist", wr)):
    if fam not in best or cur[fam] >= best[fam]["score"]:
        best[fam] = {"score": cur[fam], "ckpt": ck}
    out[fam] = best[fam]["ckpt"]
json.dump(best, open(best_f, "w"), indent=1)
print(f"[dagger] score flick {cur['flick']:.3f} wrist {cur['wrist']:.3f} -> best flick {best['flick']['score']:.3f} wrist {best['wrist']['score']:.3f}", file=sys.stderr)
print(out["flick"], out["wrist"])
PY
) 2>> $out/summary.log
    echo "[dagger] next flick=$flick wrist=$wrist" >> $out/summary.log
    if [ -f cache/pools/${tag}_r$r.pt ]; then
        if [ -f $pool ]; then merge $pool cache/pools/${tag}_r$r.pt; else merge cache/pools/${tag}_r$r.pt; fi | tee -a $out/summary.log
    fi
}
[ $# -gt 0 ] && merge "$@" | tee -a $out/summary.log
[ "${EVAL0:-0}" = 1 ] && { echo "[dagger] round 0 eval $(date +%T)" >> $out/summary.log; evaluate 0; }
for r in $(seq 1 $rounds); do
    echo "[dagger] round $r start $(date +%T) flick=$flick wrist=$wrist" | tee -a $out/summary.log
    pf=
    [ "${TRAIN_FLICK:-1}" = 1 ] && { MEM_MAX=20G scripts/run_isolated.sh $out/train_flick_r$r.log python -u scripts/solve/train.py --headless --skills "U,U'" \
        --num_envs ${FLICK_ENVS:-4096} --iterations $iters --run_name ${tag}_flick_r$r --ready_cache $CACHE --checkpoint $flick \
        --buffer_size 50000 --buffer_prob 0.5 "${COMMON[@]}" --set "$FLICK_NOM" --set "start_pool='$pool'" --set start_pool_prob=${POOL_P:-0.3} &
    pf=$!; }
    MEM_MAX=20G scripts/run_isolated.sh $out/train_wrist_r$r.log python -u scripts/solve/train.py --headless \
        --skills "x,x',regrip_left,regrip_right,R,R',L,L'" --weights 1,1,2,1,2,2,3,3 --num_envs ${WRIST_ENVS:-4096} --iterations $iters \
        --run_name ${tag}_wrist_r$r --ready_cache $CACHE --checkpoint $wrist --buffer_size 50000 --buffer_prob 0.5 \
        --set act_wrist_pos=0.005 --set act_wrist_rot=0.06 --set act_finger=0.4 "${COMMON[@]}" \
        --set "start_pool='$pool'" --set start_pool_prob=${POOL_P:-0.3} &
    pw=$!
    wait $pf $pw
    [ -n "$pf" ] && flick=$(latest "$(ls -dt logs/rsl_rl/rubik_solve/*_${tag}_flick_r$r | head -1)")
    wrist=$(latest "$(ls -dt logs/rsl_rl/rubik_solve/*_${tag}_wrist_r$r | head -1)")
    for f in flick wrist; do [ -f $out/train_${f}_r$r.log ] && grep -E "succ_" $out/train_${f}_r$r.log | tail -10 | sed "s/^ */[train $f r$r] /" >> $out/summary.log; done
    evaluate $r
done
echo "[dagger] done $(date +%T) flick=$flick wrist=$wrist" | tee -a $out/summary.log
