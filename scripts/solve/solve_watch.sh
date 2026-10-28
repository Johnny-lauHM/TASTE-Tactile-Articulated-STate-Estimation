#!/usr/bin/env bash
# full solves (no recovery, 256 scrambles of seed 8, warm-up hold) for every new checkpoint of a chain run, one at a time
#   scripts/solve/solve_watch.sh <run_name>     summary: logs/chain_train/<run>_solves.txt
cd "$(dirname "$0")/../.."
run=$1
until ls -d logs/rsl_rl/rubik_solve/*_$run >/dev/null 2>&1; do sleep 30; done
dir=$(ls -d logs/rsl_rl/rubik_solve/*_$run | tail -1)
done_list=" "
while true; do
    for m in $(ls $dir | grep -oP 'model_\K[0-9]+(?=\.pt)' | sort -n); do
        [ "$m" -lt "${EVAL_FROM:-1}" ] && continue
        [[ "$done_list" == *" $m "* ]] && continue
        out=outputs/solve/${run}_m$m
        MEM_MAX=30G scripts/run_isolated.sh logs/solve_${run}_m$m.log python -u scripts/solve/execute.py --headless --num_envs 256 --seed 8 \
            --policy all=$dir/model_$m.pt --set obs_next_skill=True --warmup 0.5 --skill_timeout 3.5 --ready_cache cache/solve_ready_ud_all4_flick.npz ${PLAN:-} \
            --set "flick_legacy_ckpt='logs/rsl_rl/rubik_flick/2026-10-17_16-28-11/model_1900.pt'" --set link_contacts=False \
            --set detent_torque=0.08 --set detent_width=0.2618 --mod ROLL_SHIFT=0.016 --mod ROLL_RING_LIFT=0.6 --out $out
        [ -f $out/report.json ] || { sleep 120; continue; }  # e.g. out of GPU memory: retry
        python3 -c "import json; s=json.load(open('$out/report.json'))['summary']; print('m$m solved', s['solved'], '/', s['n'], s['status'], 'mean t', s['mean_time_s_solved'])" >> logs/chain_train/${run}_solves.txt
        done_list="$done_list$m "
    done
    sleep 60
done
