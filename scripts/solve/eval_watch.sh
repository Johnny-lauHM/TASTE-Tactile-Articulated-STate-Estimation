#!/usr/bin/env bash
# evaluate every new checkpoint of a chain run on the full chains (warm-up hold, no settle), one at a time
#   scripts/solve/eval_watch.sh <run_name> [moves]     results: logs/chain_<run>_m<iter>.log, summary logs/chain_train/<run>_evals.txt
cd "$(dirname "$0")/../.."
run=$1 moves=${2:-"U,U',R,R',L,L',x,x'"}
until ls -d logs/rsl_rl/rubik_solve/*_$run >/dev/null 2>&1; do sleep 30; done
dir=$(ls -d logs/rsl_rl/rubik_solve/*_$run | tail -1)
done_list=" "
while true; do
    for m in $(ls $dir | grep -oP 'model_\K[0-9]+(?=\.pt)' | sort -n); do
        [ "$m" -lt "${EVAL_FROM:-1}" ] && continue
        [[ "$done_list" == *" $m "* ]] && continue
        SETTLE=0 scripts/solve/chain_eval.sh ${run}_m$m $dir/model_$m.pt mix "$moves" --warmup 0.5 ${EVAL_EXTRA:-} > /dev/null
        grep -q "chain length" logs/chain_${run}_m$m.log || { sleep 120; continue; }  # e.g. out of GPU memory: retry
        echo "m$m $(grep -h 'chain length' logs/chain_${run}_m$m.log | cut -c60-) | $(grep -h 'moves done' logs/chain_${run}_m$m.log | cut -c40-90) $(grep -h 'chains complete' logs/chain_${run}_m$m.log | grep -oP 'complete \S+')" >> logs/chain_train/${run}_evals.txt
        done_list="$done_list$m "
    done
    grep -q "finished" logs/chain_train/train${run#chain}.log 2>/dev/null && [ -f $dir/model_$(($(grep -oP 'iterations \K[0-9]+' $dir/../../../../logs/chain_train/run2.sh 2>/dev/null || echo 0))).pt ] && break
    sleep 60
done
