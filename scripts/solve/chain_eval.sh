#!/usr/bin/env bash
# Chain metric without recovery: random chains of the given moves (compiled to skills, regrips included), no retries.
# Reports chain length before the first failure, per-skill success and the handoff-ready rate of every skill end.
#   scripts/solve/chain_eval.sh <tag> <flick_ckpt> <wrist_ckpt> [moves] [extra handoff_matrix args...]
#   wrist_ckpt "mix": flick_ckpt is a chain-trained checkpoint of both families (scripts/solve/train.py --mix)
cd "$(dirname "$0")/../.."
tag=$1 flick=$2 wrist=$3 moves=${4:-"U,U',R,R'"}; shift 4 2>/dev/null || shift $#
POL=(--policy flick=$flick --policy roll=$wrist --policy regrip=$wrist)
[ "$wrist" = mix ] && POL=(--policy all=$flick --set obs_next_skill=True)
MEM_MAX=20G scripts/run_isolated.sh logs/chain_$tag.log python -u scripts/solve/handoff_matrix.py --headless --num_envs ${NENV:-512} \
    --length 40 --moves "$moves" --stop_alive 0.03 --seed ${SEED:-200} --settle ${SETTLE:-0.3} --tag chain_$tag \
    --set "flick_legacy_ckpt='logs/rsl_rl/rubik_flick/2026-10-17_16-28-11/model_1900.pt'" "${POL[@]}" \
    --set link_contacts=False --set detent_torque=0.08 --set detent_width=0.2618 --mod ROLL_SHIFT=0.016 --mod ROLL_RING_LIFT=0.6 \
    --mod FLICK_PARTNER_LIFT=${LIFT:-0} "$@"
grep -E "chains complete|chain length|skill |end state" logs/chain_$tag.log | cut -c1-160
