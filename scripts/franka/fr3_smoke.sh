#!/usr/bin/env bash
# sighted soft6 solves with the hands on FR3 arms (--set arm='fr3'); waits for 6 GB of free GPU memory
cd "$(dirname "$0")/../.."
n=${1:-32}; out=${2:-outputs/solve/fr3_smoke}; shift 2 2>/dev/null
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -gt 6000 ]; do sleep 30; done
T6=logs/rsl_rl/rubik_solve/2026-10-23_00-09-56_soft6/model_599.pt
python -u scripts/solve/execute.py --headless --num_envs $n --seed 8 \
    --policy all=$T6 --set obs_next_skill=True --warmup 0.5 --skill_timeout 3.5 --ready_cache cache/solve_ready_ud_all4_flick.npz \
    --plan_reliability outputs/solve/c7_norecover/report.json,outputs/solve/handoff/chain_chain7_m300.json,outputs/solve/c7_rel/report.json \
    --set "flick_legacy_kw={'latch_steps': 2, 'latch_min_s': 0.1}" --set "flick_legacy_ckpt='logs/rsl_rl/rubik_flick/2026-10-17_16-28-11/model_1900.pt'" --set link_contacts=False \
    --set detent_torque=0.08 --set detent_width=0.2618 --mod ROLL_SHIFT=0.016 --mod ROLL_RING_LIFT=0.6 --set "contact_model='tacmap'" --set "arm='fr3'" \
    --timeout_continue --redo_missed --out $out "$@"
