#!/usr/bin/env bash
# FR3 executor with PLAN / REC overrides (empty = no reliability planning / no recovery flags): as fr3_exec.sh
cd "$(dirname "$0")/.."
T=$1; out=$2; n=$3; shift 3
exec python -u scripts/solve/execute.py --headless --num_envs $n --seed ${SEED:-8} \
    --policy all=$T --set obs_next_skill=True --warmup 0.5 --skill_timeout 3.5 --ready_cache cache/solve_ready_ud_all4_flick.npz \
    ${PLAN---plan_reliability outputs/solve/c7_norecover/report.json,outputs/solve/handoff/chain_chain7_m300.json,outputs/solve/c7_rel/report.json} \
    --set "flick_legacy_kw={'latch_steps': 2, 'latch_min_s': 0.1}" --set "flick_legacy_ckpt='logs/rsl_rl/rubik_flick/2026-10-17_16-28-11/model_1900.pt'" --set link_contacts=False \
    --set detent_torque=0.08 --set detent_width=0.2618 --mod ROLL_SHIFT=0.016 --mod ROLL_RING_LIFT=0.6 --set "contact_model='tacmap'" \
    --set "arm='fr3'" --set "fr3_urdf_dir='assets_franka/left_m90'" --set "fr3_usd_dir='assets_franka/left_m90'" --set "fr3_layout='assets_franka/layout_left_m90_c3.json'" \
    ${REC---timeout_continue --redo_missed} --out $out "$@"
