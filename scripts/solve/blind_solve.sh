#!/usr/bin/env bash
# blind full solves: sighted chain teacher (chain9 m100) acting on the tactile/proprio cube-state estimate, skills
# triggered by the estimator's completion head; ground truth only scores. No recovery, reliability planner. A premature
# completion does not end the run (--blind_continue: the robot cannot know); strict = solved runs without one.
#   [TEACHER=<ckpt>] [DONE_HOLD=n] scripts/solve/blind_solve.sh <estimator ckpt> <out dir> [num_envs=256] [extra execute.py args...]
cd "$(dirname "$0")/../.."
est=$1; out=$2; n=${3:-256}; shift 3 2>/dev/null || shift $#
python -u scripts/solve/execute.py --headless --num_envs $n --seed 8 \
    --policy all=${TEACHER:-logs/rsl_rl/rubik_solve/2026-10-21_15-04-51_chain9/model_100.pt} --estimator $est --trigger detector --done_hold ${DONE_HOLD:-3} \
    --set obs_next_skill=True --warmup 0.5 --skill_timeout 3.5 --ready_cache cache/solve_ready_ud_all4_flick.npz \
    --plan_reliability ${PLAN_FILES:-outputs/solve/c7_norecover/report.json,outputs/solve/handoff/chain_chain7_m300.json,outputs/solve/c7_rel/report.json} \
    --set "flick_legacy_kw={'latch_steps': 2, 'latch_min_s': 0.1}" --set "flick_legacy_ckpt='logs/rsl_rl/rubik_flick/2026-10-17_16-28-11/model_1900.pt'" --set link_contacts=False \
    --set detent_torque=0.08 --set detent_width=0.2618 --mod ROLL_SHIFT=0.016 --mod ROLL_RING_LIFT=0.6 --blind_continue --out $out "$@"
