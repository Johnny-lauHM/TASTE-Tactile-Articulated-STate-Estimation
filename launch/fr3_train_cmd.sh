#!/usr/bin/env bash
# FR3 RL fine-tune of the chain teacher: train.sh <run_name> <num_envs> <iterations> <checkpoint> [extra args]
cd "$(dirname "$0")/.."
n=$1; ne=$2; it=$3; ck=$4; shift 4
exec python -u scripts/solve/train.py --headless --skills hold --num_envs $ne --iterations $it --seed 1 --run_name $n \
 --ready_cache cache/solve_ready_ud_all4_flick.npz --checkpoint $ck --lr 5e-5 --desired_kl 0.008 --resume_optimizer 1 --fix_std 0.03 --entropy 0.0 --episode_s 20 \
 --set phase_horizon_s=3.0 --set chain_train=True --set obs_next_skill=True --set terminate_on_success=False --set link_contacts=False --set detent_torque=0.08 --set detent_width=0.2618 \
 --set "flick_legacy_ckpt='logs/rsl_rl/rubik_flick/2026-10-17_16-28-11/model_1900.pt'" --set "flick_legacy_kw={'latch_steps': 2, 'latch_min_s': 0.1}" --set "contact_model='tacmap'" \
 --set "arm='fr3'" --set "fr3_urdf_dir='assets_franka/left_m90'" --set "fr3_usd_dir='assets_franka/left_m90'" --set "fr3_layout='assets_franka/layout_left_m90_c3.json'" \
 --mod ROLL_SHIFT=0.016 --mod ROLL_RING_LIFT=0.6 --mod FLICK_PARTNER_LIFT=0 "$@"
