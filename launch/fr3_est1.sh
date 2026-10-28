#!/usr/bin/env bash
# estimator fine-tune in the FR3 env (left_m90 + c3), from soft_img6/est_2500 with the soft6 teacher
cd "$(dirname "$0")/.."
exec python -u scripts/solve/train_estimator.py --headless --teacher ${TEACHER:-logs/rsl_rl/rubik_solve/2026-10-23_00-09-56_soft6/model_599.pt} \
  --num_envs ${NE:-512} --iterations ${IT:-400} --horizon 32 --replay 8 --updates 4 --lr 3e-4 --hidden 384 --done_weight 0.5 --done_label success \
  --obs_key student_ee --img_feat 16 --out ${OUT:-logs/estimator/fr3_img1} --resume ${RESUME:-logs/estimator/soft_img6/est_2500.pt} \
  --set chain_linger_s=1.0 --set "contact_model='tacmap'" \
  --set "arm='fr3'" --set "fr3_urdf_dir='assets_franka/left_m90'" --set "fr3_usd_dir='assets_franka/left_m90'" --set "fr3_layout='assets_franka/layout_left_m90_c3.json'" \
  --mod ROLL_SHIFT=0.016 --mod ROLL_RING_LIFT=0.6 "$@"
