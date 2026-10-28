# TASTE: Tactile Articulated STate Estimation

**Solve a Rubik's cube by touch.** Two Sharpa Wave SE hands on Franka FR3 arms estimate the cube's pose and the angle of
its turning layer from fingertip touch and joint angles alone, and solve the cube with no camera in the loop.

Project page: **https://johnny-lauhm.github.io/TASTE-Tactile-Articulated-STate-Estimation/**

> Status: research code, simulation only (Isaac Lab). No real-robot result yet. The paper is in preparation.

## Method

| Stage | What it does | Code |
|---|---|---|
| Sense | finger joint angles and velocities, fingertip force/torque, a 30×30 TacMap tactile image per pad, the commanded skill | `SolveEnv._est_obs`, `wave_se_tactile.py` |
| Estimate | per-pad CNN + GRU (384) → cube position, orientation, layer angle, their rates, and P(skill complete) | `solve/estimator.py`, `scripts/solve/train_estimator.py` |
| Act | PPO skill teacher (trained with privileged state) and regrasp servo, fed the estimate via `SolveEnv.perceived()` | `solve/env.py`, `scripts/solve/train.py` |
| Plan | Kociemba solution compiled into the hands' skills (U/D flicks, R/L wrist rolls, x turns, regrasps), reliability-aware | `solve/cube_logic.py`, `solve/skills.py` |
| Execute | skill state machine; blind runs trigger on the estimator's completion head, ground truth only scores | `scripts/solve/execute.py`, `scripts/solve/blind_solve.sh` |

The estimator is trained **in the loop**: rollouts are driven by its own estimates (DAgger-style, control on the true
state annealed to zero).

## Results (simulation, 256 random scrambles per entry)

| Condition | Sighted (privileged state) | Blind (TASTE) |
|---|---|---|
| Nominal, held-out scrambles | 90.6% | 83.2% |
| Randomized dynamics and sensing, held-out scrambles | 57.0% | 39.5% |

Median estimation error: position 1.6 mm, orientation 1.4°, layer angle 0.3°. See the project page for details and
limitations (idealized cube, latency sensitivity, and an open audit of how the blind orientation is anchored at skill
starts: `goal_from_perceived` in `env_cfg.py`).

## Repository layout

```
source/rubik_lab/rubik_lab/
  solve/env.py, env_cfg.py   SolveEnv (DirectRLEnv): skills, completion criteria, sim2real (obs_real, dr, tac_*, cube_vis)
  solve/estimator.py         TASTE estimator (CubeEstimator, Estimator wrapper)
  solve/cube_logic.py        hand-frame cube logic, solver, skill compiler
  wave_se_tactile.py         Sharpa Wave SE tactile model (TacMap images, pad F/T)
  fr3_arm.py, assets.py      FR3 + Wave SE articulation configs
scripts/solve/               train.py (RL), train_estimator.py, execute.py (full solves), replay.py (dashboard videos)
scripts/franka/              building the FR3 + Sharpa Wave assets
launch/                      launchers used for the reported runs (sighted execution, RL fine-tune, estimator)
tests/                       unit tests (cube logic, binding, ...)
assets/rubik_cube/           cube USD (scripts/gen_rubik_cube_usd.py); assets/tacmap_maps: maps from sharpa-tacmap (Apache-2.0)
```

## Setup

Isaac Sim 5.1, Isaac Lab v2.3.2, PyTorch 2.7 (cu128), rsl-rl-lib 3.1.2, Python 3.11:

```bash
bash scripts/setup_env.sh          # creates the `sharpa` conda env and installs source/rubik_lab
conda activate sharpa
export OMNI_KIT_ACCEPT_EULA=YES
python -m pytest -q tests/test_solve_logic.py
```

**Not included:** robot assets (Sharpa Wave SE from
[sharpa-urdf-usd-xml](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml), Franka FR3 description; both Apache-2.0), trained
checkpoints, the ready-grip caches (`cache/`), and run logs. `scripts/franka/` builds the arm + hand USDs; the
generated `.usda` files reference absolute paths that must be regenerated on a new machine.

## Usage (once assets and checkpoints are in place)

```bash
# sighted full solves (privileged state), N scrambles
SEED=21 launch/fr3_exec2.sh <teacher.pt> outputs/solve/sighted 256
# blind full solves with a TASTE estimator
TEACHER=<teacher.pt> DONE_HOLD=12 bash scripts/solve/blind_solve.sh <estimator.pt> outputs/solve/blind 256 \
  --set "contact_model='tacmap'" --timeout_continue --redo_missed
# estimator training in the loop
TEACHER=<teacher.pt> RESUME=<estimator.pt> OUT=logs/estimator/mine launch/fr3_est1.sh --set obs_real=True --set dr=True
```

Sim-to-real switches (all off by default): `obs_real`, `dr`, `dr_hand_delay`, `tac_rate_p`/`tac_lag`, `cube_vis`
(camera measurement model) with `execute.py --trigger vision`. See `env_cfg.py`.

## Citation

```bibtex
@misc{taste2026,
  title  = {TASTE: Tactile Articulated State Estimation for Blind Bimanual Rubik's Cube Solving},
  year   = {2026},
  howpublished = {\url{https://github.com/Johnny-lauHM/TASTE-Tactile-Articulated-STate-Estimation}}
}
```

## License

The code is released under the [MIT License](LICENSE). Bundled third-party files keep their own licenses:
`assets/tacmap_maps/` (Sharpa TacMap, Apache-2.0, see its LICENSE and NOTICE) and `assets/reference/robogym/` (OpenAI
robogym, MIT).
