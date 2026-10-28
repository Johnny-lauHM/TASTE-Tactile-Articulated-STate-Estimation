"""Train skill policies (privileged teacher) in the full-solve skill environment.

    python scripts/solve/train.py --headless --skills "R,R',L,L'" --run_name roll --num_envs 2048 --iterations 1500
"""
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--skills", default="U,U'")
p.add_argument("--weights", default="")
p.add_argument("--num_envs", type=int, default=2048)
p.add_argument("--iterations", type=int, default=1500)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--run_name", default="skill")
p.add_argument("--ready_cache", default="cache/solve_ready_udr_v1.npz")
p.add_argument("--checkpoint", default=None)
p.add_argument("--mix", nargs=2, default=None, metavar=("FLICK", "WRIST"),
               help="train both families together (solve.mix_policy) from these single-family checkpoints")
p.add_argument("--gamma", type=float, default=None)
p.add_argument("--lr", type=float, default=None, help="initial learning rate (adaptive schedule)")
p.add_argument("--desired_kl", type=float, default=None)
p.add_argument("--resume_optimizer", type=int, default=0, help="1: load the checkpoint's optimizer state too")
p.add_argument("--fix_std", type=float, default=0.0, help=">0: fixed action noise std (not learned) for every family")
p.add_argument("--entropy", type=float, default=None, help="entropy coefficient (0: the action noise does not grow)")
p.add_argument("--buffer_size", type=int, default=0)
p.add_argument("--buffer_prob", type=float, default=0.5)
p.add_argument("--episode_s", type=float, default=3.0)
p.add_argument("--init_std", type=float, default=0.08, help="initial exploration std (action units)")
p.add_argument("--trick_std", type=float, default=0.0, help=">0: initial std of both index fingers' joints (flicks)")
p.add_argument("--set", action="append", default=[], help="cfg overrides key=value (python literal)")
p.add_argument("--mod", action="append", default=[], help="constants of rubik_lab.solve.skills, e.g. ROLL_SHIFT=0.016")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app
import sys as _sys, traceback as _tb, os as _os
def _die(t, v, b):
    _tb.print_exception(t, v, b); _sys.stdout.flush(); _sys.stderr.flush(); _os._exit(1)
_sys.excepthook = _die  # the simulation app hangs on close after an exception

import ast, hashlib, json, os, shutil, signal
from datetime import datetime
from pathlib import Path
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rubik_lab.solve import skills as _S
for _kv in a.mod:
    _k, _v = _kv.split("=", 1)
    setattr(_S, _k, ast.literal_eval(_v))
from rubik_lab.solve.env import SolveEnv
from rubik_lab.solve.env_cfg import SolveEnvCfg
from rubik_lab.solve.agent_cfg import SolvePPOCfg

cfg = SolveEnvCfg()
cfg.scene.num_envs, cfg.seed = a.num_envs, a.seed
cfg.skills = tuple(a.skills.split(","))
if a.weights:
    cfg.skill_weights = [float(x) for x in a.weights.split(",")]
cfg.ready_cache = a.ready_cache
cfg.buffer_size, cfg.buffer_prob = a.buffer_size, a.buffer_prob
cfg.episode_length_s = a.episode_s
cfg.phase_horizon_s = a.episode_s
for kv in a.set:
    k, v = kv.split("=", 1)
    setattr(cfg, k, ast.literal_eval(v))
agent = SolvePPOCfg()
agent.seed = a.seed
if a.gamma is not None:
    agent.algorithm.gamma = a.gamma
if a.lr is not None:
    agent.algorithm.learning_rate = a.lr
if a.desired_kl is not None:
    agent.algorithm.desired_kl = a.desired_kl
if a.entropy is not None:
    agent.algorithm.entropy_coef = a.entropy
root = Path(__file__).resolve().parents[2]
run = root / f"logs/rsl_rl/rubik_solve/{datetime.now():%Y-%m-%d_%H-%M-%S}_{a.run_name}"
run.mkdir(parents=True, exist_ok=False)
for rel in ["scripts/solve/train.py"] + [f"source/rubik_lab/rubik_lab/solve/{f}" for f in os.listdir(root / "source/rubik_lab/rubik_lab/solve") if f.endswith(".py")]:
    dst = run / "source" / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / rel, dst)
(run / "env.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str))
(run / "agent.json").write_text(json.dumps(agent.to_dict(), indent=2, default=str))
(run / "args.json").write_text(json.dumps(vars(a), indent=2, default=str))
print(f"[solve_train] run {run}", flush=True)
env = SolveEnv(cfg)
wenv = RslRlVecEnvWrapper(env, clip_actions=1.0)
agent_d = agent.to_dict()
if cfg.cube_est_ckpt or cfg.cube_vis:  # blind / camera fine-tuning: the actor acts on the estimated cube state, the critic sees the true one
    agent_d["obs_groups"] = {"policy": ["policy"], "critic": ["critic"]}
if a.mix or (a.checkpoint and "fam.0.log_std" in __import__("torch").load(a.checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]):
    import rsl_rl.runners.on_policy_runner as _opr
    from rubik_lab.solve.mix_policy import MixActorCritic
    _opr.MixActorCritic = MixActorCritic  # the runner resolves policy class names in its own module
    agent_d["policy"]["class_name"] = "MixActorCritic"
runner = OnPolicyRunner(wenv, agent_d, log_dir=str(run), device=env.device)
if a.mix:
    runner.alg.policy.load_families(*a.mix)
elif a.checkpoint:
    # a chain checkpoint resumes with its optimizer state (a fresh Adam + the initial learning rate kicks the policy:
    # chain1 -> chain2 fell from 24 to 14 skills per chain in the first 100 iterations)
    runner.load(a.checkpoint, load_optimizer=bool(a.resume_optimizer))
    runner.current_learning_iteration = 0
else:
    import math
    import torch
    pol = runner.alg.policy
    with torch.no_grad():
        # residual policy: start exactly at the scripted nominal (zero mean action), explore little
        last = [m for m in pol.actor.modules() if isinstance(m, torch.nn.Linear)][-1]
        last.weight.mul_(0.01)
        last.bias.zero_()
        std = torch.full((56,), a.init_std)
        if a.trick_std > 0:
            for h in range(2):
                std[h * 28 + 6 + 5 : h * 28 + 6 + 9] = a.trick_std  # index MCP_FE, MCP_AA, PIP, DIP
        pol.log_std.copy_(std.log().to(pol.log_std.device))
if a.fix_std > 0:  # exploration noise held at this value (evaluation is deterministic; the learned std drifted up)
    import math as _m
    import torch as _t
    _pol = runner.alg.policy
    for _f in (_pol.fam if hasattr(_pol, "fam") else [_pol]):
        with _t.no_grad():
            _f.log_std.fill_(_m.log(a.fix_std))
        _f.log_std.requires_grad_(False)
    print(f"[solve_train] action noise fixed at std {a.fix_std}", flush=True)
def stop(signum, frame):
    raise KeyboardInterrupt
signal.signal(signal.SIGTERM, stop)
try:
    runner.learn(num_learning_iterations=a.iterations, init_at_random_ep_len=False)
except KeyboardInterrupt:
    runner.save(str(run / "model_interrupted.pt"))
print(f"[solve_train] finished {run}", flush=True)
os._exit(0)
