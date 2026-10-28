"""Train one full-cycle primitive in a new run, optionally warm-starting a legacy policy.

Unlike --resume, warm-start loads weights/normalizers but resets the optimizer and
iteration count. No historical run or checkpoint is overwritten.
"""
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--task", default="Rubik-Cycle-U-Sharpa-v0")
p.add_argument("--checkpoint", default=None)
p.add_argument("--num_envs", type=int, default=1024)
p.add_argument("--iterations", type=int, default=500)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--run_name", default="full_cycle")
p.add_argument("--log_dir", default=None)
p.add_argument("--cache", default=None)
p.add_argument("--noise_std", type=float, default=None)
p.add_argument("--learning_rate", type=float, default=None)
p.add_argument("--save_interval", type=int, default=25)
p.add_argument("--demo_data", default=None)
p.add_argument("--demo_steps", type=int, default=10)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app

from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import rubik_lab

cfg = load_cfg_from_registry(a.task, "env_cfg_entry_point")
agent = load_cfg_from_registry(a.task, "rsl_rl_cfg_entry_point")
if not hasattr(cfg, "cycle_stable_s"):
    raise ValueError("Use a versioned full-cycle task")
cfg.scene.num_envs, cfg.seed = a.num_envs, a.seed
if a.cache:
    cfg.grasp_cache_path = a.cache
agent.seed = a.seed
agent.save_interval = a.save_interval
if a.learning_rate is not None:
    agent.algorithm.learning_rate = a.learning_rate
    agent.algorithm.schedule = "fixed"
run = Path(a.log_dir or f"logs/rsl_rl/rubik_primitive_cycle/{datetime.now():%Y-%m-%d_%H-%M-%S}_{a.run_name}")
run.mkdir(parents=True, exist_ok=False)
root = Path(__file__).resolve().parents[1]
files = ["scripts/train_primitive.py", "source/rubik_lab/rubik_lab/primitive_cycle.py",
         "source/rubik_lab/rubik_lab/demonstration_anchor.py",
         "source/rubik_lab/rubik_lab/tasks/bimanual_twist/cycle_env.py",
         "source/rubik_lab/rubik_lab/tasks/bimanual_twist/cycle_cfg.py",
         "source/rubik_lab/rubik_lab/tasks/bimanual_twist/env.py",
         "source/rubik_lab/rubik_lab/tasks/bimanual_twist/env_cfg.py"]
manifest = {"task": a.task, "warm_start": a.checkpoint, "seed": a.seed, "iterations": a.iterations,
            "actor_observations": "privileged_training_prototype", "noise_std_override": a.noise_std, "files": {}}
for rel in files:
    src = root / rel
    target = run / "source" / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, target)
    manifest["files"][rel] = hashlib.sha256(src.read_bytes()).hexdigest()
cache = root / cfg.grasp_cache_path
manifest["cache_sha256"] = hashlib.sha256(cache.read_bytes()).hexdigest()
if a.checkpoint:
    manifest["warm_start_sha256"] = hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
(run / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
(run / "env.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str)+"\n")
(run / "agent.json").write_text(json.dumps(agent.to_dict(), indent=2, default=str)+"\n")
print(f"[cycle_train] run {run}", flush=True)
env = gym.make(a.task, cfg=cfg)
wenv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = OnPolicyRunner(wenv, agent.to_dict(), log_dir=str(run), device=env.unwrapped.device)
if a.checkpoint:
    runner.load(a.checkpoint, load_optimizer=False)
    runner.current_learning_iteration = 0
if a.noise_std is not None:
    import math
    import torch
    if a.noise_std <= 0:
        raise ValueError("noise_std must be positive")
    with torch.no_grad():
        runner.alg.policy.log_std.fill_(math.log(a.noise_std))
if a.demo_data:
    from rubik_lab.demonstration_anchor import attach_anchor
    report_path = Path(a.checkpoint).parent / 'report.json' if a.checkpoint else None
    excluded = json.loads(report_path.read_text()).get('val_grasps',[]) if report_path and report_path.exists() else []
    rehearsal = attach_anchor(runner.alg,a.demo_data,excluded_grasps=excluded,steps=a.demo_steps)
    rehearsal['sha256'] = hashlib.sha256(Path(a.demo_data).read_bytes()).hexdigest()
    (run/'rehearsal.json').write_text(json.dumps(rehearsal,indent=2)+'\n')
import signal
def request_training_stop(signum, frame):
    raise KeyboardInterrupt
signal.signal(signal.SIGINT, request_training_stop)
signal.signal(signal.SIGTERM, request_training_stop)
try:
    runner.learn(num_learning_iterations=a.iterations, init_at_random_ep_len=False)
except KeyboardInterrupt:
    runner.save(str(run / 'model_interrupted.pt'))
    print(f"[cycle_train] interrupted; saved {run / 'model_interrupted.pt'}", flush=True)
else:
    print(f"[cycle_train] finished {run}", flush=True)
env.close()
app.close()
