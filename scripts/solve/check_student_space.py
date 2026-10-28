"""Is the student's action space lossless? The same deterministic chain teacher runs chain_train chains once in teacher
mode and once executed through student_mode (teacher_mask all True): skills done per failure must match.

    python scripts/solve/check_student_space.py --headless --teacher <ckpt> --student_mode 1
"""
import argparse
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--teacher", required=True)
p.add_argument("--student_mode", type=int, default=1)
p.add_argument("--num_envs", type=int, default=1024)
p.add_argument("--steps", type=int, default=1500)
p.add_argument("--scale", default="", help="student_scale override, e.g. 0.05,0.8,3.0")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app
import torch

from rubik_lab.solve import skills as S

S.ROLL_SHIFT, S.ROLL_RING_LIFT = 0.016, 0.6
from rubik_lab.solve.env import SolveEnv
from rubik_lab.solve.env_cfg import SolveEnvCfg
from rubik_lab.solve.policies import load_policies

cfg = SolveEnvCfg()
cfg.scene.num_envs = a.num_envs
cfg.ready_cache = "cache/solve_ready_ud_all4_flick.npz"
cfg.episode_length_s = 20.0
cfg.chain_train, cfg.obs_next_skill, cfg.student_mode = True, True, bool(a.student_mode)
cfg.terminate_on_success, cfg.skills, cfg.link_contacts = False, ("hold",), False
cfg.detent_torque, cfg.detent_width = 0.08, 0.2618
cfg.flick_legacy_ckpt = "logs/rsl_rl/rubik_flick/2026-10-17_16-28-11/model_1900.pt"
if a.scale:
    cfg.student_scale = tuple(float(x) for x in a.scale.split(","))
env = SolveEnv(cfg)
clip_n = torch.zeros(56, device=env.device)
clip_t = 0
obs, _ = env.reset()
teacher = load_policies(env, {"all": a.teacher})["all"]
env.teacher_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
ok = fail = 0
fails_by = {}
for t in range(a.steps):
    with torch.no_grad():
        act = teacher(obs["policy"])
        env.teacher_raw = act
        cd = env.chain_done.clone()
        k_before = env.skill.clone()
        obs, _, term, _, _ = env.step(act)
    clip_n += (env.teacher_free.abs() > 0.999).float().sum(0)
    clip_t += env.num_envs
    if t > 200:  # past the start-up (chain_done counts the skills completed since the chain's restart)
        ok += int((env.chain_done - cd).clamp(min=0)[~term].sum())
        fail += int(term.sum())
        for k in k_before[term].tolist():
            fails_by[S.SKILLS[k]] = fails_by.get(S.SKILLS[k], 0) + 1
g = (clip_n / clip_t).view(2, 28)
print(f"[check] label clip rate: wrist pos {g[:, :3].mean():.4f} rot {g[:, 3:6].mean():.4f} fingers {g[:, 6:].mean():.4f} (max finger dim {g[:, 6:].max():.4f})", flush=True)
print(f"[check] student_mode {a.student_mode}: skills ok {ok}, failures {fail}, skills per failure {ok / max(fail, 1):.1f}, failing skills {sorted(fails_by.items(), key=lambda kv: -kv[1])}", flush=True)
os._exit(0)
