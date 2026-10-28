"""Tactile estimability data: the sighted chain teacher runs random skill chains (chain_train env: next skill on
success, restart after a failure); every control step records what the hands can sense and the privileged truth.

Recorded per step (float16, time-major (T, N, .)):
    student  : the student observation (proprio, delivered NORMAL-only pad F/T with latency / noise, skill, phase, roll,
               previous action, next skill)
    wrench   : instantaneous full pad wrench incl. friction (shear), pad frame (2 hands x 5 pads x 6)
    centroid : force-weighted contact point in the pad frame [mm] + touching flag (2 x 5 x 4)
    labels   : cube offset from its ready-grip pose [mm] (3), its rotation off the ready grip modulo symmetry, axis-angle
               [deg] (3), layer error to the skill's target [deg], layer residual to the nearest quarter [deg], done
               window (completion criteria met), skill index, time in skill [s], hand rolls (2)
    reset    : env (re)started its chain at this step (estimator hidden states restart there)

    python scripts/solve/collect_tactile.py --headless --policy <chain ckpt> --num_envs 512 --steps 3000 --out cache/tactile/probe.pt
"""
import argparse
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--policy", required=True)
p.add_argument("--num_envs", type=int, default=512)
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--out", default="cache/tactile/probe.pt")
p.add_argument("--set", action="append", default=[])
p.add_argument("--mod", action="append", default=[])
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app
import sys as _sys, traceback as _tb
def _die(t, v, b):
    _tb.print_exception(t, v, b); _sys.stdout.flush(); os._exit(1)
_sys.excepthook = _die
import ast

import torch

import isaaclab.utils.math as mu
from rubik_lab.solve import skills as S

for kv in a.mod:
    k, v = kv.split("=", 1)
    setattr(S, k, ast.literal_eval(v))
from rubik_lab.solve.binding import quarter_residual
from rubik_lab.solve.env import SolveEnv
from rubik_lab.solve.env_cfg import SolveEnvCfg
from rubik_lab.solve.policies import load_policies

cfg = SolveEnvCfg()
cfg.scene.num_envs = a.num_envs
cfg.ready_cache = "cache/solve_ready_ud_all4_flick.npz"
cfg.episode_length_s = 20.0
cfg.chain_train = True
cfg.obs_next_skill = True
cfg.terminate_on_success = False
cfg.skills = ("hold",)
cfg.link_contacts = False
cfg.detent_torque, cfg.detent_width = 0.08, 0.2618
cfg.flick_legacy_ckpt = "logs/rsl_rl/rubik_flick/2026-10-17_16-28-11/model_1900.pt"
cfg.tactile.contact_cfg = cfg.tactile.contact_cfg.replace(track_friction_forces=True)
for kv in a.set:
    k, v = kv.split("=", 1)
    setattr(cfg, k, ast.literal_eval(v))
env = SolveEnv(cfg)
dev, n = env.device, env.num_envs
obs, _ = env.reset()
pol = load_policies(env, {"all": a.policy})["all"]
tac = env.tactile


def centroid():
    """force-weighted contact point of each pad in its frame [mm] and a touching flag: (N, 2, 5, 4)."""
    out = []
    for s in tac.SIDES:
        d = env.hands[s].data
        pad_p, pad_q = d.body_pos_w[:, tac.pad_body_ids[s]], d.body_quat_w[:, tac.pad_body_ids[s]]
        f_b = torch.cat([c.data.force_matrix_w for c in tac.contact[s]], 1).norm(dim=-1)  # (N, 5, 2)
        p_b = torch.nan_to_num(torch.cat([c.data.contact_pos_w for c in tac.contact[s]], 1), nan=0.0)  # (N, 5, 2, 3)
        w = f_b.sum(-1, keepdim=True)
        c = (p_b * f_b.unsqueeze(-1)).sum(2) / w.clamp(min=1e-6)
        c = mu.quat_apply_inverse(pad_q.reshape(-1, 4), (c - pad_p).reshape(-1, 3)).view_as(c)
        touch = (w > 0.05).float()
        out.append(torch.cat([c * 1000.0 * touch, touch], -1))
    return torch.stack(out, 1)


def labels():
    dp, dq = env._cube_deviation()
    twist = env.cube.data.joint_pos[:, env.twist_id]
    return torch.cat([dp * 1000.0, torch.rad2deg(mu.axis_angle_from_quat(dq)),
                      torch.rad2deg(mu.wrap_to_pi(env.layer_target - twist)).unsqueeze(-1),
                      torch.rad2deg(quarter_residual(twist)).unsqueeze(-1),
                      (env.stable_steps > 0).float().unsqueeze(-1), env.skill.float().unsqueeze(-1),
                      env.t_skill.unsqueeze(-1), env.roll], -1)


rec = {k: [] for k in ("student", "wrench", "centroid", "labels", "reset")}
reset = torch.ones(n, dtype=torch.bool, device=dev)
for t in range(a.steps):
    with torch.no_grad():
        rec["student"].append(obs["student"].half().cpu())
        rec["wrench"].append(tac.raw_cube_wrench(check=False).flatten(1).half().cpu())
        rec["centroid"].append(centroid().flatten(1).half().cpu())
        rec["labels"].append(labels().half().cpu())
        rec["reset"].append(reset.cpu())
        obs, _, term, trunc, _ = env.step(pol(obs["policy"]))
        reset = term.clone()  # a failed chain restarts from the ready grip (time-outs continue the chain)
    if t % 500 == 0:
        print(f"[collect] step {t}/{a.steps}", flush=True)
data = {k: torch.stack(v, 0) for k, v in rec.items()}
data["student_layout"] = dict(per_hand=102, ft_slice=(72, 102), n_hands=2)
data["label_names"] = ["dx_mm", "dy_mm", "dz_mm", "rx_deg", "ry_deg", "rz_deg", "layer_err_deg", "layer_resid_deg", "done",
                       "skill", "t_skill", "roll_l", "roll_r"]
data["policy"] = a.policy
os.makedirs(os.path.dirname(a.out), exist_ok=True)
torch.save(data, a.out)
print(f"[collect] {a.steps} steps x {n} envs -> {a.out}: " + ", ".join(f"{k} {tuple(v.shape)}" for k, v in data.items() if torch.is_tensor(v)), flush=True)
os._exit(0)
