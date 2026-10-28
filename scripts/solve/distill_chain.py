"""DAgger distillation of the sighted chain teacher into a blind recurrent student, in skill chains.

The env runs in student_mode: the student's action sets the targets around the BLIND nominal (no regrip servo, no
legacy flick policy); the label is the sighted teacher system's target in that space (env.teacher_free), so the student
has to infer from touch / proprioception where the cube sits (the servo's offset) and how far a layer turned. With
probability beta per env-step the teacher's targets are executed instead (decays to 0); skills advance on the true
completion (chain_train). The student also predicts the completion window (done) and the cube offset from its
ready-grip pose (aux). Hidden states run through whole chains and restart with the env's chain.

With beta = 0 the env's chain statistics (succ_*, chain_len) are the blind student's own.

    python scripts/solve/distill_chain.py --headless --teacher <chain ckpt> --num_envs 4096 --iterations 3000 --out logs/student/blind1
"""
import argparse
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--teacher", required=True)
p.add_argument("--num_envs", type=int, default=4096)
p.add_argument("--iterations", type=int, default=3000)
p.add_argument("--horizon", type=int, default=32)
p.add_argument("--beta_iters", type=int, default=400, help="iterations over which teacher execution decays to 0")
p.add_argument("--beta_hold", type=int, default=0, help="iterations of pure teacher execution before the decay")
p.add_argument("--replay", type=int, default=0, help="keep this many recent batches; every update also trains on one of them")
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--hidden", type=int, default=384)
p.add_argument("--done_weight", type=float, default=0.5)
p.add_argument("--aux_weight", type=float, default=0.5)
p.add_argument("--out", default="logs/student/blind1")
p.add_argument("--resume", default=None)
p.add_argument("--eval_only", action="store_true", help="with --resume: the student acts alone (beta 0), no updates, nothing saved")
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
import json
import time

import torch
import torch.nn.functional as F

import isaaclab.utils.math as mu
from rubik_lab.solve import skills as S

for kv in a.mod:
    k, v = kv.split("=", 1)
    setattr(S, k, ast.literal_eval(v))
from rubik_lab.solve.env import SolveEnv
from rubik_lab.solve.env_cfg import SolveEnvCfg
from rubik_lab.solve.policies import load_policies
from rubik_lab.solve.student import StudentNet

cfg = SolveEnvCfg()
cfg.scene.num_envs = a.num_envs
cfg.ready_cache = "cache/solve_ready_ud_all4_flick.npz"
cfg.episode_length_s = 20.0
cfg.chain_train = True
cfg.obs_next_skill = True
cfg.student_mode = True
cfg.terminate_on_success = False
cfg.skills = ("hold",)
cfg.link_contacts = False
cfg.detent_torque, cfg.detent_width = 0.08, 0.2618
cfg.flick_legacy_ckpt = "logs/rsl_rl/rubik_flick/2026-10-17_16-28-11/model_1900.pt"
for kv in a.set:
    k, v = kv.split("=", 1)
    setattr(cfg, k, ast.literal_eval(v))
env = SolveEnv(cfg)
dev, n = env.device, env.num_envs
obs, _ = env.reset()
teacher = load_policies(env, {"all": a.teacher})["all"]
kw = dict(obs_dim=obs["student"].shape[-1], hidden=a.hidden, err_dim=6)
net = StudentNet(**kw).to(dev)
opt = torch.optim.Adam(net.parameters(), lr=a.lr)
start_it = 0
if a.resume:
    blob = torch.load(a.resume, map_location=dev, weights_only=False)
    net.load_state_dict(blob["student_state_dict"])
    opt.load_state_dict(blob["optimizer"])
    start_it = blob["iteration"]
os.makedirs(a.out, exist_ok=True)
if not a.eval_only:
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
log_f = open(os.path.join(a.out, "log.jsonl"), "a")
X_SKILLS = torch.tensor([S.SKILL_INDEX["x"], S.SKILL_INDEX["x'"]], device=dev)
act_std = torch.ones(56, device=dev) * 0.1  # running label scale per dimension (the loss is in units of it)


def aux_label():
    dp, dq = env._cube_deviation()
    return torch.cat([dp * 200.0, mu.axis_angle_from_quat(dq) * 10.0], -1)  # 5 mm, 0.1 rad units


replay_buf = []
h = net.init_hidden(n, dev)
reset_next = torch.ones(n, dtype=torch.bool, device=dev)
t0 = time.time()
for it in range(start_it, start_it + a.iterations):
    beta = 0.0 if a.eval_only else min(1.0, max(0.0, 1.0 - (it - a.beta_hold) / max(1, a.beta_iters)))
    O, A, D, E, R, M, PM = [], [], [], [], [], [], []
    h0 = h.detach()
    with torch.no_grad():
        for t in range(a.horizon):
            so = obs["student"]
            if not a.eval_only:
                net.norm.update(so)
            env.teacher_raw = teacher(obs["policy"])
            env.teacher_mask = torch.rand(n, device=dev) < beta
            # labels describing the state of this observation
            D.append((env.stable_steps > 0).float())
            E.append(aux_label())
            PM.append((~torch.isin(env.skill, X_SKILLS)).float())  # cube offset undefined mid-x
            M.append((~env.chain_warm).float())
            sa, _, _, h = net(so.unsqueeze(0), h, reset_next.unsqueeze(0))
            O.append(so)
            R.append(reset_next.clone())
            obs, _, term, trunc, _ = env.step(sa[0])
            A.append(env.teacher_free.clone())  # computed in this step from the observed state
            reset_next = term.clone()  # chains restart only on failure (time-outs continue)
    O, A, D, E, R, M, PM = (torch.stack(x, 0) for x in (O, A, D, E, R, M, PM))
    act_std = 0.99 * act_std + 0.01 * A[M > 0].std(0).clamp(min=0.01)

    def batch_loss(O, A, D, E, R, M, PM, h0):
        pa, pd, pe, _ = net(O, h0, R)
        w = M.unsqueeze(-1)
        loss_a = (((pa - A) / act_std).square() * w).sum() / (w.sum() * 56 + 1)
        loss_d = (F.binary_cross_entropy_with_logits(pd, D, reduction="none") * M).sum() / (M.sum() + 1)
        pm = (M * PM).unsqueeze(-1)
        loss_e = (F.smooth_l1_loss(pe, E, reduction="none") * pm).sum() / (pm.sum() * 6 + 1)
        return loss_a, loss_d, loss_e, pd, pe, pm

    loss_a, loss_d, loss_e, pd, pe, pm = batch_loss(O, A, D, E, R, M, PM, h0)
    loss = loss_a + a.done_weight * loss_d + a.aux_weight * loss_e
    if a.replay > 0:  # an older batch too (DAgger aggregation over the recent distribution, against forgetting)
        if len(replay_buf) and it % 1 == 0:
            ra, rd, re, *_ = batch_loss(*replay_buf[int(torch.randint(len(replay_buf), (1,)))])
            loss = loss + ra + a.done_weight * rd + a.aux_weight * re
        replay_buf.append((O, A, D, E, R, M, PM, h0))
        if len(replay_buf) > a.replay:
            replay_buf.pop(0)
    if not a.eval_only:
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    if it % 10 == 0:
        with torch.no_grad():
            pred = (pd > 0).float()
            tp = (pred * D * M).sum()
            prec = (tp / (pred * M).sum().clamp(min=1)).item()
            rec = (tp / (D * M).sum().clamp(min=1)).item()
            off_mm = ((pe[..., :3] - E[..., :3]).norm(dim=-1) * 5.0)[pm[..., 0] > 0].median().item()
            clip = (A.abs() > 0.999).float()[M > 0].mean().item()  # labels at the action-range limit
        log = env.extras.get("log", {})
        rec_d = dict(it=it, beta=round(beta, 3), loss_a=round(loss_a.item(), 4), loss_d=round(loss_d.item(), 4), loss_e=round(loss_e.item(), 4),
                     done_prec=round(prec, 3), done_rec=round(rec, 3), aux_offset_mm_p50=round(off_mm, 2), label_clip=round(clip, 4), t=round(time.time() - t0, 1),
                     **{k: round(float(v), 3) for k, v in log.items() if k.startswith(("succ_", "chain_len"))})
        print(json.dumps(rec_d), flush=True)
        log_f.write(json.dumps(rec_d) + "\n")
        log_f.flush()
    if not a.eval_only and ((it + 1) % 200 == 0 or it == start_it + a.iterations - 1):
        torch.save(dict(student_state_dict=net.state_dict(), student_kwargs=kw, optimizer=opt.state_dict(), iteration=it + 1,
                        teacher=a.teacher, student_scale=cfg.student_scale), os.path.join(a.out, f"student_{it + 1}.pt"))
os._exit(0)
