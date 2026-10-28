"""Blind cube-state estimator, trained in the loop (DAgger): the sighted chain teacher system runs random skill chains
while every controller acts on the ESTIMATED cube state (env cfg.cube_est); with probability beta per env-step the
simulator's state is passed instead (decays to 0). Labels: the simulator's cube state (SolveEnv.cube_state_true) and the
completion window (stable_steps > 0). Inputs: sensing only (SolveEnv._est_obs: finger joints, pad F/T, commanded skill).

With beta = 0 the env's chain statistics (succ_*, chain_len) are those of the blind system (truth only advances skills).

    python scripts/solve/train_estimator.py --headless --teacher <chain ckpt> --num_envs 2048 --iterations 800 --out logs/estimator/est1
"""
import argparse
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--teacher", required=True)
p.add_argument("--num_envs", type=int, default=2048)
p.add_argument("--iterations", type=int, default=800)
p.add_argument("--horizon", type=int, default=32)
p.add_argument("--beta_hold", type=int, default=30, help="iterations of simulator-state control before the decay")
p.add_argument("--beta_iters", type=int, default=300, help="iterations over which simulator-state control decays to 0")
p.add_argument("--replay", type=int, default=4, help="keep this many recent batches; every update also trains on one of them")
p.add_argument("--updates", type=int, default=1, help="gradient steps per iteration (the first on the new batch, then on replayed ones)")
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--hidden", type=int, default=384)
p.add_argument("--done_weight", type=float, default=0.5)
p.add_argument("--done_label", default="window", choices=["window", "success"],
               help="window: the completion criteria hold now (stable_steps > 0) | success: the skill has succeeded "
               "(criteria held cfg.stable_s; use with --set chain_linger_s=... so that the label lasts)")
p.add_argument("--obs_key", default="est", help="est: sensing only | student: + the system's own commands (finger targets, wrist "
               "tracking error, previous action); in the loop at beta 0 these carry the estimate, not the truth")
p.add_argument("--out", default="logs/estimator/est1")
p.add_argument("--resume", default=None)
p.add_argument("--init_from", default=None, help="weights of an estimator over a prefix of this input (new input columns start at zero weight)")
p.add_argument("--eval_only", action="store_true", help="with --resume: beta 0, no updates, nothing saved")
p.add_argument("--truth_dims", default="", help="diagnostic: these state columns (e.g. 0:3 position, 3:9 rotation, 9:10 layer, "
               "10:17 velocities) are passed from the simulator instead of the estimate")
p.add_argument("--img_feat", type=int, default=0, help="> 0: also read the 60x60 TacMap pad images (enables cfg.tactile.image; "
               "use with --set contact_model='tacmap': rigid pads leave the images empty), this many CNN features per pad")
p.add_argument("--zero_ft", action="store_true", help="input ablation: the pad F/T columns of the student observation are zeroed "
               "(proprioception only, with --img_feat 0); saved in the checkpoint, the executor zeroes them too")
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
import math
import time

import torch
import torch.nn.functional as F

from rubik_lab.solve import skills as S

for kv in a.mod:
    k, v = kv.split("=", 1)
    setattr(S, k, ast.literal_eval(v))
from rubik_lab.solve.env import SolveEnv
from rubik_lab.solve.env_cfg import SolveEnvCfg
from rubik_lab.solve.estimator import CubeEstimator, rot6_to_matrix
from rubik_lab.solve.policies import load_policies

cfg = SolveEnvCfg()
cfg.scene.num_envs = a.num_envs
cfg.ready_cache = "cache/solve_ready_ud_all4_flick.npz"
cfg.episode_length_s = 20.0
cfg.chain_train = True
cfg.obs_next_skill = True
cfg.cube_est = True
cfg.terminate_on_success = False
cfg.skills = ("hold",)
cfg.link_contacts = False
cfg.detent_torque, cfg.detent_width = 0.08, 0.2618
cfg.flick_legacy_ckpt = "logs/rsl_rl/rubik_flick/2026-10-17_16-28-11/model_1900.pt"
cfg.flick_legacy_kw = dict(latch_steps=2, latch_min_s=0.1)  # blind: a single wrong layer estimate does not end the flick
for kv in a.set:
    k, v = kv.split("=", 1)
    setattr(cfg, k, ast.literal_eval(v))
cfg.tactile.image = a.img_feat > 0
env = SolveEnv(cfg)
dev, n = env.device, env.num_envs
obs, _ = env.reset()
env.est_cube = env.cube_state_true()
obs = env._get_observations()
teacher = load_policies(env, {"all": a.teacher})["all"]
kw = dict(obs_dim=obs[a.obs_key].shape[-1], hidden=a.hidden, img_feat=a.img_feat)
zero_cols = []
if a.zero_ft:  # student / student_ee: per hand [q 22, qd 22, finger targets 22, wrist error 6, F/T 30] first
    assert a.obs_key in ("student", "student_ee")
    blk = env._proprio(0, "left").shape[-1]
    assert blk == 102, blk
    zero_cols = list(range(72, 102)) + list(range(blk + 72, blk + 102))
ZC = torch.tensor(zero_cols, dtype=torch.long, device=dev)


def get_eo(obs):
    eo = obs[a.obs_key]
    if len(zero_cols):
        eo = eo.clone()
        eo[:, ZC] = 0.0
    return eo


n_in = kw["obs_dim"] + 10 * a.img_feat
net = CubeEstimator(**kw).to(dev)
opt = torch.optim.Adam(net.parameters(), lr=a.lr)
start_it = 0
if a.init_from:
    sd = torch.load(a.init_from, map_location=dev, weights_only=False)["state_dict"]
    d_old = sd["enc.0.weight"].shape[1]
    d_vec = sd["norm.mean"].shape[0]  # vector inputs of the old net (image features, if any, follow them)
    w = torch.zeros(sd["enc.0.weight"].shape[0], n_in, device=dev)
    w[:, :d_vec] = sd["enc.0.weight"][:, :d_vec]
    if d_old > d_vec and a.img_feat:
        w[:, kw["obs_dim"] : kw["obs_dim"] + d_old - d_vec] = sd["enc.0.weight"][:, d_vec:]
    sd["enc.0.weight"] = w
    for k_, fill in (("norm.mean", 0.0), ("norm.var", 1.0)):
        v_ = torch.full((kw["obs_dim"],), fill, device=dev)
        v_[:d_vec] = sd[k_]
        sd[k_] = v_
    missing, _ = net.load_state_dict(sd, strict=False)
    assert all(k_.startswith("img_enc.") for k_ in missing), missing
    print(f"[est] init from {a.init_from}: {d_old} -> {n_in} inputs, new: {sorted(set(k_.split('.')[0] for k_ in missing))}", flush=True)
if a.resume:
    blob = torch.load(a.resume, map_location=dev, weights_only=False)
    net.load_state_dict(blob["state_dict"])
    if not a.eval_only:
        opt.load_state_dict(blob["optimizer"])
    start_it = blob["iteration"]
os.makedirs(a.out, exist_ok=True)
if not a.eval_only:
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
log_f = open(os.path.join(a.out, "log.jsonl"), "a")
X_SKILLS = torch.tensor([S.SKILL_INDEX["x"], S.SKILL_INDEX["x'"]], device=dev)
W = torch.ones(17, device=dev)
W[10:] = 0.3  # velocities: secondary


def errors(z, y):
    """position [mm], rotation [deg], layer [deg] errors of the normalized estimate z against the normalized label y."""
    zs, ys = net.denormalize(z), net.denormalize(y)
    e_p = (zs[..., 0:3] - ys[..., 0:3]).norm(dim=-1) * 50.0
    Ra, Rb = rot6_to_matrix(zs[..., 3:9].reshape(-1, 6)), rot6_to_matrix(ys[..., 3:9].reshape(-1, 6))
    tr = (Ra.transpose(-1, -2) @ Rb).diagonal(dim1=-2, dim2=-1).sum(-1)
    e_r = torch.rad2deg(torch.acos(((tr - 1.0) / 2.0).clamp(-1.0, 1.0))).view(e_p.shape)
    e_l = torch.rad2deg((zs[..., 9] - ys[..., 9]).abs())
    return e_p, e_r, e_l


diag = {}  # eval_only: layer / position errors per (skill, time-in-skill bucket of 0.1 s)
replay_buf = []
h = net.init_hidden(n, dev)
reset_next = torch.ones(n, dtype=torch.bool, device=dev)
t0 = time.time()
for it in range(start_it, start_it + a.iterations):
    beta = 0.0 if a.eval_only else min(1.0, max(0.0, 1.0 - (it - a.beta_hold) / max(1, a.beta_iters)))
    O, Y, D, R, M, I, SL = [], [], [], [], [], [], []
    h0 = h.detach()
    with torch.no_grad():
        for t in range(a.horizon):
            eo = get_eo(obs)
            if not a.eval_only:
                net.norm.update(eo)
            img = obs["tac_img"].unsqueeze(0) if a.img_feat else None
            z, _, h = net(eo.unsqueeze(0), h, reset_next.unsqueeze(0), img=img)
            truth = env.cube_state_true()
            mask = (torch.rand(n, device=dev) < beta).unsqueeze(-1)
            env.est_cube = torch.where(mask, truth, net.denormalize(z[0]))
            for sl in filter(None, a.truth_dims.split(",")):
                i0, i1 = map(int, sl.split(":"))
                env.est_cube[:, i0:i1] = truth[:, i0:i1]
            obs = env._get_observations()  # the teacher's cube observation from the state just set
            if a.eval_only:
                ep, er, el = errors(z[0], net.normalize(truth))
                b = (env.t_skill / 0.1).long().clamp(max=30)
                key = env.skill * 31 + b
                for name, v in (("lay", el), ("pos", ep)):
                    acc = diag.setdefault(name, [torch.zeros(len(S.SKILLS) * 31, device=dev) for _ in range(3)])
                    acc[0].index_add_(0, key, torch.ones_like(v))
                    acc[1].index_add_(0, key, v)
                    acc[2].index_add_(0, key, (v > (10.0 if name == "lay" else 3.0)).float())
            O.append(eo)
            if a.img_feat:
                I.append(img[0])
            Y.append(net.normalize(truth))
            D.append((env.success if a.done_label == "success" else env.stable_steps > 0).float())
            M.append((~env.chain_warm).float())
            SL.append(env.slip_recent)
            R.append(reset_next.clone())
            obs, _, term, trunc, _ = env.step(teacher(obs["policy"]))
            reset_next = term.clone()  # chains restart only on failure (time-outs continue)
    O, Y, D, R, M = (torch.stack(x, 0) for x in (O, Y, D, R, M))
    I = torch.stack(I, 0) if I else None

    def batch_loss(O, Y, D, R, M, h0, I):
        z, dl, _ = net(O, h0, R, img=I)
        w = M.unsqueeze(-1) * W
        loss_s = (F.smooth_l1_loss(z, Y, reduction="none") * w).sum() / (M.sum() * W.sum() + 1)
        loss_d = (F.binary_cross_entropy_with_logits(dl, D, reduction="none") * M).sum() / (M.sum() + 1)
        return loss_s, loss_d, z, dl

    loss_s, loss_d, z, dl = batch_loss(O, Y, D, R, M, h0, I)
    loss = loss_s + a.done_weight * loss_d
    if a.replay > 0 and not a.eval_only:
        if replay_buf:
            rs, rd, *_ = batch_loss(*replay_buf[int(torch.randint(len(replay_buf), (1,)))])
            loss = loss + rs + a.done_weight * rd
        replay_buf.append((O, Y, D, R, M, h0, I))
        if len(replay_buf) > a.replay:
            replay_buf.pop(0)
    if not a.eval_only:
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        for _ in range(a.updates - 1):  # more steps on replayed batches (collection is the expensive part)
            rs, rd, *_ = batch_loss(*replay_buf[int(torch.randint(len(replay_buf), (1,)))])
            opt.zero_grad()
            (rs + a.done_weight * rd).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    if it % 10 == 0:
        with torch.no_grad():
            m = M > 0
            e_p, e_r, e_l = errors(z.detach(), Y)
            q = lambda x, mm, qq: round(float(torch.quantile(x[mm].float()[:200000], qq)), 2) if mm.any() else math.nan  # noqa: E731
            pred = (dl > 0).float()
            seg = {}
            if cfg.dr_slip is not None:  # errors within 0.5 s of a wrench pulse vs the rest
                sl = torch.stack(SL, 0)
                for nm, mm in (("slip", m & sl), ("calm", m & ~sl)):
                    seg.update({f"pos_mm_p90_{nm}": q(e_p, mm, 0.9), f"layer_deg_p90_{nm}": q(e_l, mm, 0.9),
                                f"rot_deg_p90_{nm}": q(e_r, mm, 0.9), f"layer_deg_p50_{nm}": q(e_l, mm, 0.5), f"frac_{nm}": round(float(mm.float().mean()), 3)})
            tp = (pred * D * M).sum()
            prec = (tp / (pred * M).sum().clamp(min=1)).item()
            rec = (tp / (D * M).sum().clamp(min=1)).item()
        log = env.extras.get("log", {})
        rec_d = dict(it=it, beta=round(beta, 3), loss_s=round(loss_s.item(), 4), loss_d=round(loss_d.item(), 4),
                     pos_mm_p50=q(e_p, m, 0.5), pos_mm_p90=q(e_p, m, 0.9), rot_deg_p50=q(e_r, m, 0.5), rot_deg_p90=q(e_r, m, 0.9),
                     layer_deg_p50=q(e_l, m, 0.5), layer_deg_p90=q(e_l, m, 0.9), done_prec=round(prec, 3), done_rec=round(rec, 3),
                     **seg, t=round(time.time() - t0, 1), **{k: round(float(v), 3) for k, v in log.items() if k.startswith(("succ_", "chain_len"))})
        print(json.dumps(rec_d), flush=True)
        log_f.write(json.dumps(rec_d) + "\n")
        log_f.flush()
    if not a.eval_only and ((it + 1) % 100 == 0 or it == start_it + a.iterations - 1):
        torch.save(dict(state_dict=net.state_dict(), kwargs=kw, optimizer=opt.state_dict(), iteration=it + 1, teacher=a.teacher, obs_key=a.obs_key, zero_cols=zero_cols,
                        done_label=a.done_label),
                   os.path.join(a.out, f"est_{it + 1}.pt"))
if a.eval_only and diag:
    out = {}
    for name, (cnt, tot, big) in diag.items():
        for k, sk in enumerate(S.SKILLS):
            row = [(round(float(tot[k * 31 + b] / cnt[k * 31 + b]), 2), round(float(big[k * 31 + b] / cnt[k * 31 + b]), 3))
                   for b in range(31) if cnt[k * 31 + b] > 50]
            if row:
                out[f"{name}:{sk}"] = row
                print(f"[diag] {name} {sk:14s} mean / frac>{'10deg' if name == 'lay' else '3mm'} per 0.1 s: " + " ".join(f"{m}/{f}" for m, f in row), flush=True)
    json.dump(out, open(os.path.join(a.out, "diag.json"), "w"))
os._exit(0)
