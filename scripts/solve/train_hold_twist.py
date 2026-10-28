"""Passive layer-angle estimation (paper A3): the hands hold the ready grip (zero action), an external drive turns the top
layer along random trajectories (the twist joint's implicit drive on a rate-limited set point: quarter / half / back-and-forth
moves at 30-300 deg/s, dwells, free spells where only the detents act), optional wrench pulses on the cube (cfg.dr_slip).
This is the real-robot validation protocol (a person or the other hand twists while the hand only holds and senses), so
the estimator reads only what the real hand reports: finger joint angles / velocities (obs_real), optionally the pad F/T
and the 60x60 TacMap images (pooled to 30x30), no commands or skill information.

    python scripts/solve/train_hold_twist.py --headless --tier proprio|ft|img --out logs/estimator/hold_ft [--set k=v]
Labels (CubeEstimator units): cube position / rotation off the ready grip, layer residual to the nearest quarter [rad],
velocities; done head = the layer is aligned (|residual| < 5 deg, |speed| < 0.25 rad/s).
"""
import argparse
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--tier", default="ft", choices=["proprio", "ft", "img"])
p.add_argument("--num_envs", type=int, default=512)
p.add_argument("--iterations", type=int, default=400)
p.add_argument("--horizon", type=int, default=32)
p.add_argument("--replay", type=int, default=8)
p.add_argument("--updates", type=int, default=4)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--hidden", type=int, default=384)
p.add_argument("--img_feat", type=int, default=16)
p.add_argument("--episode_s", type=float, default=12.0, help="mean s between re-grips (a fresh ready state)")
p.add_argument("--out", default="logs/estimator/hold_ft")
p.add_argument("--resume", default=None)
p.add_argument("--eval_only", action="store_true")
p.add_argument("--set", action="append", default=[])
p.add_argument("--mod", action="append", default=[])
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app
import sys as _sys, traceback as _tb


def _die(t, v, b):
    _tb.print_exception(t, v, b)
    _sys.stdout.flush()
    os._exit(1)


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
from rubik_lab.solve.env import SIDES, SolveEnv
from rubik_lab.solve.env_cfg import SolveEnvCfg
from rubik_lab.solve.estimator import CubeEstimator, rot6_to_matrix

cfg = SolveEnvCfg()
cfg.scene.num_envs = a.num_envs
cfg.ready_cache = "cache/solve_ready_ud_all4_flick.npz"
cfg.episode_length_s = 1000.0
cfg.skills = ("hold",)
cfg.terminate_on_success = False
cfg.link_contacts = False
cfg.detent_torque, cfg.detent_width = 0.08, 0.2618
cfg.contact_model = "tacmap"
cfg.arm, cfg.fr3_urdf_dir, cfg.fr3_usd_dir = "fr3", "assets_franka/left_m90", "assets_franka/left_m90"
cfg.fr3_layout = "assets_franka/layout_left_m90_c3.json"
cfg.obs_real = True
for kv in a.set:
    k, v = kv.split("=", 1)
    setattr(cfg, k, ast.literal_eval(v))
cfg.tactile.image = a.tier == "img"
env = SolveEnv(cfg)
env.chain_mode = True  # only drops end an episode; re-grips are forced below
dev, n = env.device, env.num_envs
QUARTER = 0.5 * math.pi
env.reset()
zero = torch.zeros(n, env.cfg.action_space, device=dev)


def inputs():
    parts = []
    for h in range(2):
        parts += [env.meas_q(h), 0.1 * env.meas_qd(h)]
        if a.tier != "proprio":
            ft = env.meas_ft(h)
            parts.append(torch.cat([ft[..., :3] * 0.2, ft[..., 3:] * 20.0], -1).flatten(1))
    return torch.nan_to_num(torch.cat(parts, -1)).clamp(-10.0, 10.0)


def tac_img(obs):
    return obs["tac_img"] if a.tier == "img" else None


# ---------------------------------------------------------------- external layer drive
class Drive:
    """per env: set point moving at a random rate to random goals (a quarter, a half, back, a 45 deg overshoot), dwells,
    and free spells (no drive: friction and the detents settle the layer). The layer follows the set point through the
    twist joint's implicit PhysX drive (stable at any gain; an explicit PD held over the 25 ms control step chatters)."""
    KP, KD, TAU_MAX = 2.0, 0.03, 0.3  # [Nm/rad], [Nm s/rad], [Nm]: a firm human twist, beats the 0.08 Nm detents

    def __init__(self):
        self.goal = torch.zeros(n, device=dev)
        self.sp = torch.zeros(n, device=dev)
        self.rate = torch.ones(n, device=dev)
        self.dwell = torch.zeros(n, device=dev)
        self.free = torch.zeros(n, dtype=torch.bool, device=dev)
        self.tid = [env.twist_id]
        self.d0 = env.cube.data.joint_damping[:, env.twist_id].clone()  # the passive layer damping (USD)
        self.on = None  # drive engaged per env, as last written to the sim
        env.cube.write_joint_effort_limit_to_sim(torch.full((n, 1), self.TAU_MAX, device=dev), joint_ids=self.tid)

    def reset(self, ids):
        th = env.cube.data.joint_pos[ids, env.twist_id]
        self.goal[ids], self.sp[ids], self.dwell[ids] = th, th, torch.rand(len(ids), device=dev) * 1.0
        self.free[ids] = False

    def step(self):
        dt = env.step_dt
        th = env.cube.data.joint_pos[:, env.twist_id]
        at = (self.sp - self.goal).abs() < 1e-4
        self.dwell = torch.where(at, self.dwell - dt, self.dwell)
        new = at & (self.dwell <= 0)
        k = int(new.sum())
        if k:
            base = torch.round(th[new] / QUARTER) * QUARTER
            r = torch.rand(k, device=dev)
            step = torch.where(r < 0.6, QUARTER, torch.where(r < 0.8, 2 * QUARTER, torch.where(r < 0.9, 0.5 * QUARTER, 0.0)))
            sgn = torch.where(torch.rand(k, device=dev) < 0.5, -1.0, 1.0)
            self.goal[new] = base + sgn * step
            self.rate[new] = torch.exp(torch.empty(k, device=dev).uniform_(math.log(math.radians(30)), math.log(math.radians(300))))
            self.dwell[new] = torch.empty(k, device=dev).uniform_(0.2, 1.5)
            self.free[new] = torch.rand(k, device=dev) < 0.25
            at = (self.sp - self.goal).abs() < 1e-4
        d = (self.goal - self.sp).clamp(-1.0, 1.0)
        self.sp = self.sp + torch.sign(d) * torch.minimum(d.abs(), self.rate * dt)
        moving = (self.goal - self.sp).abs() > 1e-4
        # a free spell starts once the set point is reached: the layer coasts / settles on its own
        on = ~(self.free & at)
        if self.on is None or bool((on != self.on).any()):
            self.on = on.clone()
            env.cube.write_joint_stiffness_to_sim(torch.where(on, self.KP, 0.0).unsqueeze(-1), joint_ids=self.tid)
            env.cube.write_joint_damping_to_sim(torch.where(on, torch.full_like(self.d0, self.KD), self.d0).unsqueeze(-1),
                                                joint_ids=self.tid)
        env.cube.set_joint_position_target((th + mu_wrap(self.sp - th)).unsqueeze(-1), joint_ids=self.tid)
        env.cube.set_joint_velocity_target(torch.where(moving, torch.sign(d) * self.rate, 0.0).unsqueeze(-1), joint_ids=self.tid)
        env.extra_twist_effort[:] = 0.0


def mu_wrap(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


drive = Drive()
drive.reset(torch.arange(n, device=dev))


def labels():
    """(N, 17) cube state off the ready grip; [9] the layer residual to the nearest quarter. (N,) aligned."""
    y = env.cube_state_true().clone()
    th = env.cube.data.joint_pos[:, env.twist_id]
    res = th - torch.round(th / QUARTER) * QUARTER
    y[:, 9] = res
    om = env.cube.data.joint_vel[:, env.twist_id]
    return y, (res.abs() < math.radians(5.0)) & (om.abs() < 0.25)


kw = dict(obs_dim=inputs().shape[-1], hidden=a.hidden, img_feat=a.img_feat if a.tier == "img" else 0)
net = CubeEstimator(**kw).to(dev)
opt = torch.optim.Adam(net.parameters(), lr=a.lr)
start_it = 0
if a.resume:
    blob = torch.load(a.resume, map_location=dev, weights_only=False)
    net.load_state_dict(blob["state_dict"])
    if not a.eval_only:
        opt.load_state_dict(blob["optimizer"])
    start_it = blob["iteration"]
os.makedirs(a.out, exist_ok=True)
if not a.eval_only:
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
log_f = open(os.path.join(a.out, "eval.jsonl" if a.eval_only else "log.jsonl"), "a")
W = torch.ones(17, device=dev)
W[10:] = 0.3


def errors(z, y):
    zs, ys = net.denormalize(z), net.denormalize(y)
    e_p = (zs[..., 0:3] - ys[..., 0:3]).norm(dim=-1) * 50.0
    Ra, Rb = rot6_to_matrix(zs[..., 3:9].reshape(-1, 6)), rot6_to_matrix(ys[..., 3:9].reshape(-1, 6))
    tr = (Ra.transpose(-1, -2) @ Rb).diagonal(dim1=-2, dim2=-1).sum(-1)
    e_r = torch.rad2deg(torch.acos(((tr - 1.0) / 2.0).clamp(-1.0, 1.0))).view(e_p.shape)
    e_l = torch.rad2deg(mu_wrap(4.0 * (zs[..., 9] - ys[..., 9])).abs() / 4.0)  # residual is defined modulo a quarter
    e_w = (zs[..., 16] - ys[..., 16]).abs() * 10.0  # layer speed [rad/s]
    return e_p, e_r, e_l, e_w


replay_buf = []
h = net.init_hidden(n, dev)
reset_next = torch.ones(n, dtype=torch.bool, device=dev)
regrip_t = torch.empty(n, device=dev).uniform_(0, a.episode_s)
obs = env._get_observations()
t0 = time.time()
for it in range(start_it, start_it + a.iterations):
    O, Y, D, R, I, SL, MV, OM = [], [], [], [], [], [], [], []
    h0 = h.detach()
    with torch.no_grad():
        for t in range(a.horizon):
            x = inputs()
            if not a.eval_only:
                net.norm.update(x)
            img = tac_img(obs)
            y, al = labels()
            O.append(x)
            I.append(img)
            Y.append(net.normalize(y))
            D.append(al.float())
            R.append(reset_next.clone())
            SL.append(env.slip_recent)
            MV.append(env.cube.data.joint_vel[:, env.twist_id].abs() > 0.25)
            OM.append(env.cube.data.joint_vel[:, env.twist_id].abs())
            z, _, h = net(x.unsqueeze(0), h, reset_next.unsqueeze(0), img=None if img is None else img.unsqueeze(0))
            drive.step()
            obs, _, term, _, _ = env.step(zero)
            regrip_t -= env.step_dt
            due = (regrip_t <= 0) & ~term
            if due.any():  # a fresh grip from the ready cache
                ids = due.nonzero().squeeze(-1)
                env._reset_idx(ids)
                regrip_t[ids] = torch.empty(len(ids), device=dev).uniform_(0.5 * a.episode_s, 1.5 * a.episode_s)
            done_ids = (term | due).nonzero().squeeze(-1)
            if len(done_ids):
                drive.reset(done_ids)
            reset_next = term | due
    O, Y, D, R = (torch.stack(v, 0) for v in (O, Y, D, R))
    I = torch.stack(I, 0) if a.tier == "img" else None
    M = torch.ones_like(D)

    def batch_loss(O, Y, D, R, h0, I):
        z, dl, _ = net(O, h0, R, img=I)
        # the residual is periodic: penalize the wrapped difference (in label units) instead of the raw one
        dz = z - Y
        dz9 = mu_wrap(4.0 * dz[..., 9] * net.scale[9]) / 4.0 / net.scale[9]
        dz = torch.cat([dz[..., :9], dz9.unsqueeze(-1), dz[..., 10:]], -1)
        loss_s = (F.smooth_l1_loss(dz, torch.zeros_like(dz), reduction="none") * W).mean() * 17 / W.sum()
        loss_d = F.binary_cross_entropy_with_logits(dl, D)
        return loss_s, loss_d, z, dl

    loss_s, loss_d, z, dl = batch_loss(O, Y, D, R, h0, I)
    if not a.eval_only:
        loss = loss_s + 0.5 * loss_d
        if replay_buf:
            rs, rd, *_ = batch_loss(*replay_buf[int(torch.randint(len(replay_buf), (1,)))])
            loss = loss + rs + 0.5 * rd
        replay_buf.append((O, Y, D, R, h0, I))
        if len(replay_buf) > a.replay:
            replay_buf.pop(0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        for _ in range(a.updates - 1):
            rs, rd, *_ = batch_loss(*replay_buf[int(torch.randint(len(replay_buf), (1,)))])
            opt.zero_grad()
            (rs + 0.5 * rd).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    if it % 10 == 0 or a.eval_only:
        with torch.no_grad():
            e_p, e_r, e_l, e_w = errors(z.detach(), Y)
            warm = torch.ones_like(D, dtype=torch.bool)
            warm[:4] = False  # the first steps after a truncated BPTT window start from a carried hidden state: keep
            q = lambda v, mm, qq: round(float(torch.quantile(v[mm].float()[:200000], qq)), 2) if mm.any() else math.nan  # noqa: E731
            sl, mv = torch.stack(SL, 0), torch.stack(MV, 0)
            pred = (dl > 0).float()
            tp = (pred * D).sum()
            rec = dict(it=it, loss_s=round(loss_s.item(), 4), loss_d=round(loss_d.item(), 4),
                       pos_mm_p50=q(e_p, warm, 0.5), pos_mm_p90=q(e_p, warm, 0.9), rot_deg_p90=q(e_r, warm, 0.9),
                       layer_deg_p50=q(e_l, warm, 0.5), layer_deg_p90=q(e_l, warm, 0.9),
                       layer_deg_p90_moving=q(e_l, warm & mv, 0.9), layer_deg_p90_still=q(e_l, warm & ~mv, 0.9),
                       speed_p90=q(e_w, warm, 0.9), true_speed_p90=q(torch.stack(OM, 0), warm, 0.9),
                       true_speed_p99=q(torch.stack(OM, 0), warm, 0.99), frac_moving=round(mv.float().mean().item(), 3),
                       aligned_prec=round((tp / pred.sum().clamp(min=1)).item(), 3), aligned_rec=round((tp / D.sum().clamp(min=1)).item(), 3),
                       t=round(time.time() - t0, 1))
            if cfg.dr_slip is not None:
                rec.update(layer_deg_p90_slip=q(e_l, warm & sl, 0.9), layer_deg_p90_calm=q(e_l, warm & ~sl, 0.9),
                           pos_mm_p90_slip=q(e_p, warm & sl, 0.9), pos_mm_p90_calm=q(e_p, warm & ~sl, 0.9))
        print(json.dumps(rec), flush=True)
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
    if not a.eval_only and ((it + 1) % 100 == 0 or it == start_it + a.iterations - 1):
        torch.save(dict(state_dict=net.state_dict(), kwargs=kw, optimizer=opt.state_dict(), iteration=it + 1, tier=a.tier),
                   os.path.join(a.out, f"est_{it + 1}.pt"))
os._exit(0)
