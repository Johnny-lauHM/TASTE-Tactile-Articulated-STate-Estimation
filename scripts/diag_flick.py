"""Play a flick checkpoint: video of env 0 + per-step trace (layer angle, cube rotation/position, trick finger force,
holding pad forces) + batch statistics.   python scripts/diag_flick.py --task Rubik-Flick-D-Sharpa-v0 --ckpt <model.pt> --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Rubik-Flick-D-Sharpa-v0")
parser.add_argument("--ckpt", required=True)
parser.add_argument("--wrist", type=float, default=-1.0, help="wrist stiffness override (-1 = cfg, 0 = USD)")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--out", default="logs/demo/diag_flick.mp4")
parser.add_argument("--eye", type=float, nargs=3, default=(-0.40, -0.30, 0.46))
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--det", action="store_true", help="legacy alias; deterministic is now the default")
parser.add_argument("--stochastic", action="store_true")
parser.add_argument("--notch", type=float, default=0.0, help="explicit physics setting; original trained policies used 0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab.utils.math as mu  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import rubik_lab  # noqa: E402, F401
from rubik_lab.moves import ALL_MOVES  # noqa: E402
from rubik_lab.cube_contacts import LINKS  # noqa: E402
from rubik_lab.tasks.bimanual_twist.env_cfg import CUBE_CENTER  # noqa: E402

ROOT = "/home/yuhang/workspace/rubik's_cube"
cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
agent = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
cfg.scene.num_envs = args.num_envs
cfg.notch_torque = args.notch
cfg.gravity_curriculum = False
if args.wrist >= 0:
    cfg.wrist_stiffness = None if args.wrist == 0 else args.wrist
cfg.viewer.origin_type = "env"
cfg.viewer.env_index = 0
cfg.viewer.eye = tuple(args.eye)
cfg.viewer.lookat = CUBE_CENTER
cfg.viewer.resolution = (960, 640)
env = gym.make(args.task, cfg=cfg, render_mode="rgb_array")
u = env.unwrapped
u.record = True
wenv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = OnPolicyRunner(wenv, agent.to_dict(), log_dir=None, device=agent.device)
runner.load(os.path.join(ROOT, args.ckpt))
policy = runner.get_inference_policy(device=u.device)
if args.stochastic and not args.det:
    policy = lambda o: runner.alg.policy.act(o).detach()  # noqa: E731
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 20)
except OSError:
    font = ImageFont.load_default()

obs = wenv.get_observations()[0] if isinstance(wenv.get_observations(), tuple) else wenv.get_observations()
T = u.max_episode_length
writer = imageio.get_writer(os.path.join(ROOT, args.out), fps=20)
n = u.num_envs
env.reset()
obs = wenv.get_observations()[0] if isinstance(wenv.get_observations(), tuple) else wenv.get_observations()
mv = [ALL_MOVES[int(m)] for m in u.move_id[:8].cpu()]
print(f"[diag] moves(env0..7) {mv}", flush=True)
peak_rot = torch.zeros(n, device=u.device)
peak_pos = torch.zeros(n, device=u.device)
ep_out = []  # (success, dropped, length, peak_rot, peak_pos, move)
ep_len = torch.zeros(n, device=u.device)
counts = torch.zeros(n, dtype=torch.long, device=u.device)
steps = int(args.episodes * T)
for k in range(steps):
    with torch.no_grad():
        a = policy(obs)
    obs, rew, dones, extras = wenv.step(a)
    snap = u.snapshot  # pre-reset state of this step
    rot = torch.rad2deg(snap["cube_rot_error"])
    pos = snap["cube_pos_error"] * 1000
    peak_rot = torch.maximum(peak_rot, rot)
    peak_pos = torch.maximum(peak_pos, pos)
    ended = dones.bool()
    if ended.any():
        for i in torch.nonzero(ended & (counts < args.episodes)).flatten().tolist():
            counts[i] += 1
            ep_out.append((bool(snap["primitive_done"][i]), bool(snap["dropped"][i]), float(ep_len[i] + 1) * u.step_dt, float(peak_rot[i]), float(peak_pos[i]), ALL_MOVES[int(snap["move_id"][i])]))
        peak_rot[ended] = 0.0
        peak_pos[ended] = 0.0
        ep_len[ended] = -1.0
    ep_len += 1
    lf = snap["tactile_layer_force"]
    err = torch.rad2deg(snap["twist_error"])
    if k % 5 == 0 and k < 3 * T:
        h = int(u.flick_hand[snap["move_id"][0]])
        f_tr = lf[0, h, :, 1].cpu().numpy()
        print(
            f"[diag] t={k * u.step_dt:4.2f}s err {err[0]:6.1f} deg | cube rot {rot[0]:5.1f} deg pos {pos[0]:4.1f} mm | "
            f"trick-hand pads on layer {np.array2string(f_tr, precision=1)} N | core pads L {np.array2string(lf[0, 0, :, 0].cpu().numpy(), precision=1)} "
            f"R {np.array2string(lf[0, 1, :, 0].cpu().numpy(), precision=1)} | a {np.array2string(a[0].cpu().numpy(), precision=2)}",
            flush=True,
        )
    if k % 5 == 0 and k < 3 * T:
        cf = u.cube_contacts.forces()[0]  # (2 bodies [core, top], 2 hands, 27 links)
        on_top = [f"{'LR'[hh]}:{LINKS[li]}={cf[1, hh, li]:.1f}" for hh in range(2) for li in range(len(LINKS)) if cf[1, hh, li] > 0.3]
        print(f"[diag]      links on moving layer: {' '.join(on_top)}", flush=True)
    if k % 2 == 0 and k < 3 * T:
        img = Image.fromarray(env.render()[..., :3])
        ImageDraw.Draw(img).text((16, 12), f"{ALL_MOVES[int(snap['move_id'][0])]}  err {err[0]:.0f} deg  cube rot {rot[0]:.0f} deg", fill=(255, 255, 255), font=font)
        writer.append_data(np.asarray(img))
    if ended[0]:
        print(f"[diag] env0 episode ended at {int(snap['episode_step'][0]) * u.step_dt:.2f}s: {'SUCCESS' if snap['primitive_done'][0] else ('DROPPED' if snap['dropped'][0] else 'timeout')}", flush=True)
    if (counts >= args.episodes).all():
        break
if not (counts >= args.episodes).all():
    raise RuntimeError("Incomplete per-environment episode quota")
succ = np.array([e[0] for e in ep_out]); drop = np.array([e[1] for e in ep_out]); L = np.array([e[2] for e in ep_out])
pr = np.array([e[3] for e in ep_out]); pp = np.array([e[4] for e in ep_out]); mvs = np.array([e[5] for e in ep_out])
print(f"[diag] {len(ep_out)} episodes: success {succ.mean():.0%}, dropped {drop.mean():.0%}, timeout {(~succ & ~drop).mean():.0%}", flush=True)
for m in sorted(set(mvs)):
    s = mvs == m
    print(f"[diag]   {m:3s}: n {s.sum():4d} success {succ[s].mean():.0%}", flush=True)
if succ.any():
    print(f"[diag] successful episodes: time median {np.median(L[succ]):.2f}s | peak cube rot median {np.median(pr[succ]):.1f} deg (90% {np.percentile(pr[succ], 90):.1f}) | peak pos median {np.median(pp[succ]):.1f} mm (90% {np.percentile(pp[succ], 90):.1f})", flush=True)
if (~succ).any():
    print(f"[diag] failed episodes: peak cube rot median {np.median(pr[~succ]):.1f} deg | peak pos median {np.median(pp[~succ]):.1f} mm", flush=True)
writer.close()
print(f"[diag] video -> {args.out}", flush=True)
env.close()
app.close()
