"""Check the Wave SE tactile emulation: 60x60 TacMap images + 6-D F/T per pad, 30 fps sample-and-hold, 20 ms latency.

Starts from the grasp cache, holds, then squeezes all fingers a bit more. Saves
    logs/tactile/pads_<t>.png      10 delivered images (rows: left/right hand, cols: thumb..pinky) with |F|, Fz
    logs/tactile/timing.png        raw vs delivered |F| of one pad at physics rate (sample & hold + latency)
and prints the step time with images on.
    python scripts/viz_tactile.py --headless
"""

import argparse
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--squeeze_deg", type=float, default=15.0, help="extra flexion added to every finger joint")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import rubik_lab  # noqa: E402, F401
from rubik_lab.assets import FINGERTIPS  # noqa: E402
from rubik_lab.tasks.bimanual_twist.env_cfg import BimanualTwistEnvCfg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "logs/tactile")
os.makedirs(OUT, exist_ok=True)
SIDES = ("left", "right")

cfg = BimanualTwistEnvCfg()
cfg.gravity_curriculum = False  # real gravity
cfg.scene.num_envs = args.num_envs
cfg.ee_action_mode = "absolute"
cfg.episode_length_s = 1e6
cfg.ee_init_pos_noise = 0.0
cfg.cube_drop_height = -10.0
cfg.tactile.image = True
env = gym.make("Rubik-Twist-Bimanual-Sharpa-v0", cfg=cfg)
u = env.unwrapped
n, dev = u.num_envs, u.device
env.reset()
tac = u.tactile
print(f"[viz] physics dt={u.physics_dt * 1000:.2f} ms, capture every {tac.capture_every} steps "
      f"({1 / (tac.capture_every * u.physics_dt):.1f} fps), latency {tac.latency_steps} steps "
      f"({tac.latency_steps * u.physics_dt * 1000:.1f} ms)", flush=True)  # fmt: skip
q0 = {s: u.finger_tgt[s].clone() for s in SIDES}


def action(extra):
    acts = []
    for s in SIDES:
        lo, hi = u.finger_lower[s], u.finger_upper[s]
        f = torch.max(torch.min(q0[s] + extra, hi), lo)
        acts.append(torch.cat([u.ee_pos_tgt[s], u.ee_quat_tgt[s], 2 * (f - lo) / (hi - lo) - 1], -1))
    return torch.cat(acts, -1)


def save_pads(tag):
    img = tac.image[0].cpu().numpy()  # (2, 5, 60, 60)
    ft = tac.ft[0].cpu().numpy()  # (2, 5, 6)
    fig, ax = plt.subplots(2, 5, figsize=(13, 5.6))
    for h in range(2):
        for f in range(5):
            a = ax[h, f]
            a.imshow(img[h, f], cmap="magma", vmin=0, vmax=255)
            a.set_title(f"{SIDES[h]} {FINGERTIPS[f]}\n|F|={np.linalg.norm(ft[h, f, :3]):.2f}N Fz={ft[h, f, 2]:+.2f}", fontsize=9)
            a.axis("off")
    fig.suptitle(f"Wave SE tactile (delivered, env 0) - {tag}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"pads_{tag}.png"), dpi=90)
    plt.close(fig)
    nz = (tac.image > 0).flatten(3).sum(-1).float()  # (n, 2, 5) active taxels per pad
    print(f"  [{tag}] active taxels per pad (env-mean) L={nz[:, 0].mean(0).int().tolist()} R={nz[:, 1].mean(0).int().tolist()} "
          f"| |F| per pad env0 L={np.linalg.norm(ft[0, :, :3], axis=-1).round(2).tolist()} "
          f"R={np.linalg.norm(ft[1, :, :3], axis=-1).round(2).tolist()}", flush=True)  # fmt: skip


# 1) hold the cached grasp
for _ in range(20):
    env.step(action(0.0))
save_pads("hold")

# 2) squeeze, recording raw vs delivered force of the most loaded pad at physics rate
h, f = divmod(int(torch.linalg.norm(tac.ft[0, :, :, :3], dim=-1).argmax()), 5)
raw_log, del_log = [], []
orig_step = tac.step


def logging_step():
    orig_step()
    raw_log.append(torch.linalg.norm(tac.raw_ft()[0, h, f, :3]).item())
    del_log.append(torch.linalg.norm(tac.ft[0, h, f, :3]).item())


tac.step = logging_step
torch.cuda.synchronize()
t0 = time.time()
steps = 40
for k in range(steps):
    s = min(1.0, (k + 1) / 20)
    env.step(action(np.radians(args.squeeze_deg) * s))
torch.cuda.synchronize()
dt = (time.time() - t0) / steps
tac.step = orig_step
save_pads("squeeze")
print(f"[viz] env.step with images on: {dt * 1000:.1f} ms for {n} envs "
      f"({n * cfg.decimation / dt:.0f} physics-steps*envs/s)", flush=True)  # fmt: skip

fig, ax = plt.subplots(figsize=(10, 3.5))
t = np.arange(len(raw_log)) * u.physics_dt * 1000
ax.plot(t, raw_log, lw=1, label="raw (physics rate)")
ax.step(t, del_log, where="post", lw=1.5, label="delivered (30 fps, 20 ms latency, noise, 0.05 N quantization)")
ax.set_xlabel("time [ms]")
ax.set_ylabel(f"|F| {SIDES[h]} {FINGERTIPS[f]} [N]")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "timing.png"), dpi=90)
print(f"[viz] saved to {OUT}", flush=True)
env.close()
app.close()
