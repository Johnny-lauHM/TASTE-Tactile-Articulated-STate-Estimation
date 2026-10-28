"""Play a trained checkpoint move by move and record a demo video (env 0) with the Wave SE tactile panel.

Each move runs on all envs (real gravity) until env 0 ends or the episode times out; outcome stats over all envs are
printed per move. Video: left = viewport, right = 10 delivered 60x60 tactile images + layer angle / status.
    python scripts/play_demo.py --headless --enable_cameras --checkpoint logs/rsl_rl/.../model_250.pt --moves "U,U',R,R'"
"""

import argparse
import glob
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=None, help="model_*.pt (default: newest in logs/rsl_rl)")
parser.add_argument("--moves", default="U,U',R,R'")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--max_steps", type=int, default=400)
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import rubik_lab  # noqa: E402, F401
from rubik_lab.assets import FINGERTIPS  # noqa: E402
from rubik_lab.moves import MOVE_INDEX  # noqa: E402
from rubik_lab.tasks.bimanual_twist.agents.rsl_rl_ppo_cfg import BimanualTwistPPORunnerCfg  # noqa: E402
from rubik_lab.tasks.bimanual_twist.env_cfg import CUBE_CENTER, BimanualTwistEnvCfg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ckpt = args.checkpoint or max(glob.glob(os.path.join(ROOT, "logs/rsl_rl/*/*/model_*.pt")), key=os.path.getmtime)
moves = args.moves.split(",")
out = args.out or os.path.join(ROOT, f"logs/demo/{os.path.basename(os.path.dirname(ckpt))}_{os.path.basename(ckpt)[:-3]}.mp4")
os.makedirs(os.path.dirname(out), exist_ok=True)

cfg = BimanualTwistEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.gravity_curriculum = False  # real gravity
cfg.episode_length_s = args.max_steps * cfg.decimation * cfg.sim.dt
cfg.tactile.image = True
cfg.viewer.origin_type = "env"
cfg.viewer.env_index = 0
cfg.viewer.eye = (-0.32, -0.26, 0.44)  # front-right: thumbs and the R face visible
cfg.viewer.lookat = CUBE_CENTER
cfg.viewer.resolution = (960, 720)
env = gym.make("Rubik-Twist-Bimanual-Sharpa-v0", cfg=cfg, render_mode="rgb_array")
u = env.unwrapped
u.record = True
venv = RslRlVecEnvWrapper(env, clip_actions=BimanualTwistPPORunnerCfg().clip_actions)
runner = OnPolicyRunner(venv, BimanualTwistPPORunnerCfg().to_dict(), log_dir=None, device=u.device)
runner.load(ckpt)
policy = runner.get_inference_policy(device=u.device)
print(f"[demo] checkpoint {ckpt}", flush=True)

fig, axes = plt.subplots(2, 5, figsize=(6.4, 3.6), dpi=100)
writer = imageio.get_writer(out, fps=20)


def panel(move, step, status):
    img = u.tactile.image[0].cpu().numpy()
    ft = u.tactile.ft[0].cpu().numpy()
    for h in range(2):
        for f in range(5):
            a = axes[h, f]
            a.clear()
            a.imshow(img[h, f], cmap="magma", vmin=0, vmax=255)
            a.set_title(f"{'LR'[h]}-{FINGERTIPS[f]}\n{np.linalg.norm(ft[h, f, :3]):.1f}N", fontsize=7)
            a.axis("off")
    twist = np.degrees(u._cube_state()[2][0].item())
    fig.suptitle(f"move {move}  t={step / 40:4.1f}s  layer={twist:+6.1f} deg  target={np.degrees(u.move_target[u.move_id[0]].item()):+.0f}\n{status}", fontsize=8)  # fmt: skip
    fig.tight_layout()
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3]


torch.inference_mode().__enter__()  # env buffers get rebound inside step(); keep resets in the same mode
for move in moves:
    u.move_pool = torch.tensor([MOVE_INDEX[move]], device=u.device)
    venv.reset()
    obs = venv.get_observations()
    latch = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
    ended = torch.zeros_like(latch)
    prim = torch.zeros_like(latch)
    drop = torch.zeros_like(latch)
    status = ""
    for k in range(args.max_steps):
        act = policy(obs)
        obs, _, dones, extras = venv.step(act)
        snap, live = u.snapshot, ~ended  # pre-reset flags (the auto-reset clears the live buffers)
        latch |= live & snap["layer_done_latch"]
        prim |= live & snap["primitive_done"]
        drop |= live & snap["dropped"]
        ended |= dones.bool()
        status = "PRIMITIVE DONE" if prim[0] else ("layer reached target" if latch[0] else ("DROPPED" if drop[0] else "in progress"))
        if k % 2 == 0:
            frame = u.render()
            tac = panel(move, k, status)
            h = frame.shape[0]
            tac = np.asarray(Image.fromarray(tac).resize((int(tac.shape[1] * h / tac.shape[0]), h)))
            writer.append_data(np.concatenate([frame[..., :3], tac], 1))
        if ended[0]:
            for _ in range(10):  # hold the last frame ~0.5 s
                writer.append_data(np.concatenate([frame[..., :3], tac], 1))
            break
    print(f"[demo] {move:3s}: over {u.num_envs} envs  layer reached target {latch.float().mean():.0%}  "
          f"primitive done {prim.float().mean():.0%}  dropped {drop.float().mean():.0%}   (env0: {status}, {k + 1} steps)", flush=True)  # fmt: skip
writer.close()
print(f"[demo] video -> {out}", flush=True)
env.close()
app.close()
