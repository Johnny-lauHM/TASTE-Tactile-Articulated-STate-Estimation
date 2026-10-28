"""Where every fingertip sits on the cube for each hand-roll state (left, right in quarter turns), from chain start
states recorded by handoff_matrix.py --dump: fingertip position in cubie units along the F / R / U axes of the env
frame (hand frame), normal force on the U layer vs. the rest of the cube (U bound as the free layer), plus a picture.

    python scripts/solve/grip_geometry.py --headless --pool cache/pools/h6_pool.pt --out outputs/solve/grip
"""
import argparse
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--pool", default="cache/pools/h6_pool.pt")
p.add_argument("--out", default="outputs/solve/grip")
p.add_argument("--settle", type=int, default=4, help="control steps held after the write")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.enable_cameras = True
app = AppLauncher(a).app
import imageio.v2 as imageio
import torch

from rubik_lab.constants import CUBE_SIZE, FINGERTIPS
from rubik_lab.solve import skills as S
from rubik_lab.solve.env import FACE_NORMAL, SolveEnv
from rubik_lab.solve.env_cfg import SolveEnvCfg

pool = torch.load(a.pool)
cfg = SolveEnvCfg()
cfg.scene.num_envs = 1
cfg.ready_cache = "cache/solve_ready_ud_all4_flick.npz"
cfg.episode_length_s = 1e6
cfg.skills = ("hold",)
cfg.link_contacts = False
cfg.viewer.origin_type = "env"
cfg.viewer.env_index = 0
cfg.viewer.resolution = (640, 480)
cfg.viewer.eye = (-0.30, -0.28, 0.42)
cfg.viewer.lookat = (0.0, 0.0, 0.3)
env = SolveEnv(cfg, render_mode="rgb_array")
env.reset()
dev = env.device
ids = torch.zeros(1, dtype=torch.long, device=dev)
axes = torch.tensor([FACE_NORMAL[f] for f in "FRU"], device=dev)  # rows: F, R, U directions
unit = CUBE_SIZE / 3
tips = {s: env.hands[s].find_bodies([f"{s}_{f}_fingertip" for f in FINGERTIPS], preserve_order=True)[0] for s in S.HANDS}
os.makedirs(a.out, exist_ok=True)
for _ in range(5):
    env.render()
roll = pool["roll"].round().long()
for combo in [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0), (1, -1), (-1, 1)]:
    m = ((roll[:, 0] == combo[0]) & (roll[:, 1] == combo[1])).nonzero()
    if len(m) == 0:
        continue
    st = {k: v[m[0]].to(dev) for k, v in pool.items()}
    env._write_full(ids, st)
    env._rebind(ids, torch.tensor([FACE_NORMAL["U"]], device=dev))
    for _ in range(a.settle * cfg.decimation):  # physics only (an env step could end the episode and reset the grip)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
    c = env.cube.data.root_pos_w[0]
    lf = env.tactile.layer_forces()[0]  # (2 hands, 5 pads, [core, top = U layer])
    print(f"== roll left {combo[0]:+d} right {combo[1]:+d}   (F R U in cubies from the center; |c|=1 is a face row)")
    for h, s in enumerate(S.HANDS):
        pos = env.hands[s].data.body_pos_w[0, tips[s]] - c
        for j, f in enumerate(FINGERTIPS):
            fru = (axes @ pos[j]) / unit
            print(f"  {s:5s} {f:6s} F {fru[0]:+5.2f} R {fru[1]:+5.2f} U {fru[2]:+5.2f}   force core {lf[h, j, 0]:5.2f} N  U-layer {lf[h, j, 1]:5.2f} N")
    imageio.imwrite(f"{a.out}/roll_L{combo[0]:+d}_R{combo[1]:+d}.png", env.render()[..., :3])
os._exit(0)
