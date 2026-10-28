"""Render the canonical side grip from a few viewpoints (sanity check of the ready state)."""
import argparse, os
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--cache", default="cache/g70s_cycle_demo1_v3.npz")
p.add_argument("--out", default="outputs/solve/look")
p.add_argument("--steps", type=int, default=20)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.enable_cameras = True
app = AppLauncher(a).app
import gymnasium as gym, torch, numpy as np
from PIL import Image
import rubik_lab  # noqa
from rubik_lab.tasks.bimanual_twist.cycle_cfg import CycleUEnvCfg
from rubik_lab.tasks.bimanual_twist.env_cfg import CUBE_CENTER
cfg = CycleUEnvCfg(); cfg.scene.num_envs = 1; cfg.grasp_cache_path = a.cache
cfg.viewer.origin_type = "env"; cfg.viewer.env_index = 0; cfg.viewer.lookat = CUBE_CENTER; cfg.viewer.resolution = (640, 480)
views = {"front": (-0.40, 0.0, 0.36), "front_right": (-0.30, -0.28, 0.42), "top": (-0.05, 0.0, 0.70), "back": (0.40, 0.05, 0.40), "right": (0.0, -0.45, 0.32)}
env = gym.make("Rubik-Cycle-U-Sharpa-v0", cfg=cfg, render_mode="rgb_array"); u = env.unwrapped
env.reset()
act = torch.zeros(1, u.cfg.action_space, device=u.device)
for _ in range(a.steps): env.step(act)
os.makedirs(a.out, exist_ok=True)
vc = u.viewport_camera_controller
for name, eye in views.items():
    vc.update_view_location(eye=eye, lookat=CUBE_CENTER)
    for _ in range(3): u.sim.render()
    img = env.render(); Image.fromarray(img).save(f"{a.out}/{name}.png")
print("cube", u.cube.data.root_pos_w[0] - u.scene.env_origins[0], u.cube.data.joint_pos[0])
app.close()
