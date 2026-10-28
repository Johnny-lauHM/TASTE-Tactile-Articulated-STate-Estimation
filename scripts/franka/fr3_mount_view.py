"""Close-ups of one side's FR3 flange -> printed adapter -> Sharpa hand, the robot alone (no cube, no other arm), the arm
in a neutral pose with the flange pointing forward, fingers open; cameras placed in the FR3 flange frame (link8).

    python scripts/franka/fr3_mount_view.py --headless --enable_cameras [--usd ".../{side}/robot.usd"] [--tag mount_]
"""
import argparse
import json
import os

from isaaclab.app import AppLauncher

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
FR3 = os.path.expanduser("~/workspace/fr3-sharpa-rubik")
p = argparse.ArgumentParser()
p.add_argument("--usd", default=os.path.join(FR3, "assets/{side}/robot.usd"))
p.add_argument("--out", default=os.path.join(ROOT, "outputs/franka"))
p.add_argument("--tag", default="mount_")
p.add_argument("--grip", action="store_true", help="fingers at the ready grip (cache) instead of open")
p.add_argument("--sides", default="left,right")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as mu  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402

# flange pointing forward (+x), elbow up: a readable pose for both sides
NEUTRAL = {"fr3_joint1": 0.0, "fr3_joint2": 0.0, "fr3_joint3": 0.0, "fr3_joint4": -1.5708, "fr3_joint5": 0.0,
           "fr3_joint6": 1.5708, "fr3_joint7": 0.0}
SIDES = {s: y for s, y in {"left": 2.0, "right": -2.0}.items() if s in a.sides.split(",")}  # far apart: one per picture


def fingers(s):
    if not a.grip:
        return {f"{s}_.*": 0.0}
    from rubik_lab.constants import FINGER_JOINTS
    d = np.load(os.path.join(ROOT, "cache/solve_ready_ud_all4_flick.npz"))
    return {f"{s}_{n}": float(v) for n, v in zip(FINGER_JOINTS, d[f"{s}_fingers_act"][0])}


def main():
    ctx = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 240, device=a.device, gravity=(0.0, 0.0, 0.0)))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2500.0))
    robots = {}
    for s, y in SIDES.items():
        robots[s] = Articulation(ArticulationCfg(
            prim_path=f"/World/{s}",
            spawn=sim_utils.UsdFileCfg(usd_path=a.usd.format(side=s), rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, y, 0.0), joint_pos={**NEUTRAL, **fingers(s)}),
            actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=1e4, damping=100.0)}))
    cam = Camera(CameraCfg(prim_path="/World/cam", height=720, width=960, data_types=["rgb"],
                           spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.02, 20.0))))
    ctx.reset()
    for _ in range(4):
        ctx.step()
    os.makedirs(a.out, exist_ok=True)
    report = {}
    for s, r in robots.items():
        d = r.data
        i8, ia, ih = (r.find_bodies(b)[0][0] for b in ("fr3_link8", f"{s}_adapter", f"{s}_hand_flange"))
        p8, q8 = d.body_pos_w[0, i8], d.body_quat_w[0, i8]
        ph, qh = d.body_pos_w[0, ih], d.body_quat_w[0, ih]
        c = (p8 + d.body_pos_w[0, ia] + ph) / 3.0
        R8 = mu.matrix_from_quat(q8)
        z8, x8, y8 = R8[:, 2], R8[:, 0], R8[:, 1]  # flange axis and the two flange-plane axes
        Rh = mu.matrix_from_quat(qh)
        # hand axes in the flange frame: hand +z = finger direction, +x = palm normal
        rel = (R8.T @ Rh).cpu().numpy()
        report[s] = {"hand_fingers_dir_in_flange_frame": np.round(rel[:, 2], 3).tolist(),
                     "hand_palm_normal_in_flange_frame": np.round(rel[:, 0], 3).tolist(),
                     "hand_flange_offset_in_flange_frame_mm": np.round((R8.T @ (ph - p8)).cpu().numpy() * 1000, 1).tolist()}
        views = {"side_x": x8 * 0.38, "side_y": y8 * 0.38, "along_axis": z8 * 0.38 + x8 * 0.05, "oblique": (x8 + y8 + 0.6 * z8) * 0.25}
        for vn, off in views.items():
            cam.set_world_poses_from_view((c + off).unsqueeze(0), c.unsqueeze(0))
            for _ in range(3):
                ctx.render()
            cam.update(dt=0.0)
            fn = os.path.join(a.out, f"{a.tag}{s}_{vn}.png")
            imageio.imwrite(fn, cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8))
            print("[mount]", fn, flush=True)
    print("[mount] " + json.dumps(report), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
