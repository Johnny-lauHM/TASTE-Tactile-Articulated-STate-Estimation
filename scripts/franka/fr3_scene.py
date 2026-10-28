"""Render the two FR3 + adapter + Sharpa arms (~/workspace/fr3-sharpa-rubik assets) holding the cube in the solve
env's ready grip, posed kinematically from scripts/franka/fr3_reach.py (no physics settling).

    python scripts/franka/fr3_scene.py --headless --enable_cameras [--reach assets_franka/fr3_reach.json] [--out outputs/franka]
"""
import argparse
import json
import os

from isaaclab.app import AppLauncher

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
FR3 = os.path.expanduser("~/workspace/fr3-sharpa-rubik")
p = argparse.ArgumentParser()
p.add_argument("--reach", default=os.path.join(ROOT, "assets_franka/fr3_reach.json"))
p.add_argument("--poses", default="ready,L,x")
p.add_argument("--out", default=os.path.join(ROOT, "outputs/franka"))
p.add_argument("--tag", default="fr3_")
p.add_argument("--assembly", action="store_true", help="close-ups of each side's FR3 flange -> adapter -> hand instead of the views")
p.add_argument("--usd", default=os.path.join(FR3, "assets/{side}/robot.usd"), help="robot USD per side ({side} is filled in)")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402

REACH = json.load(open(a.reach))
BUILD = json.load(open(os.path.join(FR3, "reports/asset_build.json")))
C = np.array(REACH["cube_position"])
VIEWS = {"front": (tuple(C + [0.95, 0.0, 0.3]), tuple(C + [0, 0, -0.05])), "oblique": (tuple(C + [0.8, -1.0, 0.55]), tuple(C + [-0.15, 0, -0.1])),
         "close": (tuple(C + [0.35, -0.25, 0.2]), tuple(C))}


def yaw_quat(deg):
    h = np.radians(deg) / 2
    return (float(np.cos(h)), 0.0, 0.0, float(np.sin(h)))


def robot(side):
    return Articulation(ArticulationCfg(
        prim_path=f"/World/{side}",
        spawn=sim_utils.UsdFileCfg(usd_path=a.usd.format(side=side),
                                   rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)),
        init_state=ArticulationCfg.InitialStateCfg(pos=tuple(REACH["base"][side]), rot=yaw_quat(REACH.get("base_yaw_deg", {}).get(side, 0.0)), joint_pos=dict(
            zip(BUILD[side]["joint_names"], REACH["poses"]["ready"][side]["arm_q"] + REACH["fingers"][side]))),
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=1e4, damping=100.0)}))


def main():
    import sys
    sys.excepthook = lambda *e: (__import__("traceback").print_exception(*e), os._exit(1))
    ctx = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 240, device=a.device, gravity=(0.0, 0.0, 0.0)))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2500.0))
    robots = {s: robot(s) for s in ("left", "right")}
    cube = Articulation(ArticulationCfg(
        prim_path="/World/cube",
        spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(ROOT, "assets/rubik_cube/rubik_cube_2body.usda"),
                                   rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)),
        init_state=ArticulationCfg.InitialStateCfg(pos=tuple(REACH["cube_position"]), rot=tuple(REACH["cube_quaternion"])),
        actuators={"p": ImplicitActuatorCfg(joint_names_expr=["twist_joint"], stiffness=0.0, damping=0.0)}))
    cams = {v: Camera(CameraCfg(prim_path=f"/World/cam_{v}", height=720, width=960, data_types=["rgb"],
                                spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 20.0))))
            for v in VIEWS}
    ctx.reset()
    for v, (eye, tgt) in VIEWS.items():
        cams[v].set_world_poses_from_view(torch.tensor([eye], device=a.device, dtype=torch.float32), torch.tensor([tgt], device=a.device, dtype=torch.float32))
    os.makedirs(a.out, exist_ok=True)
    if a.assembly:  # cameras around each side's mount (FR3 link8 -> adapter -> hand flange), at the ready pose
        for s, r in robots.items():
            names = BUILD[s]["joint_names"]
            q = torch.zeros_like(r.data.joint_pos)
            vals = dict(zip(names, REACH["poses"]["ready"][s]["arm_q"] + REACH["fingers"][s]))
            for j, nm in enumerate(r.joint_names):
                q[0, j] = vals.get(nm, 0.0)
            r.write_joint_state_to_sim(q, torch.zeros_like(q))
            r.set_joint_position_target(q)
            r.write_data_to_sim()
        for _ in range(4):
            ctx.step()
        for s, r in robots.items():
            pts = [r.data.body_pos_w[0, r.find_bodies(b)[0][0]] for b in ("fr3_link8", f"{s}_adapter", f"{s}_hand_flange")]
            c = torch.stack(pts).mean(0)
            axis = pts[2] - pts[0]
            side_dir = torch.tensor([0.0, 1.0 if s == "left" else -1.0, 0.0], device=a.device)
            for vname, off in {"outside": side_dir * 0.32 + torch.tensor([0.0, 0.0, 0.08], device=a.device),
                               "above": torch.tensor([0.05, 0.0, 0.32], device=a.device),
                               "back": torch.tensor([-0.32, 0.0, 0.10], device=a.device) + side_dir * 0.08}.items():
                cam = cams["close"]
                cam.set_world_poses_from_view((c + off).unsqueeze(0), c.unsqueeze(0))
                for _ in range(3):
                    ctx.render()
                cam.update(dt=0.0)
                img = cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
                fn = os.path.join(a.out, f"{a.tag}{s}_{vname}.png")
                imageio.imwrite(fn, img)
                print("[scene]", fn, flush=True)
        os._exit(0)
    for pose in a.poses.split(","):
        for s, r in robots.items():
            names = BUILD[s]["joint_names"]
            q = torch.zeros_like(r.data.joint_pos)
            vals = dict(zip(names, REACH["poses"][pose][s]["arm_q"] + REACH["fingers"][s]))
            for j, n in enumerate(r.joint_names):
                q[0, j] = vals.get(n, 0.0)
            r.write_joint_state_to_sim(q, torch.zeros_like(q))
            r.set_joint_position_target(q)
            r.write_data_to_sim()
        deg = {"x": 90.0, "x'": -90.0}.get(pose, 0.0)  # a whole-cube roll turns the cube with both wrists
        h = np.radians(deg) / 2
        qy = np.array([np.cos(h), 0.0, np.sin(h), 0.0])
        q0 = np.array(REACH["cube_quaternion"])
        qc = np.array([qy[0] * q0[0] - qy[2] * q0[2], qy[0] * q0[1] + qy[2] * q0[3], qy[0] * q0[2] + qy[2] * q0[0],
                       qy[0] * q0[3] - qy[2] * q0[1]])
        cube.write_root_pose_to_sim(torch.tensor([list(REACH["cube_position"]) + qc.tolist()], device=a.device, dtype=torch.float32))
        for _ in range(4):
            ctx.step()
        for _ in range(3):
            ctx.render()
        for v, c in cams.items():
            c.update(dt=0.0)
            img = c.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
            fn = os.path.join(a.out, f"{a.tag}{pose.replace(chr(39), 'p')}_{v}.png")
            imageio.imwrite(fn, img)
            print("[scene]", fn, flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
