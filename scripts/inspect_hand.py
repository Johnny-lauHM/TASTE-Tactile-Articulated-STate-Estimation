"""Print joints / bodies / limits / gains of the float-base Sharpa hands and collision extents in the wrist frame.

    python scripts/inspect_hand.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from pxr import Usd, UsdGeom  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

from rubik_lab.assets import RUBIK_CUBE_CFG, sharpa_hand_cfg  # noqa: E402

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 240))
hands = {}
for i, side in enumerate(("left", "right")):
    cfg = sharpa_hand_cfg(side).replace(prim_path=f"/World/{side}_hand")
    cfg.init_state.pos = (0.0, 0.5 * i, 0.5)
    hands[side] = Articulation(cfg)
cube = Articulation(RUBIK_CUBE_CFG.replace(prim_path="/World/cube"))
sim.reset()

for side, h in hands.items():
    print(f"\n===== {side} hand: {h.num_joints} joints, {h.num_bodies} bodies, fixed base={h.is_fixed_base}")
    lim = h.data.joint_pos_limits[0]
    for j, name in enumerate(h.joint_names):
        print(
            f"  {name:24s} [{lim[j, 0]:+.3f}, {lim[j, 1]:+.3f}]  kp={h.data.joint_stiffness[0, j]:.4g}"
            f"  kd={h.data.joint_damping[0, j]:.4g}  effort={h.data.joint_effort_limits[0, j]:.4g}"
            f"  armature={h.data.joint_armature[0, j]:.4g}"
        )
    print("  bodies:", h.body_names)
    for a_name, act in h.actuators.items():
        print(f"  actuator {a_name}: {type(act).__name__}  kp={act.stiffness[0].tolist()}")

print(f"\n===== cube: joints={cube.joint_names} bodies={cube.body_names} masses={cube.data.default_mass[0].tolist()}")

# collision extents of each link expressed in the wrist frame (zero joint configuration)
stage = sim.stage
for side in ("left", "right"):
    root = f"/World/{side}_hand"
    wrist = next(p for p in Usd.PrimRange(stage.GetPrimAtPath(root)) if p.GetName() == f"{side}_hand_wrist")
    w_inv = UsdGeom.Xformable(wrist).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
    print(f"\n===== {side} collision extents in wrist frame (x=palm normal, z=fingers)")
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if prim.GetName() != "collisions" and "collision" not in prim.GetName().lower():
            continue
        if not prim.IsA(UsdGeom.Imageable):
            continue
        bbox = cache.ComputeWorldBound(prim)
        bbox.Transform(w_inv)
        r = bbox.ComputeAlignedRange()
        if r.IsEmpty():
            continue
        lo, hi = r.GetMin(), r.GetMax()
        print(f"  {str(prim.GetPath()).replace(root, ''):60s} x[{lo[0]:+.3f},{hi[0]:+.3f}] y[{lo[1]:+.3f},{hi[1]:+.3f}] z[{lo[2]:+.3f},{hi[2]:+.3f}]")

app.close()
