"""Unrestricted rotation in installed PhysX, independently of cube contacts."""
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--physics-hz', type=int, default=240)
parser.add_argument('--representation', choices=['articulation', 'rigid'], default='articulation')
parser.add_argument('--solver', choices=['TGS', 'PGS'], default='TGS')
parser.add_argument('--axis-set', choices=['pure', 'mixed'], default='pure')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.out.with_suffix('.json').exists() or args.out.with_suffix('.npz').exists():
    raise FileExistsError(args.out)
app = AppLauncher(args).app
import numpy as np
import torch
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext
from isaaclab.utils.math import quat_error_magnitude, quat_from_angle_axis, axis_angle_from_quat

dt = 1 / args.physics_hz
sim = SimulationContext(SimulationCfg(dt=dt, device=args.device, gravity=(0., 0., 0.),
                        physx=PhysxCfg(enable_external_forces_every_iteration=True, solver_type=1 if args.solver == 'TGS' else 0)))
stage = sim.stage
for index in range(6):
    path = f'/World/fixture_{index}'
    root_prim = UsdGeom.Xform.Define(stage, path)
    root_prim.AddTranslateOp().Set(Gf.Vec3d(index * .2, 0, 0))
    if args.representation == "articulation":
        UsdPhysics.ArticulationRootAPI.Apply(root_prim.GetPrim())
        PhysxSchema.PhysxArticulationAPI.Apply(root_prim.GetPrim())
    for name in ('core', 'rotor'):
        body = UsdGeom.Xform.Define(stage, path + '/' + name)
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
        mass.CreateMassAttr(.01)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(.0001, .0001, .0001))
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
        rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim())
        rigid.CreateDisableGravityAttr(True)
        rigid.CreateAngularDampingAttr(0.)
    fixed = UsdPhysics.FixedJoint.Define(stage, path + '/anchor')
    fixed.CreateBody1Rel().SetTargets([Sdf.Path(path + '/core')])
    fixed.CreateLocalPos0Attr(Gf.Vec3f(index * .2, 0, 0))
    if index < 3:
        joint = UsdPhysics.SphericalJoint.Define(stage, path + '/joint')
        joint.CreateConeAngle0LimitAttr(-1.)
        joint.CreateConeAngle1LimitAttr(-1.)
    else:
        joint = UsdPhysics.Joint.Define(stage, path + '/joint')
        for axis in ('transX', 'transY', 'transZ'):
            limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
            limit.CreateLowAttr(0.)
            limit.CreateHighAttr(0.)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(path + '/core')])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(path + '/rotor')])
    joint.CreateCollisionEnabledAttr(False)
asset = (Articulation(ArticulationCfg(prim_path='/World/fixture_.*', actuators={}))
         if args.representation == 'articulation' else
         RigidObject(RigidObjectCfg(prim_path='/World/fixture_.*/rotor')))
sim.reset()
asset.reset()
axes = (torch.eye(3, device=sim.device) if args.axis_set == 'pure' else
        torch.tensor([[1., 1., 0.], [0., 1., 1.], [1., 1., 1.]], device=sim.device))
axes = (axes / axes.norm(dim=-1, keepdim=True)).repeat(2, 1)
velocity = 2 * axes
if args.representation == 'articulation':
    if asset.num_joints != 3:
        raise RuntimeError(f'Expected3free rotational DoF, got {asset.joint_names}')
    rotor = asset.find_bodies('rotor')[0][0]
    asset.write_joint_state_to_sim(torch.zeros_like(velocity), velocity)
else:
    asset.write_root_velocity_to_sim(torch.cat((torch.zeros_like(velocity), velocity), -1))
asset.write_data_to_sim()
trace = []
try:
    for step in range(4 * args.physics_hz):
        sim.step(render=False)
        asset.update(dt)
        expected = quat_from_angle_axis(torch.full((6,), 2 * (step + 1) * dt, device=sim.device), axes)
        if args.representation == 'articulation':
            orientation = asset.data.body_quat_w[:, rotor]
            angular_velocity = asset.data.body_ang_vel_w[:, rotor]
            coordinates, speeds = asset.data.joint_pos, asset.data.joint_vel
        else:
            orientation = asset.data.root_quat_w
            angular_velocity = asset.data.root_ang_vel_w
            coordinates, speeds = axis_angle_from_quat(orientation), angular_velocity
        error = quat_error_magnitude(orientation, expected)
        trace.append(torch.cat((orientation, angular_velocity, coordinates, speeds, error[:, None]), -1).cpu().numpy())
    trace = np.asarray(trace)
    result = {'scope': 'Six isolated isotropic rotors, no collision or gravity; not a cube or hand skill.',
              'joint_types': ['spherical_negative_limits'] * 3 + ['D6_locked_translation'] * 3,
              'representation': args.representation, 'device': sim.device, 'solver': args.solver,
              'joint_names': asset.joint_names if args.representation == 'articulation' else ['observed_rotvec_x', 'observed_rotvec_y', 'observed_rotvec_z'], 'physics_hz': args.physics_hz, 'initial_speed_rad_s': 2,
              'axis_set': args.axis_set, 'axes': axes.cpu().tolist(),
              'max_orientation_error_deg': np.rad2deg(trace[:, :, -1].max(axis=0)).tolist(),
              'final_orientation_error_deg': np.rad2deg(trace[-1, :, -1]).tolist(),
              'finite': bool(np.isfinite(trace).all()),
              'trace_columns': ['qw', 'qx', 'qy', 'qz', 'wx', 'wy', 'wz',
                                'joint_q0', 'joint_q1', 'joint_q2', 'joint_v0', 'joint_v1', 'joint_v2', 'error_rad']}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out.with_suffix('.npz'), trace=trace)
    args.out.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    del asset
    sim.clear_all_callbacks()
    sim.clear_instance()
    app.close()
