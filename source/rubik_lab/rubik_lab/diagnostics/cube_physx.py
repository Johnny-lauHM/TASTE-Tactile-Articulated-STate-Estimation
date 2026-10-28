"""No-hand PhysX counterpart of the pinned source cube fixture; never skill acceptance."""
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--seconds-per-move', type=float, default=6.)
parser.add_argument('--position-iterations', type=int, default=8)
parser.add_argument('--record-contacts', action='store_true')
parser.add_argument('--contact-piece')
parser.add_argument('--rest-offset', type=float, default=0.)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.out.with_suffix('.json').exists() or args.out.with_suffix('.npz').exists():
    raise FileExistsError(args.out)
app = AppLauncher(args).app
import numpy as np
import torch
from pxr import Gf, Sdf, UsdPhysics, PhysxSchema
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul, quat_error_magnitude

root = Path(__file__).resolve().parents[4]
dt = 1 / 960
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args.device,
    gravity=(0., 0., 0.), physx=sim_utils.PhysxCfg(solver_type=0)))
cube = Articulation(ArticulationCfg(prim_path='/World/cube', spawn=sim_utils.UsdFileCfg(
    usd_path=str(root / 'assets/rubik_cube/robogym_ball_reference.usda'),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
    collision_props=sim_utils.CollisionPropertiesCfg(rest_offset=args.rest_offset, contact_offset=.0001),
    activate_contact_sensors=args.record_contacts),
    actuators={'passive': ImplicitActuatorCfg(joint_names_expr=['.*'], stiffness=0., damping=.0005, effort_limit_sim=1., velocity_limit_sim=1000.)}))
anchor = UsdPhysics.FixedJoint.Define(sim.stage, '/World/cube/fixture_anchor')
anchor.CreateBody1Rel().SetTargets([Sdf.Path('/World/cube/core')])
anchor.CreateLocalPos0Attr(Gf.Vec3f(0., 0., 0.))
PhysxSchema.PhysxArticulationAPI(sim.stage.GetPrimAtPath("/World/cube")).CreateSolverPositionIterationCountAttr(args.position_iterations)
contact_sensor = None
if args.record_contacts:
    filter_paths = [str(p.GetPath()) for p in sim.stage.GetPrimAtPath("/World/cube").GetChildren() if p.GetName().startswith("piece_")]
    sensor_path = "/World/cube/" + args.contact_piece if args.contact_piece else "/World/cube/piece_.*"
    contact_sensor = ContactSensor(ContactSensorCfg(prim_path=sensor_path,
        track_contact_points=True, max_contact_data_count_per_prim=128,
        filter_prim_paths_expr=filter_paths))
sim.reset()
cube.reset()
if cube.num_joints != 66 or cube.num_bodies != 27:
    raise RuntimeError(f'Unexpected topology: {cube.num_joints} joints, {cube.num_bodies} bodies')
# Damping has torque/(rad/s) units through the native tensor API.
cube.write_joint_damping_to_sim(torch.full_like(cube.data.joint_pos, .0005))
faces = {'U': ('pos_z', 2, 1), 'D': ('neg_z', 2, -1), 'R': ('neg_y', 1, -1),
         'L': ('pos_y', 1, 1), 'F': ('neg_x', 0, -1), 'B': ('pos_x', 0, 1)}
face_names = list(faces)
joints = [cube.find_joints('driver_' + faces[f][0])[0][0] for f in face_names]
body_ids = [i for i, name in enumerate(cube.body_names) if name.startswith('piece_')]
expected_pos = cube.data.body_com_pos_w[0, body_ids].clone()
expected_quat = cube.data.body_quat_w[0, body_ids].clone()
target = torch.zeros(6, device=sim.device)
integral = torch.zeros_like(target)
sequence = ['U', 'R', "U'", "R'"]
trace, poses, records, contact_summaries = [], [], [], []
try:
    for stage, move in enumerate(sequence):
        face = move[0]
        _, axis, sign = faces[face]
        angle = sign * (torch.pi / 2 if move.endswith("'") else -torch.pi / 2)
        vector = torch.zeros(3, device=sim.device)
        vector[axis] = 1
        rotation = quat_from_angle_axis(torch.tensor(angle, device=sim.device), vector)
        selected = expected_pos[:, axis] * sign > .009
        expected_pos[selected] = quat_apply(rotation.expand(int(selected.sum()), 4), expected_pos[selected])
        expected_quat[selected] = quat_mul(rotation.expand(int(selected.sum()), 4), expected_quat[selected])
        target[face_names.index(face)] += angle
        for step in range(round(args.seconds_per_move / dt)):
            q, velocity = cube.data.joint_pos, cube.data.joint_vel
            error = target - q[0, joints]
            raw = .1 * error - .02 * velocity[0, joints] + .1 * integral
            control = raw.clamp(-.05, .05)
            integrate = (raw.abs() <= .05) | (error * raw < 0)
            integral = (integral + integrate * error * dt).clamp(-.1, .1)
            # Explicitly declared smooth approximation of the native absolute frictionloss.
            torque = -.0005 * torch.tanh(velocity / .05)
            torque[0, joints] += control
            cube.set_joint_effort_target(torque)
            cube.write_data_to_sim()
            sim.step(render=False)
            cube.update(dt)
            if contact_sensor is not None:
                force, points, normals, separation, counts, starts = contact_sensor.contact_physx_view.get_contact_data(dt=dt)
                counts, starts = counts.flatten().long(), starts.flatten().long()
                total = int(counts.sum())
                minimum, maximum = 0., 0.
                if total:
                    pair = torch.repeat_interleave(torch.arange(len(counts), device=counts.device), counts)
                    within = torch.arange(total, device=counts.device) - torch.repeat_interleave(counts.cumsum(0) - counts, counts)
                    indices = starts[pair] + within
                    minimum = float(separation[indices].min())
                    maximum = float(force[indices].abs().max())
                contact_summaries.append([stage, step, total, minimum, maximum])
            trace.append(torch.cat((torch.tensor([[stage, step]], device=sim.device),
                                    cube.data.joint_pos, cube.data.joint_vel, torque, cube._joint_effort_target_sim), -1)[0].cpu().numpy())
            poses.append(torch.cat((cube.data.body_com_pos_w, cube.data.body_quat_w), -1)[0].cpu().numpy())
            if not torch.isfinite(cube.data.body_quat_w).all():
                raise RuntimeError('Nonfinite physical state')
        position_error = (cube.data.body_com_pos_w[0, body_ids] - expected_pos).norm(dim=-1)
        rotation_error = quat_error_magnitude(cube.data.body_quat_w[0, body_ids], expected_quat)
        record = {'move': move, 'center_angle_error_deg': float(torch.rad2deg(target[face_names.index(face)] - cube.data.joint_pos[0, joints[face_names.index(face)]])),
                  'cubelet_position_error_mm': (position_error * 1000).cpu().tolist(),
                  'cubelet_orientation_error_deg': torch.rad2deg(rotation_error).cpu().tolist(),
                  'all_cubelets_within_2mm_5deg': bool((position_error < .002).all() and (rotation_error < torch.deg2rad(torch.tensor(5.))).all())}
        records.append(record)
        print(json.dumps({'move': move, 'pass': record['all_cubelets_within_2mm_5deg'], 'center_error': record['center_angle_error_deg']}), flush=True)
    result = {'scope': 'Center-torque-driven fixed-core source-size PhysX fixture; no hands, no sensor-only controller, no acceptance claim.',
              'asset': 'assets/rubik_cube/robogym_ball_reference.usda', 'device': sim.device,
              'solver': 'PGS', 'rest_offset_m': args.rest_offset, 'record_contacts': args.record_contacts, 'contact_piece': args.contact_piece,
              'contact_view_shape': [contact_sensor.contact_physx_view.sensor_count, contact_sensor.contact_physx_view.filter_count] if contact_sensor is not None else None,
              'contact_summary_columns': ['stage', 'step', 'reported_contact_count', 'minimum_separation_m', 'max_normal_force_N'],
              'position_iterations': args.position_iterations, 'physics_hz': 960, 'seconds_per_move': args.seconds_per_move,
              'sequence': sequence, 'joint_names': cube.joint_names, 'body_names': cube.body_names,
              'actual_mass_kg': cube.root_physx_view.get_masses().cpu().tolist(),
              'actual_joint_armature': cube.data.joint_armature.cpu().tolist(),
              'actual_joint_damping': cube.data.joint_damping.cpu().tolist(),
              'source_differences': ['MuJoCo dry frictionloss replaced by explicit0.0005*tanh(velocity/0.05)',
                                     'Rolling/torsionalcontactfriction absent; slidingfriction0.2 only',
                                     'PGS/convexhull contactsolver and0.1mmcontactoffset differ'],
              'trace_columns': {'stage_step': [0, 2], 'q': [2, 68], 'qd': [68, 134], 'commanded_torque': [134, 200], 'physx_effort_target': [200, 266]},
              'poses': 'physics_step,body,COMxyz+quaternion_wxyz', 'episodes': records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out.with_suffix('.npz'), trace=np.asarray(trace), poses=np.asarray(poses), contact_summary=np.asarray(contact_summaries))
    args.out.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    if contact_sensor is not None:
        del contact_sensor
    del cube
    sim.clear_all_callbacks()
    sim.clear_instance()
    app.close()
