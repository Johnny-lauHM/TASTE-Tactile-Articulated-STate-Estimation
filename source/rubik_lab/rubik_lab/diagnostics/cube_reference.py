"""Minimal native robogym collision-model test; no hand or learned-skill claim.

Run in the isolated MuJoCo dependency directory, not the Isaac Python path.
Only initialization uses mj_resetData. All face motion comes from applied torque.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--seconds-per-move', type=float, default=2.0)
parser.add_argument('--integral-gain', type=float, default=0.0)
parser.add_argument('--cubelet-joints', choices=['native-euler', 'ball'], default='native-euler')
parser.add_argument('--cases', choices=['all', 'sequence'], default='all')
parser.add_argument('--contact-time-constant', type=float)
args = parser.parse_args()
if args.out.with_suffix('.json').exists() or args.out.with_suffix('.npz').exists():
    raise FileExistsError(args.out)
root = Path(__file__).resolve().parents[4]
assets = root / 'assets/reference/robogym'
from rubik_lab.diagnostics.robogym_reference import load_model
model = load_model(assets, args.cubelet_joints)
if args.contact_time_constant is not None:
    if args.contact_time_constant < 2 * model.opt.timestep:
        raise ValueError("Contacttimeconstant mustrespect MuJoCo2dt safetybound")
    model.geom_solref[:, 0] = args.contact_time_constant
data = mujoco.MjData(model)
body_ids = np.array([i for i in range(model.nbody)
                     if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or '').startswith('cubelet:')])
body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(i)) for i in body_ids]
if len(body_ids) != 26:
    raise ValueError('Expected26physical cubelets')
faces = {'U': ('pos_z', 2, 1), 'D': ('neg_z', 2, -1),
         'R': ('neg_y', 1, -1), 'L': ('pos_y', 1, 1),
         'F': ('neg_x', 0, -1), 'B': ('pos_x', 0, 1)}
face_names = list(faces)
joints = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                   'cubelet:driver:' + faces[f][0]) for f in face_names])
qids, vids = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
cases = [[f + suffix] for f in face_names for suffix in ('', "'", '2')]
cases.append(['U', 'R', "U'", "R'"])
if args.cases == "sequence":
    cases = [cases[-1]]
records, raw, contacts = [], [], []
for trial, sequence in enumerate(cases):
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    expected_pos = data.xpos[body_ids].copy()
    expected_rot = data.xmat[body_ids].reshape(-1, 3, 3).copy()
    targets = np.zeros(6)
    integral = np.zeros(6)
    for stage, move in enumerate(sequence):
        face, suffix = move[0], move[1:]
        _, axis, sign = faces[face]
        angle = np.pi if suffix == '2' else (-np.pi / 2 if suffix == '' else np.pi / 2)
        angle *= sign
        selected = expected_pos[:, axis] * sign > 0.009
        vector = np.zeros(3)
        vector[axis] = 1
        cross = np.array([[0, -vector[2], vector[1]], [vector[2], 0, -vector[0]],
                          [-vector[1], vector[0], 0]])
        rotation = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross
        expected_pos[selected] = expected_pos[selected] @ rotation.T
        expected_rot[selected] = rotation @ expected_rot[selected]
        targets[face_names.index(face)] += angle
        automatic_resets = 0
        for step in range(round(args.seconds_per_move / model.opt.timestep)):
            error = targets - data.qpos[qids]
            unconstrained = 0.1 * error - 0.02 * data.qvel[vids] + args.integral_gain * integral
            torque = np.clip(unconstrained, -0.05, 0.05)
            # Conditional integration prevents windup while the torque saturates.
            integrate = (np.abs(unconstrained) <= .05) | (error * unconstrained < 0)
            integral = np.clip(integral + integrate * error * model.opt.timestep, -.1, .1)
            data.qfrc_applied[:] = 0
            data.qfrc_applied[vids] = torque
            previous_time = data.time
            mujoco.mj_step(model, data)
            automatic_resets += int(data.time <= previous_time)
            minimum = 0.0
            worst_pair = [-1, -1]
            max_normal = 0.0
            max_tangent = 0.0
            wrench = np.zeros(6)
            for contact_index, contact in enumerate(data.contact):
                if contact.dist < minimum:
                    minimum = float(contact.dist)
                    worst_pair = [int(contact.geom1), int(contact.geom2)]
                mujoco.mj_contactForce(model, data, contact_index, wrench)
                max_normal = max(max_normal, abs(float(wrench[0])))
                max_tangent = max(max_tangent, float(np.linalg.norm(wrench[1:3])))
            contacts.append([trial, stage, step, previous_time, minimum, max_normal, max_tangent, *worst_pair])
            if not np.isfinite(data.qpos).all():
                raise RuntimeError('Nonfinite physical state')
            raw.append(np.concatenate(([trial, stage, step, data.ncon, data.time], torque,
                                       data.qpos.copy(), data.qvel.copy())))
        mujoco.mj_forward(model, data)
        pos_error = np.linalg.norm(data.xpos[body_ids] - expected_pos, axis=1)
        actual_rot = data.xmat[body_ids].reshape(-1, 3, 3)
        relative = actual_rot @ expected_rot.transpose(0, 2, 1)
        rot_error = np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1))
        records.append({'trial': trial, 'stage': stage, 'sequence': sequence, 'move': move,
                        'center_angle_error_deg': float(np.rad2deg(targets[face_names.index(face)] - data.qpos[qids[face_names.index(face)]])),
                        'cubelet_position_error_mm': (1000 * pos_error).tolist(),
                        'cubelet_orientation_error_deg': np.rad2deg(rot_error).tolist(),
                        'automatic_reset_count': automatic_resets,
                        'all_cubelets_within_2mm_5deg': bool(not automatic_resets and (pos_error < .002).all() and (rot_error < np.deg2rad(5)).all())})
    print('reference', trial, sequence, records[-1]['all_cubelets_within_2mm_5deg'], flush=True)
args.out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(args.out.with_suffix('.npz'), trace=np.asarray(raw, dtype=np.float64), contact_summary=np.asarray(contacts, dtype=np.float64))
report = {'scope': 'Native source-model diagnostic: fixed core, no hands, privileged center-joint torque servos. Never a Sharpa skill or acceptance test.',
          'source_revision': '9ff1111fd557b1a01911d7d8c9d59a453071c96e',
          'mujoco_version': mujoco.__version__, 'numpy_version': np.__version__,
          'cubelet_joints': args.cubelet_joints, 'cases': args.cases,
          'geom_solref': model.geom_solref.tolist(), 'geom_solimp': model.geom_solimp.tolist(),
          'contact_summary_columns': ['trial', 'stage', 'step', 'force_evaluation_time_before_integration', 'min_distance_m', 'max_normal_force_N', 'max_tangent_force_N', 'deepest_geom1', 'deepest_geom2'],
          'geom_names': [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)],
          'peak_penetration_mm': float(-1000 * np.asarray(contacts)[:, 4].min()),
          'source_changes': ['Removed textures/materials only for headless use', 'Removed radial spring joints as native FullPerpendicularSimulation does',
                             'Dynamic core retained and physically welded to world, preserving native parent collision filtering; gravity disabled', 'Added bounded torque servos to six existing center hinges; no cubelet pose writes during execution'],
          'timestep': model.opt.timestep, 'seconds_per_move': args.seconds_per_move,
          'servo': {'kp': .1, 'kd': .02, 'ki': args.integral_gain, 'integral_bound': .1, 'torque_limit_Nm': .05},
          'joint_count': model.njnt, 'nq': model.nq, 'nv': model.nv,
          'cubelet_names': body_names, 'face_order': face_names,
          'trace_columns': {'metadata': ['trial', 'stage', 'step', 'ncon', 'sim_time'], 'torque': [5, 11],
                            'qpos': [11, 11 + model.nq], 'qvel': [11 + model.nq, 11 + model.nq + model.nv]},
          'episodes': records}
args.out.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
