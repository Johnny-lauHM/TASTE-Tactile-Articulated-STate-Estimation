"""Audit the scripted finger path horizon on cached and actual completed states.

Inverse kinematics only: no claim of collision-free mechanical execution.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from rubik_lab.flick import FlickPlanner
from rubik_lab.hand_kinematics import quat_to_mat

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
if args.out.with_suffix('.json').exists() or args.out.with_suffix('.npz').exists():
    raise FileExistsError(args.out)
cache = np.load('cache/g70s_cycle_demo1_v3.npz')
record = np.load('outputs/diagnostics/continuation_wait0.npz')
trace = record['trace']
origins = []
for environment in range(trace.shape[1]):
    pose = record['poses'][0, environment]
    estimates = [np.round(2 * (pose[7 + h * 7:10 + h * 7] - cache[side + '_ee_pos'][0])) / 2
                 for h, side in enumerate(('left', 'right'))]
    if not np.array_equal(*estimates):
        raise ValueError('Ambiguous recorded environment origin')
    origins.append(estimates[0])
entries = []
for trial in np.unique(trace[:, :, 0]).astype(int):
    for environment in range(trace.shape[1]):
        rows = np.flatnonzero((trace[:, environment, 0] == trial) & (trace[:, environment, 3] == 0)
                              & (trace[:, environment, 4] > 0) & (trace[:, environment, 10] > 0))
        if len(rows):
            entries.append((int(rows[0]), environment, trial))
results, arrays = {}, {}
for move, side, hand, direction in [('U', 'right', 1, -1), ("U'", 'left', 0, 1)]:
    positions, rotations, fingers, phase, labels = [], [], [], [], []
    core_rotation = quat_to_mat(torch.tensor(cache['cube_quat'][0])).numpy()
    positions.append(core_rotation.T @ (cache[side + '_ee_pos'][0] - cache['cube_pos'][0]))
    rotations.append(core_rotation.T @ quat_to_mat(torch.tensor(cache[side + '_ee_quat'][0])).numpy())
    fingers.append(cache[side + '_fingers'][0])
    phase.append(float(cache['cube_twist'][0]) * direction)
    labels.append({'source': 'canonical_cache'})
    for frame, environment, trial in entries:
        pose = record['poses'][frame, environment]
        rotation = quat_to_mat(torch.tensor(pose[3:7])).numpy()
        wrist = pose[7 + hand * 7:14 + hand * 7]
        positions.append(rotation.T @ (wrist[:3] - origins[environment] - pose[:3]))
        rotations.append(rotation.T @ quat_to_mat(torch.tensor(wrist[3:])).numpy())
        fingers.append(record['sensory'][frame, environment, hand * 22:(hand + 1) * 22])
        phase.append(-np.deg2rad(trace[frame, environment, 5]) * direction)
        labels.append({'source': 'actual_first_complete', 'trial': int(trial), 'env': int(environment),
                       'first_move': 'U' if trace[frame, environment, 2] < 2 else "U'", 'frame': int(frame)})
    inputs = [torch.tensor(np.asarray(x), dtype=torch.float32, device='cuda:0')
              for x in (positions, rotations, fingers, phase)]
    for horizon in (70, 95):
        planner = FlickPlanner('cuda:0', max_deg=horizon)
        q, dp, dr, ok = planner.solve(move, *inputs[:3], phi=inputs[3])
        reach = planner.reach_deg(ok).cpu().numpy()
        key = ('U' if move == 'U' else 'Up') + '_' + str(horizon)
        arrays[key + '_prefix_ok'] = ok.cpu().numpy()
        arrays[key + '_raw_ok'] = planner.last_ok_raw.cpu().numpy()
        arrays[key + '_pad'] = planner.last_pads.cpu().numpy()
        arrays[key + '_target'] = planner.last_targets.cpu().numpy()
        arrays[key + '_q'] = q.cpu().numpy()
        results[key] = {'horizon_degrees': horizon, 'reach_degrees': reach.tolist(),
                        'at_least_90_count': int((reach >= 90).sum()),
                        'count': len(reach), 'inputs': labels}
        print(key, 'cache', reach[0], 'actual median', np.median(reach[1:]), '90deg', int((reach >= 90).sum()), flush=True)
args.out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(args.out.with_suffix('.npz'), **arrays)
args.out.with_suffix('.json').write_text(json.dumps({'scope': 'Kinematic path-horizon diagnostic only; no collision/force/control acceptance.',
    'input_trace': 'outputs/diagnostics/continuation_wait0.npz', 'all_first_complete_states_included': True,
    'wrist_assistance': False, 'results': results}, indent=2) + '\n')
