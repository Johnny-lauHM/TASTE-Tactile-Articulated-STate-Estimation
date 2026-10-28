"""Plot predeclared env0/trial0 U->U geometry from complete diagnostic traces."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from rubik_lab.hand_kinematics import HandFK, quat_to_mat

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
if args.out.exists():
    raise FileExistsError(args.out)
kinematics = {side: HandFK(side, 'cpu', pts_per_link=24) for side in ('left', 'right')}
figure = plt.figure(figsize=(14, 9))
faces = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
for row, name in enumerate(('continuation_wait0', 'continuation_cartesian_replay')):
    record = np.load(Path('outputs/diagnostics') / (name + '.npz'))
    trace = record['trace'][:, 0]
    cache = np.load('cache/g70s_cycle_demo1_v3.npz')
    initial_pose = record['poses'][0, 0]
    candidates = [np.round((initial_pose[7 + 7 * h:10 + 7 * h] - cache[side + '_ee_pos'][0]) * 2) / 2
                  for h, side in enumerate(('left', 'right'))]
    if not np.array_equal(candidates[0], candidates[1]):
        raise ValueError('Cannot reconstruct known1mgrid origin consistently from both wrists')
    origin = record['env_origins'][0] if 'env_origins' in record else candidates[0]
    valid = (trace[:, 0] == 0) & (trace[:, 4] > 0)
    first = np.flatnonzero(valid & (trace[:, 3] == 0))
    second = np.flatnonzero(valid & (trace[:, 3] == 1))
    selected = [first[0], first[-1], second[-1]]
    for column, frame in enumerate(selected):
        axis = figure.add_subplot(2, 3, row * 3 + column + 1, projection='3d')
        pose = record['poses'][frame, 0]
        center = pose[:3]
        rotation = quat_to_mat(torch.tensor(pose[3:7])).numpy()
        # For this fixed U->U pair, cumulative ideal target is -90*(stage+1).
        angle = np.deg2rad(-90 * (trace[frame, 3] + 1) - trace[frame, 5])
        rz = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        for low, high, rot, color in ((-.035, .035 / 3, np.eye(3), 'gray'),
                                       (.035 / 3, .035, rz, 'gold')):
            vertices = np.array([[x, y, z] for x in (-.035, .035) for y in (-.035, .035) for z in (low, high)]) @ rot.T * 1000
            axis.add_collection3d(Poly3DCollection([vertices[face] for face in faces], alpha=.18, facecolor=color, edgecolor=color))
        for hand, side in enumerate(('left', 'right')):
            wrist = torch.tensor(pose[7 + 7 * hand:14 + 7 * hand]).unsqueeze(0)
            wrist[:, :3] -= torch.tensor(origin)
            joints = torch.tensor(record['sensory'][frame, 0, 22 * hand:22 * (hand + 1)]).unsqueeze(0)
            fk = kinematics[side]
            state = fk.forward(wrist[:, :3], quat_to_mat(wrist[:, 3:]), joints)
            points = fk.points(state)
            for link, cloud in points.items():
                local = (cloud[0].numpy() - center) @ rotation * 1000
                color = ('royalblue' if hand == 0 else 'crimson') if '_index_' in link else 'silver'
                axis.scatter(*local.T, s=2, alpha=.45, color=color)
            pad, _ = fk.pads(state)['index']
            local = (pad[0].numpy() - center) @ rotation * 1000
            axis.scatter(*local, s=50, color='royalblue' if hand == 0 else 'crimson')
        label = ('initial', 'first complete', 'second terminal')[column]
        axis.set_title(name.replace('continuation_', '') + '\n' + label + f' | error {trace[frame, 5]:.1f} deg')
        axis.set(xlim=(-85, 85), ylim=(-100, 100), zlim=(-60, 100), xlabel='cube X (mm)', ylabel='cube Y (mm)', zlabel='cube Z (mm)')
        axis.set_box_aspect((170, 200, 160))
        axis.view_init(elev=28, azim=130)
figure.suptitle('Recorded env 0, trial 0: geometry from q/wrist poses;1mgrid origin inferred from both cached wrists; no resimulation')
figure.tight_layout()
args.out.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(args.out, dpi=140)
