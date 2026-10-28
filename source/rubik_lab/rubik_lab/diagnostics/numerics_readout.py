"""Independent continuous-time check of the isolated loaded friction bench."""
import argparse
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp

parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=Path, default=Path('outputs/diagnostics'))
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
if args.out.exists():
    raise FileExistsError(args.out)

names = ['friction_implicit_each_iteration', 'friction_implicit_loaded_960',
         'friction_loaded_reference']
reports = {name: json.loads((args.directory / (name + '.json')).read_text()) for name in names}
reference = reports['friction_implicit_loaded_960']
matrices = np.asarray(reference['generalized_mass_matrix'])
if matrices.shape != (8, 1, 1):
    raise ValueError('Reference must be the fixed-core, single-joint bench')
inertia = matrices[:, 0, 0] + np.asarray(reference['inertia_lower_bound'])
damping = np.asarray(reference['base_damping'])
initial = np.asarray(reference['initial_omega_rad_s'])
load = reference['external_torque_Nm']


def derivative(time, state):
    velocity = state[8:]
    return np.concatenate((velocity, (load - 0.01 * np.tanh(velocity / 0.05)
                                      - damping * velocity) / inertia))


solution = solve_ivp(derivative, (0, 1), np.concatenate((np.zeros(8), initial)),
                     rtol=1e-10, atol=1e-12, dense_output=True)
if not solution.success:
    raise RuntimeError(solution.message)
result = {
    'scope': 'Independent ODE check of isolated numerical friction; not contact or real-cube calibration.',
    'equation': 'I*omega_dot = external_torque - 0.01*tanh(omega/0.05) - base_damping*omega',
    'inertia_kg_m2': inertia.tolist(),
    'inertia_source': 'Native fixed-core generalized mass matrix plus separately configured joint armature.',
    'scipy_version': scipy.__version__,
    'solver': {'method': 'RK45', 'rtol': 1e-10, 'atol': 1e-12},
    'comparisons': {},
}
for name, report in reports.items():
    if not report['forces_each_iteration'] or report['external_torque_Nm'] != load:
        raise ValueError('Mismatched force timing or load: ' + name)
    if not np.allclose(report['initial_omega_rad_s'], initial):
        raise ValueError('Mismatched initial state: ' + name)
    trace = np.load(args.directory / (name + '.npz'))['trace']
    times = np.arange(1, len(trace) + 1) / report['physics_hz']
    expected = solution.sol(times)
    angle_error = np.rad2deg(trace[:, :, 3] - expected[:8].T)
    result['comparisons'][name] = {
        'final_angle_error_deg': angle_error[-1].tolist(),
        'max_absolute_angle_error_deg': np.abs(angle_error).max(axis=0).tolist(),
        'final_speed_error_rad_s': (trace[-1, :, 2] - expected[8:, -1]).tolist(),
    }
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result['comparisons'], indent=2))
