"""Step-consistent friction impulse bound for numerical diagnosis.

A positive joint armature is a conservative lower bound on effective rotational
inertia. Bounding friction impulse by that inertia times speed prevents friction
alone from reversing the velocity in one explicit step. This is not a calibrated
real-cube friction model; compare against smaller-step integration before use.
"""
import torch


def bounded_friction_torque(omega, coefficient, velocity_scale, dt, inertia_lower_bound):
    """Caller must supply positive inertia bounds in kg*m^2 and dt in seconds."""
    if dt <= 0 or velocity_scale <= 0 or coefficient < 0:
        raise ValueError("Invalid friction integration parameters")
    requested = coefficient * torch.tanh(omega.abs() / velocity_scale)
    impulse_bound = inertia_lower_bound * omega.abs() / dt
    return -omega.sign() * torch.minimum(requested, impulse_bound)


def friction_damping(omega, coefficient, velocity_scale):
    """Nonnegative damping matching the smooth friction law at current speed.

    Apply through an implicit joint drive, not as an explicit -damping*omega
    effort. The zero-speed limit is coefficient/velocity_scale.
    """
    if velocity_scale <= 0 or coefficient < 0:
        raise ValueError("Invalid friction parameters")
    scaled = omega.abs() / velocity_scale
    ratio = torch.tanh(scaled) / scaled.clamp(min=1e-12)
    ratio = torch.where(scaled < 1e-6, torch.ones_like(ratio), ratio)
    return (coefficient / velocity_scale) * ratio
