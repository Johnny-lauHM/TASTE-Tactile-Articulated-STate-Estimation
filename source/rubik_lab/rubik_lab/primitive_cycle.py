"""Batched full-primitive acceptance, independent of Isaac Sim.

This monitor consumes privileged measurements for training/evaluation only. It is
NOT a tactile completion detector and must never be described as one.
"""
from __future__ import annotations

import math
import torch


class CycleMonitor:
    def __init__(self, n: int, device: str, dt: float, stable_seconds: float):
        self.required_steps = max(1, math.ceil(stable_seconds / dt - 1e-9))
        self.stable_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.reached = torch.zeros(n, dtype=torch.bool, device=device)
        self.failed = torch.zeros(n, dtype=torch.bool, device=device)
        self.done = torch.zeros(n, dtype=torch.bool, device=device)
        self.ready = torch.zeros(n, dtype=torch.bool, device=device)

    def reset(self, ids):
        for field in (self.stable_steps, self.reached, self.failed, self.done, self.ready):
            field[ids] = 0

    def update(self, *, layer_ok, fingers_ready, grasp_ok, motion_ok, failed):
        # Motion limits apply to the whole primitive, not just its terminal frame.
        self.failed |= failed | ~motion_ok
        self.reached |= layer_ok
        self.ready = self.reached & layer_ok & fingers_ready & grasp_ok & ~self.failed
        self.stable_steps = torch.where(self.ready, self.stable_steps + 1, 0)
        self.done = (self.stable_steps >= self.required_steps) & ~self.failed
        return self.done


class PoseWindow:
    """Anchor a candidate dwell window; reject cumulative drift, not velocity noise."""
    def __init__(self, n, device, pos_limit, rot_limit):
        self.pos = torch.zeros(n, 3, device=device)
        self.quat = torch.zeros(n, 4, device=device)
        self.pos_limit, self.rot_limit = pos_limit, rot_limit

    def update(self, pos, quat, begin):
        self.pos[begin] = pos[begin]
        self.quat[begin] = quat[begin]
        # q and -q represent the same rotation. Normalize to tolerate roundoff.
        q = torch.nn.functional.normalize(quat, dim=-1)
        anchor = torch.nn.functional.normalize(self.quat, dim=-1)
        angle = 2 * torch.acos((q * anchor).sum(-1).abs().clamp(max=1.0))
        return ((pos - self.pos).norm(dim=-1) <= self.pos_limit) & (angle <= self.rot_limit)


def limit_finger_command(previous,requested,rate_rad_s,dt,residual_range):
    """Bound normalized command increments; does not write physical joint states."""
    if rate_rad_s<0 or dt<=0 or residual_range<=0:
        raise ValueError("Invalid command slew parameters")
    if rate_rad_s==0:return requested
    maximum=rate_rad_s*dt/residual_range
    return previous+(requested-previous).clamp(-maximum,maximum)


def pad_progress_bonus(best_error,error,pad_force,whole_finger_force,weight):
    """Reward only new best angular progress made with instrumented pad contact.

    No reward for waiting under force or repeatedly revisiting the same angle.
    Truth is used for training reward only; completion conditions are unchanged.
    """
    if weight<0:raise ValueError('Pad progress weight must be nonnegative')
    improvement=(best_error-error.abs()).clamp(min=0,max=.1)
    fraction=(pad_force/whole_finger_force.clamp(min=.5)).clamp(0,1)
    fraction=torch.where(whole_finger_force>.5,fraction,0.)
    return torch.minimum(best_error,error.abs()),weight*improvement*fraction


def participant_finger_metrics(q, qd, ready, forces, hands, cols, links, participating):
    """Truth-only completion metrics for every finger used in this primitive.

    q/qd/ready: [env, hand, joint]; forces: [env, hand, link].
    hands/participating: [env, participant]; cols/links add joint/link axes.
    Release force is summed per whole finger, then maximized across participants:
    each finger retains its existing limit, rather than sharing a force budget.
    Participation must persist until the primitive ends, including withdrawal.
    """
    if participating.dtype != torch.bool or participating.shape != hands.shape:
        raise ValueError('Participation must be a boolean env-by-participant mask')
    if not participating.any(-1).all():
        raise ValueError('Every environment needs at least one participating finger')
    rows = torch.arange(q.shape[0], device=q.device)[:, None, None]
    index = (rows, hands[:, :, None], cols)
    error = (q[index] - ready[index]).abs().amax(-1)
    speed = qd[index].abs().amax(-1)
    selected = forces[torch.arange(q.shape[0], device=q.device)[:, None], hands]
    release = torch.where(links.bool(), selected, 0.).sum(-1)
    result = []
    for metric in (error, speed, release):
        # A participating NaN/Inf must never disappear through reduction.
        metric = torch.where(torch.isfinite(metric), metric, torch.inf)
        result.append(torch.where(participating, metric, 0.).amax(-1))
    return tuple(result)
