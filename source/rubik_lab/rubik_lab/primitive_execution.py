"""Shared actor-state handling for physically held primitive starts."""
import torch


def apply_ready_hold(actions, ready, waiting, actors):
    """Hold physical ready commands without consuming a recurrent motion phase.

    Only the per-primitive actors are reset. No environment, sensory clock,
    object pose, or task-state reference is changed.
    """
    ids = waiting.nonzero().flatten()
    if ids.numel():
        for actor in actors:
            if hasattr(actor, "reset"):
                actor.reset(ids)
    return torch.where(waiting[:, None], ready, actions)


def command_for_filtered_target(target, previous, alpha, limit=1.0):
    """Bounded command whose existing EMA approaches a desired normalized target."""
    if not 0 < alpha <= 1 or limit <= 0:
        raise ValueError("Expected alpha in (0,1] and positive command limit")
    return ((target - (1-alpha)*previous)/alpha).clamp(-limit,limit)
