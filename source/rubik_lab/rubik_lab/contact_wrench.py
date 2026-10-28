"""Reduce PhysX filtered contact buffers to per-body-pair wrenches.

Forces and moments are in world axes, moments about the supplied sensor origin.
This uses each reported contact/friction point, preserving force couples that a
mean contact point approximation loses. No cube pose or layer labels are used.
Only configured contact filters are covered; this is not an all-contact wrench.
"""
import torch


def sum_contact_segments(values, counts, starts, check=True):
    """Sum disjoint packed contact segments; unused buffer entries may be NaN. check=False skips the buffer validation
    (each check is a GPU -> CPU sync: ~10x slower data collection)."""
    counts=counts.long();starts=starts.long()
    if counts.shape!=starts.shape or values.ndim!=2:
        raise ValueError('Expected matrix buffer and equally-shaped counts/starts')
    # Empty segments may have sentinel starts; they must not index the buffer.
    safe_start=torch.where(counts>0,starts,0)
    end=safe_start+counts
    if check and bool((counts<0).any() or (safe_start<0).any() or (end>len(values)).any()):
        raise ValueError('Contact segment outside available buffer')
    # Track invalid entries separately: unused NaNs are harmless, live ones fail.
    finite=torch.isfinite(values).all(-1)
    invalid_prefix=torch.cat([counts.new_zeros(1),(~finite).long().cumsum(0)])
    if check and bool((invalid_prefix[end]-invalid_prefix[safe_start]).any()):
        raise ValueError('Nonfinite active contact data')
    prefix=torch.cat([values.new_zeros(1,values.shape[-1]),torch.nan_to_num(values).cumsum(0)],0)
    return prefix[end]-prefix[safe_start]


def point_contact_wrench(forces,points,counts,starts,origins,check=True):
    """Return (...,6) for each pair, with origins broadcast over pair dimensions."""
    if forces.shape!=points.shape or forces.shape[-1]!=3:
        raise ValueError('Force and point buffers must have matching (K,3) shape')
    output_dtype=forces.dtype
    # Global world-frame moment sums can be large across distant environments.
    # Accumulate in float64 before shifting the moment origin, avoiding loss of
    # a small local wrench behind an unrelated large contact segment.
    forces=forces.double();points=points.double();origins=origins.double()
    force=sum_contact_segments(forces,counts,starts,check)
    moment_world=sum_contact_segments(torch.cross(points,forces,dim=-1),counts,starts,check)
    moment=moment_world-torch.cross(torch.broadcast_to(origins,force.shape),force,dim=-1)
    return torch.cat([force,moment],-1).to(output_dtype)
