"""Candidate support-contact geometry for diagnostic IK, not a force model.

Keeps each measured support pad near its initial signed convex/core overlap
while allowing tangential travel. The face footprint and normal guards prevent
using an unrelated edge or opposite-facing pad to satisfy overlap alone.
"""
import torch
from rubik_lab.diagnostics.convex_geometry import convex_box_overlap_torch


class SupportContact:
    BAND_M = 50e-6
    NORMAL_DOT = .95
    EDGE_INSET_M = .002
    FINGERS = ('thumb', 'middle')

    def __init__(self, fk, features, frames, pads, optimizer_reserve_m=0.):
        if not 0 <= optimizer_reserve_m < self.BAND_M:
            raise ValueError('Support optimization reserve must be in [0,50um)')
        self.optimizer_band_m = self.BAND_M - optimizer_reserve_m
        self.links = {f: fk.pad_links[f] for f in self.FINGERS}
        self.features = {f: features[self.links[f]] for f in self.FINGERS}
        sample = pads['thumb'][0]
        # Exact two-body USD core box, in the original cube frame.
        self.center = sample.new_tensor([0., 0., -.011667])
        self.half = sample.new_tensor([.035, .035, .0233335])
        self.rotation = torch.eye(3, dtype=sample.dtype, device=sample.device)
        self.initial_normals = {f: pads[f][1].detach().clone() for f in self.FINGERS}
        self.initial_overlaps = {}
        self.face_signs = {}
        for f in self.FINGERS:
            position, normal = pads[f]
            sign = 1. if float(position[0, 0]) >= 0 else -1.
            # This candidate is only defined for the measured opposing x faces.
            if float(position[0, 0])*sign < .025 or float(normal[0, 0])*sign > -.5:
                raise ValueError('Support pad is not on an inward-facing x side: '+f)
            self.face_signs[f] = sign
            self.initial_overlaps[f] = self._overlap(f, frames).detach().clone()
        initial, _ = self.measure(frames, pads)
        if not bool(torch.isfinite(initial).all() and (initial >= 0).all()):
            raise ValueError('Initial support state violates contact candidate domain')

    def _overlap(self, finger, frames):
        position, rotation = frames[self.links[finger]]
        return convex_box_overlap_torch(self.features[finger], rotation[0],
            position[0], self.rotation, self.center, self.half)

    def measure(self, frames, pads, optimization=False):
        band = self.optimizer_band_m if optimization else self.BAND_M
        constraints, details = [], {}
        for f in self.FINGERS:
            position, normal = pads[f]
            overlap = self._overlap(f, frames)
            drift = overlap - self.initial_overlaps[f]
            alignment = (normal*self.initial_normals[f]).sum()
            footprint = self.half[1:] - self.EDGE_INSET_M - (position[0, 1:]-self.center[1:]).abs()
            face = position[0, 0]*self.face_signs[f] - .025
            components = torch.cat([
                (1-(drift/band).square()).reshape(1),
                ((alignment-self.NORMAL_DOT)/(1-self.NORMAL_DOT)).reshape(1),
                footprint/.01, (face/.01).reshape(1)])
            constraints.append(components.amin())
            details[f] = dict(overlap=overlap, overlap_drift=drift,
                normal_alignment=alignment, footprint_margin=footprint,
                same_face_margin=face, components=components)
        return torch.stack(constraints), details

    @staticmethod
    def serialize(details):
        return {finger: {name: value.detach().cpu().tolist()
                         for name, value in fields.items()}
                for finger, fields in details.items()}
