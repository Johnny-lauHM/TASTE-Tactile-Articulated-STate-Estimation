"""Sharpa Wave SE fingertip tactile emulation (on-board inference mode).

Per elastomer pad (5 per hand) the real Wave SE in on-board mode reports (sharpa.com/pages/wave, 2nd spec column):
    - a 60x60 tactile image at 30 fps (spatial resolution 2 mm)
    - 6-D force/torque Fx Fy Fz Mx My Mz, range 0-30 N, force resolution 0.05 N
    - ~20 ms latency
This module reproduces that interface in Isaac Lab:
    image : official Sharpa TacMap (penetration-depth map by ray casting from the elastomer surface map along its
            normals, sharpa-robotics/sharpa-tacmap), at resolution_step=4 -> 60x60, uint8 like upstream
    F/T   : legacy approximation from PhysX NORMAL contact forces only (friction omitted).
            F = net normal contact force on the pad; M = sum over the cube bodies of
            (mean contact point - pad origin) x contact force (point-force approximation per touched body).
            Both expressed in the pad (elastomer link) frame.
    timing: frames are captured every round(1/fps / physics_dt) physics steps (global sensor clock) and delivered
            after round(latency / physics_dt) steps; the policy reads the latest delivered frame (sample and hold).
    noise : Gaussian noise on F/T, clipping to the range, quantization to the resolution.

TacMap is geometric penetration, not calibrated soft-pad deformation. With rigid
collision contacts it can remain entirely zero under nonzero contact force.
Geometry version 2 fixes target mesh pose composition; earlier image data had
misplaced target meshes and must not be used as valid tactile-image supervision.

Unknowns to calibrate on the real hand (see MEMORY.md): F/T sign convention and moment reference point of the
real sensor, image intensity mapping of the on-board 60x60 output vs TacMap's deform_quantize.

`step()` must be called once per physics step (from DirectRLEnv._apply_action) and `reset(env_ids)` on resets.
"""

from __future__ import annotations

import math
import os

import torch

import isaaclab.utils.math as math_utils
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.utils import configclass

from .assets import ASSET_DIR, FINGERTIPS
from .tacmap import SharpaTacmap, SharpaTacmapCfg

MAP_DIR = os.path.join(ASSET_DIR, "tacmap_maps")


@configclass
class WaveSETactileCfg:
    geometry_version: int = 2  # TacMap target offsets composed with physics-body poses
    fps: float = 30.0
    latency: float = 0.02  # s
    force_range: float = 30.0  # N, per axis clip
    force_resolution: float = 0.05  # N
    force_noise_std: float = 0.02  # N
    moment_noise_std: float = 2e-4  # Nm
    moment_resolution: float = 1e-4  # Nm (not in the spec sheet; placeholder)
    image: bool = False  # TacMap 60x60 images (ray casting, expensive: enable for data collection, not for PPO)
    image_resolution_step: int = 4  # 240 / 4 = 60
    image_max_distance: float = 0.015
    # contact sensor template; {side} / {finger} are filled in. Filters = cube bodies [core, top] (labels, moments)
    contact_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/{side}_hand/{side}_{finger}_elastomer",
        history_length=1,
        track_contact_points=True,
        max_contact_data_count_per_prim=16,
        filter_prim_paths_expr=["/World/envs/env_.*/cube/core", "/World/envs/env_.*/cube/top"],
    )
    # TacMap ray-cast targets (collision geometry of the cube bodies, not the sticker visuals)
    image_targets: tuple = ("/World/envs/env_.*/cube/core/collision", "/World/envs/env_.*/cube/top/collision")


def _tacmap_cfg(cfg: WaveSETactileCfg, side: str, finger: str) -> SharpaTacmapCfg:
    thumb = finger == "thumb"
    return SharpaTacmapCfg(
        prim_path=f"/World/envs/env_.*/{side}_hand/{side}_{finger}_elastomer",
        mesh_prim_paths=[SharpaTacmapCfg.RaycastTargetCfg(prim_expr=p) for p in cfg.image_targets],
        update_period=0.0,  # lazy: rays are cast when .data is read, i.e. on WaveSETactile's capture clock
        pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=(0.5, 0.5)),  # required by the base class, unused
        offset=SharpaTacmapCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["distance_along_normal"],
        # same maps as the official env_cfg: 4F uses the *_origin maps, the thumb the plain ones
        points_npy=os.path.join(MAP_DIR, "tactileSensor_map_TH_point.npy" if thumb else "tactileSensor_map_4F_point_origin.npy"),
        normals_npy=os.path.join(MAP_DIR, "tactileSensor_map_TH_normal.npy" if thumb else "tactileSensor_map_4F_normal_origin.npy"),
        resolution_step=cfg.image_resolution_step,
        max_distance=cfg.image_max_distance,
        correction_scale=1e-3,
        debug_viz=False,
    )


class WaveSETactile:
    SIDES = ("left", "right")

    def __init__(self, cfg: WaveSETactileCfg, scene):
        """Create the sensors and register them in the scene (call inside _setup_scene, before cloning is fine)."""
        self.cfg = cfg
        self.contact = {}
        self.images = {}
        cc = cfg.contact_cfg
        for s in self.SIDES:
            self.contact[s] = [ContactSensor(cc.replace(prim_path=cc.prim_path.format(side=s, finger=f))) for f in FINGERTIPS]
            for f, sensor in zip(FINGERTIPS, self.contact[s]):
                scene.sensors[f"{s}_{f}_contact"] = sensor
            if cfg.image:
                self.images[s] = [SharpaTacmap(_tacmap_cfg(cfg, s, f)) for f in FINGERTIPS]
                for f, sensor in zip(FINGERTIPS, self.images[s]):
                    scene.sensors[f"{s}_{f}_tacmap"] = sensor

    def initialize(self, hands: dict, num_envs: int, physics_dt: float, device):
        """Call after the simulation started (env __init__ after super().__init__)."""
        self.hands, self.n, self.dev = hands, num_envs, device
        self.physics_dt = physics_dt
        self.pad_body_ids = {
            s: hands[s].find_bodies([f"{s}_{f}_elastomer" for f in FINGERTIPS], preserve_order=True)[0] for s in self.SIDES
        }
        self.capture_every = max(1, round(1.0 / (self.cfg.fps * physics_dt)))
        self.latency_steps = max(0, round(self.cfg.latency / physics_dt))
        assert self.latency_steps < self.capture_every, "latency longer than the frame period is not supported"
        self.step_count = 0
        from rubik_lab.frame_timing import SensorFrameClock
        self._frame_clock = SensorFrameClock(num_envs,device,physics_dt)
        npad = len(FINGERTIPS)
        side = 240 // self.cfg.image_resolution_step
        z = lambda *shape, dtype=torch.float32: torch.zeros(num_envs, 2, npad, *shape, device=device, dtype=dtype)  # noqa: E731
        # [pending capture, delivered frame]
        self._ft = [z(6), z(6)]
        self._img = [z(side, side, dtype=torch.uint8), z(side, side, dtype=torch.uint8)] if self.cfg.image else None
        self._pending_step = -(10**9)
        self._has_pending = False

    # ------------------------------------------------------------------ raw
    def raw_ft(self) -> torch.Tensor:
        """noise-free 6-D F/T (num_envs, 2, 5, 6) in the pad frames, at the current physics step."""
        out = []
        for s in self.SIDES:
            body = self.hands[s].data
            pad_p = body.body_pos_w[:, self.pad_body_ids[s]]  # (N, 5, 3)
            pad_q = body.body_quat_w[:, self.pad_body_ids[s]]
            f_net = torch.cat([c.data.net_forces_w for c in self.contact[s]], 1)  # (N, 5, 3)
            f_b = torch.cat([c.data.force_matrix_w for c in self.contact[s]], 1)  # (N, 5, 2, 3)
            p_b = torch.cat([c.data.contact_pos_w for c in self.contact[s]], 1)  # (N, 5, 2, 3), NaN if no contact
            r = torch.nan_to_num(p_b - pad_p.unsqueeze(2), nan=0.0)
            m = torch.cross(r, f_b, dim=-1).sum(2)  # (N, 5, 3)
            q = pad_q.reshape(-1, 4)
            f_pad = math_utils.quat_apply_inverse(q, f_net.reshape(-1, 3)).view_as(f_net)
            m_pad = math_utils.quat_apply_inverse(q, m.reshape(-1, 3)).view_as(m)
            out.append(torch.cat([f_pad, m_pad], -1))
        return torch.stack(out, 1)

    def raw_cube_wrench(self, check: bool = True) -> torch.Tensor:
        """Diagnostic instantaneous cube-contact wrench, including friction.

        Requires track_friction_forces=True. Per-contact force moments are about
        the pad origin, expressed in pad axes. Excludes hand/self contacts and
        unreported torsional couples; no noise, quantization, or delivery delay.
        This is NOT substituted into the legacy actor input by this method.
        """
        if not self.cfg.contact_cfg.track_friction_forces:
            raise ValueError("Enable friction tracking before scene construction")
        from .contact_wrench import point_contact_wrench
        sides=[]
        for side in self.SIDES:
            hand=self.hands[side].data
            pads=[]
            for finger,sensor in enumerate(self.contact[side]):
                body=self.pad_body_ids[side][finger]
                origin=hand.body_pos_w[:,body,None,:]
                magnitude,points,normals,_,counts,starts=sensor.contact_physx_view.get_contact_data(dt=self.physics_dt)
                normal=point_contact_wrench(magnitude.reshape(-1,1)*normals,points,counts,starts,origin,check)
                forces,points,counts,starts=sensor.contact_physx_view.get_friction_data(dt=self.physics_dt)
                friction=point_contact_wrench(forces,points,counts,starts,origin,check)
                wrench=(normal+friction).sum(-2)
                quat=hand.body_quat_w[:,body]
                pads.append(torch.cat([math_utils.quat_apply_inverse(quat,wrench[:,:3]),
                                       math_utils.quat_apply_inverse(quat,wrench[:,3:])],-1))
            sides.append(torch.stack(pads,1))
        return torch.stack(sides,1)

    def raw_image(self) -> torch.Tensor | None:
        """latest TacMap images (num_envs, 2, 5, 60, 60) uint8, as computed by the sensors (no latency)."""
        if not self.cfg.image:
            return None
        imgs = []
        for s in self.SIDES:
            per = [t.data.output["distance_along_normal"].view(self.n, -1) for t in self.images[s]]
            side = int(round(math.sqrt(per[0].shape[-1])))
            imgs.append(torch.stack([p.view(self.n, side, side) for p in per], 1))
        return torch.stack(imgs, 1)

    def _process_ft(self, ft: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        f, m = ft[..., :3], ft[..., 3:]
        f = f + c.force_noise_std * torch.randn_like(f)
        m = m + c.moment_noise_std * torch.randn_like(m)
        f = (f.clamp(-c.force_range, c.force_range) / c.force_resolution).round() * c.force_resolution
        m = (m / c.moment_resolution).round() * c.moment_resolution
        return torch.cat([f, m], -1)

    # --------------------------------------------------------------- timing
    def step(self):
        """once per physics step: capture on the sensor clock, deliver after the latency (sample and hold)."""
        if self.step_count % self.capture_every == 0:
            self._ft[0].copy_(self._process_ft(self.raw_ft()))
            if self._img is not None:
                self._img[0].copy_(self.raw_image())  # lazy sensors: the rays are cast here, i.e. at `fps`
            self._pending_step, self._has_pending = self.step_count, True
            self._frame_clock.capture(self.step_count)
        if self._has_pending and self.step_count >= self._pending_step + self.latency_steps:
            self._ft[1].copy_(self._ft[0])
            if self._img is not None:
                self._img[1].copy_(self._img[0])
            self._has_pending = False
            self._frame_clock.deliver()
        self.step_count += 1

    def reset(self, env_ids):
        self._frame_clock.reset(env_ids)
        for buf in self._ft:
            buf[env_ids] = 0.0
        if self._img is not None:
            for buf in self._img:
                buf[env_ids] = 0

    # -------------------------------------------------------------- outputs
    @property
    def timing(self):
        """[age in seconds, valid flag] of the delivered frame at the current boundary.

        Capture occurs before a physics step; post-step reads include that elapsed step.
        A zeroed reset buffer is invalid until a post-reset capture is delivered.
        """
        return self._frame_clock.observe(self.step_count)

    @property
    def ft(self) -> torch.Tensor:
        """Delivered legacy normal-only F/T approximation, shape (N, 2, 5, 6)."""
        return self._ft[1]

    @property
    def image(self) -> torch.Tensor | None:
        """delivered 60x60 images (num_envs, 2, 5, 60, 60) uint8, or None if images are disabled."""
        return None if self._img is None else self._img[1]

    def layer_forces(self) -> torch.Tensor:
        """ground-truth NORMAL contact force norms per cube body (num_envs, 2, 5, 2 [core, top]) - labels only."""
        return torch.stack(
            [torch.linalg.norm(torch.cat([c.data.force_matrix_w for c in self.contact[s]], 1), dim=-1) for s in self.SIDES], 1
        )
