from __future__ import annotations

import math

import os
from collections.abc import Sequence

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from rubik_lab.assets import ASSET_DIR, FINGER_JOINTS, FINGERTIPS
from rubik_lab.assets import CONTACT_MODELS, sharpa_hand_cfg
from rubik_lab.cube_contacts import CubeContacts
from rubik_lab.ee_pose_controller import FloatBaseEEController
from rubik_lab.wave_se_tactile import WaveSETactile
from rubik_lab.moves import ALL_MOVES, FACE_NORMALS, FACE_QUATS, MOVE_INDEX, move_angle, move_face, primitive_spec

from .env_cfg import BimanualTwistEnvCfg, home_grip_pose

SIDES = ("left", "right")
PROJECT_DIR = os.path.dirname(ASSET_DIR)


class BimanualTwistEnv(DirectRLEnv):
    """One move primitive per episode from the bimanual home grip.

    extras (every step), for training a completion detector from tactile + proprioception:
        extras["move_done"]          (N,) bool   layer at the target angle and settled
        extras["primitive_done"]     (N,) bool   move done + hands back at the grasp pose and touching (ends the episode)
        extras["twist_error"]        (N,)        signed target - current layer angle [rad]
        extras["tactile_layer_force"] (N, 2 hands, 5 tips, 2 bodies [core, top]) NORMAL contact force norms [N] (ground truth)
        extras["tactile_ft"]         (N, 2, 5, 6) legacy normal-only F/T approximation, delivered at sensor rate/latency
        extras["tactile_image"]      (N, 2, 5, 60, 60) uint8 Wave SE TacMap images (only if cfg.tactile.image)
    """

    cfg: BimanualTwistEnvCfg

    def __init__(self, cfg: BimanualTwistEnvCfg, render_mode: str | None = None, **kwargs):
        # "flick": finger-trick RL -- the grip is frozen, the policy drives only the trick finger's 4 joints (rubik_lab.flick.FLICKS);
        # internally expanded to a residual action
        self.flick_mode = cfg.ee_action_mode == "flick"
        per_hand = {"fingers": 22, "residual": 28, "relative": 28, "absolute": 29, "flick": 28}[cfg.ee_action_mode]
        cfg.action_space = 4 if self.flick_mode else 2 * per_hand
        # per hand: fingers q, qd (44) + ee pose (7) + ee - home (6) + tip pos (15) + Wave SE F/T 5x6 (30) = 102;
        # cube 16; move one-hot + target 19; layer-done latch 1; wrist mask 2; nominal roll phase 1
        cfg.observation_space = 2 * 102 + 16 + len(ALL_MOVES) + 1 + 1 + 2 + 1 + cfg.action_space
        # contact model (hands are rebuilt so the choice also applies to cfgs created before it was set)
        cm = CONTACT_MODELS[cfg.contact_model]
        for side in SIDES:
            setattr(cfg, f"{side}_hand_cfg", sharpa_hand_cfg(side, contact_model=cfg.contact_model, wrist_stiffness=cfg.wrist_stiffness))
        cfg.cube_cfg.spawn.collision_props.contact_offset = cm["obj_contact_offset"]
        cfg.cube_cfg.spawn.collision_props.rest_offset = cm["obj_rest_offset"]
        self.home = {s: home_grip_pose(s, cfg.grip_tilt_deg, cfg.grip_palm_gap, cfg.grip_cube_along_fingers) for s in SIDES}
        for side in SIDES:
            hand_cfg = getattr(cfg, f"{side}_hand_cfg")
            hand_cfg.init_state.pos, hand_cfg.init_state.rot = self.home[side]
        if cfg.gravity_curriculum:
            cfg.sim.gravity = (0.0, 0.0, -cfg.gravity_start)
        super().__init__(cfg, render_mode, **kwargs)
        self.per_hand = per_hand
        self.gravity = abs(cfg.sim.gravity[2])
        self.drop_rate_ema = 1.0
        n, dev = self.num_envs, self.device

        self.ctrl, self.finger_ids, self.tip_ids = {}, {}, {}
        self.finger_lower, self.finger_upper = {}, {}
        for side, hand in self.hands.items():
            self.ctrl[side] = FloatBaseEEController(hand, side, self.cfg.ee_body)
            self.finger_ids[side] = hand.find_joints([f"{side}_{j}" for j in FINGER_JOINTS], preserve_order=True)[0]
            self.tip_ids[side] = hand.find_bodies([f"{side}_{f}_fingertip" for f in FINGERTIPS], preserve_order=True)[0]
            lim = hand.data.soft_joint_pos_limits[:, self.finger_ids[side]]
            mid, half = lim.mean(-1), 0.5 * (lim[..., 1] - lim[..., 0]) * self.cfg.dof_limits_scale
            self.finger_lower[side], self.finger_upper[side] = mid - half, mid + half

        self.tactile.initialize(self.hands, n, self.physics_dt, dev)
        self.core_id = self.cube.find_bodies("core")[0][0]
        self.top_id = self.cube.find_bodies("top")[0][0]
        self.twist_id = self.cube.find_joints("twist_joint")[0][0]

        # move table (indexed by ALL_MOVES order)
        self.move_pool = torch.tensor([MOVE_INDEX[m] for m in self.cfg.moves], device=dev)
        self.move_face_quat = torch.tensor([FACE_QUATS[move_face(m)] for m in ALL_MOVES], device=dev)
        # per-move execution spec (rubik_lab.moves.primitive_spec); unsupported moves (D/F/B/X2) get no hands
        specs = [primitive_spec(m) for m in ALL_MOVES]
        self.move_hands = torch.tensor([sp["hands"] if sp else (0, 0) for sp in specs], device=dev, dtype=torch.float32)
        self.move_axis = torch.tensor([sp["axis"] if sp else FACE_NORMALS[move_face(m)] for m, sp in zip(ALL_MOVES, specs)], device=dev)
        self.move_roll = torch.tensor([sp["angle"] if sp else 0.0 for sp in specs], device=dev)  # signed, about move_axis
        self.move_target = torch.tensor([sp["layer"] if sp else move_angle(m) for m, sp in zip(ALL_MOVES, specs)], device=dev)
        self.move_shift = torch.tensor([sp["shift"] if sp else ((0, 0, 0), (0, 0, 0)) for sp in specs], device=dev, dtype=torch.float32)
        self.move_is_rot = torch.tensor([m[0] == "x" for m in ALL_MOVES], device=dev)
        self.move_face_id = torch.tensor(["RLUDFB".index("R" if m[0] == "x" else m[0]) for m in ALL_MOVES], device=dev)
        # human finger tricks (rubik_lab.flick.FLICKS): the trick hand flicks with one finger (+ a small wrist assist),
        # no wrist roll, no grip shift; the layer target is the move angle
        from rubik_lab.flick import FLICKS

        self.move_is_flick = torch.zeros(len(ALL_MOVES), dtype=torch.bool, device=dev)
        for m in self.cfg.flick_moves:
            mi = MOVE_INDEX[m]
            hand = FLICKS[m][0]
            self.move_hands[mi] = torch.tensor([1.0, 0.0] if hand == "left" else [0.0, 1.0], device=dev)
            self.move_shift[mi] = 0.0
            self.move_roll[mi] = 0.0
            self.move_target[mi] = move_angle(m)
            self.move_is_flick[mi] = True
        one = self.move_hands
        self.move_twist_hand = torch.where((one[:, 1] > 0) & (one[:, 0] == 0), 1, torch.where((one[:, 0] > 0) & (one[:, 1] == 0), 0, -1))
        self.move_uses_wrist = one.sum(-1) > 0

        # persistent targets (env frame for EE poses)
        self.ee_pos_tgt = {s: torch.tensor(self.home[s][0], device=dev).repeat(n, 1) for s in SIDES}
        self.ee_quat_tgt = {s: torch.tensor(self.home[s][1], device=dev).repeat(n, 1) for s in SIDES}
        self.finger_tgt = {s: torch.zeros(n, len(FINGER_JOINTS), device=dev) for s in SIDES}
        self._base_tgt = {s: torch.zeros(n, 6, device=dev) for s in SIDES}
        self.ws_lo = torch.tensor(self.cfg.ee_workspace[0], device=dev)
        self.ws_hi = torch.tensor(self.cfg.ee_workspace[1], device=dev)

        self.actions = torch.zeros(n, self.cfg.action_space, device=dev)
        self.prev_actions = torch.zeros_like(self.actions)
        self.move_id = torch.zeros(n, dtype=torch.long, device=dev)
        self.cube_center = torch.zeros(n, 3, device=dev)  # env frame, set on reset
        self.cube_quat0 = torch.zeros(n, 4, device=dev)  # logical cube orientation at reset
        self.prev_twist_err = torch.zeros(n, device=dev)
        self.move_done = torch.zeros(n, dtype=torch.bool, device=dev)
        self.layer_done_latch = torch.zeros(n, dtype=torch.bool, device=dev)  # layer reached the target once
        self.primitive_done = torch.zeros(n, dtype=torch.bool, device=dev)
        self.wrist_mask = torch.zeros(n, 2, device=dev)  # 1 = this hand's wrist may move in this episode
        self.ee_grasp_tgt = {s: torch.zeros(n, 3, device=dev) for s in SIDES}  # EE target of the grasp (clamp center)
        self.ee_grasp_quat = {s: torch.zeros(n, 4, device=dev) for s in SIDES}
        self.finger_grasp = {s: torch.zeros(n, len(FINGER_JOINTS), device=dev) for s in SIDES}  # finger targets of the grasp
        self.action_smooth = torch.zeros(n, 2 * per_hand, device=dev)
        if self.flick_mode:
            from rubik_lab.flick import FLICKS

            # per move: trick hand (0/1) and the 4 joint columns of the trick finger; -1 for non-flick moves
            self.flick_hand = torch.full((len(ALL_MOVES),), -1, dtype=torch.long, device=dev)
            self.flick_cols = torch.zeros(len(ALL_MOVES), 4, dtype=torch.long, device=dev)
            self.flick_pad = torch.zeros(len(ALL_MOVES), dtype=torch.long, device=dev)  # tactile pad index of the trick finger
            from rubik_lab.cube_contacts import LINKS as _LINKS

            # all links of the trick finger (the policies push with the distal phalanx body, not only the pad)
            self.flick_links = torch.zeros(len(ALL_MOVES), len(_LINKS), device=dev)
            self.flick_finger_mask = torch.zeros(len(ALL_MOVES), 22, device=dev)  # the trick finger's flexion joints
            finger_of = [j.split("_")[0] for j in FINGER_JOINTS]
            flexj = [("FE" in j or j.endswith(("PIP", "DIP", "IP"))) for j in FINGER_JOINTS]
            self.pinky_flex = torch.tensor([finger_of[i] == "pinky" and flexj[i] for i in range(22)], device=dev).float()
            self.finger_flex = {f: torch.tensor([finger_of[i] == f and flexj[i] for i in range(22)], device=dev).float() for f in FINGERTIPS}
            for m, (side, finger, *_) in FLICKS.items():
                mi = MOVE_INDEX[m]
                self.flick_hand[mi] = 0 if side == "left" else 1
                self.flick_cols[mi] = torch.tensor([i for i, j in enumerate(FINGER_JOINTS) if j.startswith(finger)][:4], device=dev)
                self.flick_finger_mask[mi] = self.finger_flex[finger]
                self.flick_pad[mi] = FINGERTIPS.index(finger)
                self.flick_links[mi] = torch.tensor([ln.startswith(finger) for ln in _LINKS], device=dev).float()
            self.flick_lift = torch.zeros(n, 2, 22, device=dev)  # fingers lifted off the moving layer (joint units)
            self.flick_lift_valid = torch.zeros(n, dtype=torch.bool, device=dev)
        self.roll_nominal = torch.zeros(n, device=dev)
        self.roll_back = torch.zeros(n, device=dev)
        self.roll_frac_at_latch = torch.zeros(n, device=dev)
        self.first_reach = torch.zeros(n, dtype=torch.bool, device=dev)
        self.roll_axis = torch.zeros(n, 3, device=dev)  # face normal of this episode's move (env frame)
        self.cube_quat_goal = torch.zeros(n, 4, device=dev)  # logical orientation the move should end in
        self.peak_rot = torch.zeros(n, device=dev)  # per-episode peak cube rotation / displacement (calm-success bonus)
        self.peak_pos = torch.zeros(n, device=dev)
        self.ee_home_pos = {s: torch.zeros(n, 3, device=dev) for s in SIDES}  # actual EE pose of the grasp
        self.ee_home_quat = {s: torch.zeros(n, 4, device=dev) for s in SIDES}
        self.reset_face_match = torch.zeros(n, dtype=torch.bool, device=dev)  # diagnostics: cached grasp had the same twist axis

        self.forced_grasp_idx = None
        self.episode_grasp_idx = torch.full((n,), -1, dtype=torch.long, device=dev)  # cache index used at the last reset
        self.ep_hands = torch.zeros(n, 2, device=dev)  # rotating hands of this episode
        self.ep_shift = torch.zeros(n, 2, 3, device=dev)  # grip shift per hand (logical)
        self.ep_edge = torch.ones(n, dtype=torch.long, device=dev)  # edge (twisting) hand of the grip state
        self.regrip_pos = {s: torch.zeros(n, 3, device=dev) for s in SIDES}  # EE target of the grasp to end in
        self.regrip_quat = {s: torch.zeros(n, 4, device=dev) for s in SIDES}
        self.regrip_fingers = {s: torch.zeros(n, len(FINGER_JOINTS), device=dev) for s in SIDES}
        self._regrip_act = None
        self.forced_move_id = None
        self.broken_count = 0
        self.nonfinite_obs_count = 0
        self._warned_nonfinite = False
        self.record = False  # set True to fill self.snapshot every step (scripts/collect_tactile_data.py)
        self.snapshot = {}
        self.grasp_cache = None
        path = os.path.join(PROJECT_DIR, self.cfg.grasp_cache_path) if self.cfg.grasp_cache_path else None
        if path and os.path.exists(path):
            self.grasp_cache = {k: torch.tensor(v, device=dev, dtype=torch.float32) for k, v in np.load(path).items()}
            print(f"[rubik_lab] grasp cache: {len(self.grasp_cache['cube_pos'])} grasps from {path}")
            # (6 faces, N grasps) sampling weights: grasps whose internal axis matches the face, else all grasps
            n_g = len(self.grasp_cache["cube_pos"])
            if "move_id" in self.grasp_cache:
                g_face = self.move_face_id[self.grasp_cache["move_id"].long()]
                mask = (g_face.unsqueeze(0) == torch.arange(6, device=dev).unsqueeze(1)).float()
            else:
                mask = torch.zeros(6, n_g, device=dev)
            if "cells" in self.grasp_cache:
                # contact compatibility with the move's layer: the own-side hand (R/L) touches only that layer, the
                # other hand never; for finger moves (U/D/F/B) no hand touches the layer at the start
                from rubik_lab.cube_contacts import face_layer_mask

                cells = self.grasp_cache["cells"] > 0.5  # (N, 2, 27)
                for fi, face in enumerate("RLUDFB"):
                    layer = face_layer_mask(face, dev)
                    inside = (cells & layer).any(-1)  # (N, 2)
                    outside = (cells & ~layer).any(-1)
                    hand = {"R": 1, "L": 0}.get(face)
                    if any(m[0] == face for m in self.cfg.flick_moves):
                        compat = torch.ones_like(inside[:, 0])  # finger tricks: the teacher lifts fingers on the layer
                    elif hand is None:
                        compat = ~inside.any(-1)
                    else:
                        compat = ~outside[:, hand] & ~inside[:, 1 - hand]
                    both = mask[fi] * compat.float()
                    mask[fi] = both if both.sum() > 0 else compat.float()  # prefer axis match + compatible
                n_compat = {f: int((mask[i] > 0).sum()) for i, f in enumerate("RLUDFB")}
                print(f"[rubik_lab] grasps usable per move face: {n_compat}")
            if "flick_reach" in self.grasp_cache:  # offline planner reach [deg] per (grasp, move) from scripts/flick_reach.py
                reach = self.grasp_cache["flick_reach"]  # (N, M), -1 = not computed
                for fi, face in enumerate("RLUDFB"):
                    ms = [MOVE_INDEX[m] for m in self.cfg.flick_moves if m[0] == face]
                    if ms:
                        ok = (reach[:, ms] >= self.cfg.flick_reach_min).any(-1).float()
                        if (mask[fi] * ok).sum() > 0:
                            mask[fi] = mask[fi] * ok
                            print(f"[rubik_lab] {face}: {int((mask[fi] > 0).sum())} grasps with flick reach >= {self.cfg.flick_reach_min} deg")
            mask[mask.sum(-1) == 0] = 1.0
            self._grasp_face_mask = mask
            # per-move mask: role-based grip states (A = left center + right edge, B = mirror) and x partners
            mm = mask[self.move_face_id].clone()  # (M, N)
            if "state" in self.grasp_cache:
                st = self.grasp_cache["state"]
                has_partner = self.grasp_cache["partner"] >= 0
                for mi, m in enumerate(ALL_MOVES):
                    need = torch.ones(n_g, dtype=torch.bool, device=dev)
                    if m[0] == "R":
                        need = st == 0  # left center holds, right (edge) twists
                    elif m[0] == "L":
                        need = st == 1
                    elif m[0] == "x":
                        need = has_partner
                        mm[mi] = 1.0  # any internal axis / contacts (whole-cube rotation)
                    row = mm[mi] * need.float()
                    mm[mi] = row if row.sum() > 0 else need.float()
                print("[rubik_lab] grasps usable per primitive: " + ", ".join(f"{m}={int((mm[MOVE_INDEX[m]] > 0).sum())}" for m in ("R", "L", "U", "x")))
            if "teach_rate" in self.grasp_cache and self.cfg.teach_rate_min > 0:
                # teacher-curated: per primitive only grasps the scripted teacher solved often enough
                from rubik_lab.moves import PRIMITIVES

                tr = self.grasp_cache["teach_rate"]  # (N, len(PRIMITIVES)), NaN = not evaluated
                for pi, m in enumerate(PRIMITIVES):
                    good = torch.nan_to_num(tr[:, pi], nan=-1.0) >= self.cfg.teach_rate_min
                    row = mm[MOVE_INDEX[m]] * good.float()
                    if row.sum() > 0:
                        mm[MOVE_INDEX[m]] = row
                print("[rubik_lab] teacher-curated grasps per primitive: " + ", ".join(
                    f"{m}={int((mm[MOVE_INDEX[m]] > 0).sum())}" for m in PRIMITIVES))
            self._grasp_move_mask = mm
        elif path:
            print(f"[rubik_lab] no grasp cache at {path}: starting from open hands (run scripts/gen_grasp_cache.py)")

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self.hands = {"left": Articulation(self.cfg.left_hand_cfg), "right": Articulation(self.cfg.right_hand_cfg)}
        self.cube = Articulation(self.cfg.cube_cfg)
        self.tactile = WaveSETactile(self.cfg.tactile, self.scene)
        self.cube_contacts = CubeContacts(self.scene) if self.cfg.link_contacts else None
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        for side in SIDES:
            self.scene.articulations[f"{side}_hand"] = self.hands[side]
        self.scene.articulations["cube"] = self.cube
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ---------------------------------------------------------------- actions
    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions[:] = self.actions
        self.actions = actions.clone().clamp(-self.cfg.clip_actions, self.cfg.clip_actions)
        self._update_roll_nominal()
        full = self._expand_flick_action(self.actions) if self.flick_mode else self.actions
        self.action_smooth = self.cfg.action_ema * full + (1.0 - self.cfg.action_ema) * self.action_smooth
        origins = self.scene.env_origins
        for i, side in enumerate(SIDES):
            a = full[:, i * self.per_hand : (i + 1) * self.per_hand]
            lo, hi = self.finger_lower[side], self.finger_upper[side]
            if self.cfg.ee_action_mode in ("residual", "flick"):
                a = self.action_smooth[:, i * self.per_hand : (i + 1) * self.per_hand]
                w = a[:, :6] * self.wrist_mask[:, i : i + 1]  # wrist frozen unless this move is this hand's roll
                if self.cfg.wrist_prior:  # nominal roll about the face axis (action units), policy residual on top
                    w = w.clone()
                    w[:, 3:6] += self.wrist_mask[:, i : i + 1] * self.roll_axis * (self.roll_nominal / self.cfg.residual_rot_range).unsqueeze(-1)
                dq = math_utils.quat_from_angle_axis(torch.linalg.norm(w[:, 3:6], dim=-1) * self.cfg.residual_rot_range, w[:, 3:6])
                c = self.cube_center
                self.ee_pos_tgt[side] = c + math_utils.quat_apply(dq, self.ee_grasp_tgt[side] - c) + w[:, :3] * self.cfg.residual_pos_range
                self.ee_quat_tgt[side] = math_utils.quat_mul(dq, self.ee_grasp_quat[side])
                self.finger_tgt[side] = self.finger_grasp[side] + a[:, 6:] * self.cfg.residual_finger_range
            elif self.cfg.ee_action_mode == "fingers":  # wrists hold the grasp pose
                self.finger_tgt[side] = self.finger_tgt[side] + a * self.cfg.finger_scale
            elif self.cfg.ee_action_mode == "relative":
                w = a[:, :6] * self.wrist_mask[:, i : i + 1]  # wrist frozen unless this move is this hand's roll
                self.ee_pos_tgt[side] += w[:, :3] * self.cfg.ee_pos_scale
                lim = self.cfg.ee_pos_limit
                self.ee_pos_tgt[side] = torch.max(torch.min(self.ee_pos_tgt[side], self.ee_grasp_tgt[side] + lim), self.ee_grasp_tgt[side] - lim)
                dq = math_utils.quat_from_angle_axis(
                    torch.linalg.norm(w[:, 3:6], dim=-1) * self.cfg.ee_rot_scale, w[:, 3:6]
                )  # world-frame delta rotation; quat_from_angle_axis normalizes the axis
                self.ee_quat_tgt[side] = math_utils.normalize(math_utils.quat_mul(dq, self.ee_quat_tgt[side]))
                self.finger_tgt[side] = self.finger_tgt[side] + a[:, 6:] * self.cfg.finger_scale
            else:
                # absolute actions are not clipped to [-1, 1] for the pose part
                a = actions[:, i * self.per_hand : (i + 1) * self.per_hand]
                self.ee_pos_tgt[side] = a[:, :3].clone()
                self.ee_quat_tgt[side] = math_utils.normalize(a[:, 3:7])
                self.finger_tgt[side] = lo + 0.5 * (a[:, 7:].clamp(-1, 1) + 1.0) * (hi - lo)
            self.ee_pos_tgt[side] = torch.max(torch.min(self.ee_pos_tgt[side], self.ws_hi), self.ws_lo)
            self.finger_tgt[side] = torch.max(torch.min(self.finger_tgt[side], hi), lo)
            self._base_tgt[side] = self.ctrl[side].joint_targets(self.ee_pos_tgt[side] + origins, self.ee_quat_tgt[side])

    def _expand_flick_action(self, pol: torch.Tensor) -> torch.Tensor:
        """(N, 4) trick-finger action -> (N, 56) residual action: wrists frozen, grip fingers held, the other hand's
        fingers that touch the moving layer and the trick hand's pinky lifted (as the scripted teacher does)."""
        n, P = self.num_envs, self.per_hand
        full = torch.zeros(n, 2 * P, device=self.device)
        mid = self.move_id
        h = self.flick_hand[mid].clamp(min=0)
        cols = self.flick_cols[mid]  # (N, 4)
        rows = torch.arange(n, device=self.device)
        full[rows.unsqueeze(-1), (h * P + 6).unsqueeze(-1) + cols] = pol[:, :4]
        # lift mask from the contacts one step after the reset (sensor data of the reset step is stale)
        need = ~self.flick_lift_valid & (self.episode_length_buf >= 1)
        if need.any():
            touch = self.cube_contacts.forces()[:, 1] > 0.05  # (N, 2, 27 links)
            from rubik_lab.cube_contacts import LINKS

            lift = torch.zeros(n, 2, 22, device=self.device)
            for hh in range(2):
                for li, ln in enumerate(LINKS):
                    f = ln.split("_")[0]
                    if f not in self.finger_flex or f in ("thumb", "middle"):
                        continue
                    own_trick = (h == hh) & (self.flick_finger_mask[mid] * self.finger_flex[f]).sum(-1).bool()
                    lift[:, hh] += (touch[:, hh, li] & ~own_trick).float().unsqueeze(-1) * self.finger_flex[f]
            lift[rows, h] += self.pinky_flex
            self.flick_lift[need] = lift[need].clamp(max=1.0)
            self.flick_lift_valid[need] = True
        lift_res = self.cfg.flick_lift * self.flick_lift / self.cfg.residual_finger_range  # (N, 2, 22) in action units
        full[:, 6:P] -= lift_res[:, 0]
        full[:, P + 6 :] -= lift_res[:, 1]
        # the holding thumbs/middles squeeze a bit harder than the cached grasp (friction headroom against the push)
        hold = (self.finger_flex["thumb"] + self.finger_flex["middle"]).clamp(max=1.0) * (self.cfg.flick_hold_press / self.cfg.residual_finger_range)
        full[:, 6:P] += hold
        full[:, P + 6 :] += hold
        return full.clamp(-1.0, 1.0)

    def _apply_action(self):
        for side, hand in self.hands.items():
            hand.set_joint_position_target(self._base_tgt[side], joint_ids=self.ctrl[side].joint_ids)
            hand.set_joint_position_target(self.finger_tgt[side], joint_ids=self.finger_ids[side])
        self.tactile.step()  # Wave SE sensor clock runs at physics rate
        # passive cube detent: stable at multiples of 90 deg
        twist = self.cube.data.joint_pos[:, self.twist_id : self.twist_id + 1]
        omega = self.cube.data.joint_vel[:, self.twist_id : self.twist_id + 1]
        tau = -self.cfg.detent_torque * torch.sin(4.0 * twist) - self.cfg.twist_friction_torque * torch.tanh(omega / 0.05)
        if self.cfg.notch_torque > 0.0:
            # alignment notch of a real cube: a short, stiff spring only within +-notch_width_deg of a 90 deg multiple
            # (holds the turned layer while the hand lets go and regrips; it does NOT pull a 45 deg layer anywhere)
            w = math.radians(self.cfg.notch_width_deg)
            dev_ = math_utils.wrap_to_pi(4.0 * twist) / 4.0  # signed deviation from the nearest multiple of 90 deg
            tau = tau - self.cfg.notch_torque * (dev_ / w) * (dev_.abs() < w).float()
        self.cube.set_joint_effort_target(tau, joint_ids=[self.twist_id])

    # ----------------------------------------------------------- observations
    def _cube_state(self):
        d = self.cube.data
        core_pos = d.body_pos_w[:, self.core_id] - self.scene.env_origins
        # logical cube orientation: undo the per-move rotation of the internal twist axis
        logical_quat = math_utils.quat_mul(d.body_quat_w[:, self.core_id], math_utils.quat_inv(self.move_face_quat[self.move_id]))
        twist = d.joint_pos[:, self.twist_id]
        twist_vel = d.joint_vel[:, self.twist_id]
        return core_pos, logical_quat, twist, twist_vel

    def hand_cells(self, threshold=0.05):
        """(N, 2 hands, 27) bool: logical cubie cells touched by each hand (any link)."""
        core_pos_w = self.cube.data.body_pos_w[:, self.core_id]
        return self.cube_contacts.cells(core_pos_w, self._cube_state()[1], self.cfg.cube_size, threshold)

    def _tips(self, side):
        return self.hands[side].data.body_pos_w[:, self.tip_ids[side]] - self.scene.env_origins.unsqueeze(1)

    def _get_observations(self) -> dict:
        core_pos, cube_quat, twist, twist_vel = self._cube_state()
        obs = []
        for i, side in enumerate(SIDES):
            hand = self.hands[side]
            q = hand.data.joint_pos[:, self.finger_ids[side]]
            lo, hi = self.finger_lower[side], self.finger_upper[side]
            ee_p, ee_q = self.ctrl[side].ee_pose_w()
            dp_home, dr_home = self._ee_home_error(side)
            obs += [
                2.0 * (q - lo) / (hi - lo) - 1.0,
                0.1 * hand.data.joint_vel[:, self.finger_ids[side]],
                ee_p - self.scene.env_origins - self.cube_center,
                math_utils.quat_unique(ee_q),
                10.0 * dp_home,
                dr_home,
                (self._tips(side) - core_pos.unsqueeze(1)).flatten(1),
                self._tactile_obs(i),
            ]
        obs += [
            core_pos - self.cube_center,
            math_utils.quat_unique(cube_quat),
            torch.stack([twist.sin(), twist.cos(), 0.1 * twist_vel], -1),
            self.cube.data.body_lin_vel_w[:, self.core_id],
            0.1 * self.cube.data.body_ang_vel_w[:, self.core_id],
            torch.nn.functional.one_hot(self.move_id, len(ALL_MOVES)).float(),
            self.move_roll[self.move_id].unsqueeze(-1),
            self.layer_done_latch.float().unsqueeze(-1),
            self.wrist_mask,
            (self.roll_nominal * self.wrist_mask.amax(-1)).unsqueeze(-1),
            self.actions,
        ]
        obs = torch.cat(obs, -1)
        bad = ~torch.isfinite(obs)
        if bad.any():  # NaN guard: one bad env must not poison the PPO update (v2/v3 crashed with std = NaN)
            self.nonfinite_obs_count += int(bad.any(-1).sum())
            if not self._warned_nonfinite:
                self._warned_nonfinite = True
                print(f"[rubik_lab] WARNING non-finite obs in {int(bad.any(-1).sum())} envs, columns {bad.any(0).nonzero().flatten().tolist()[:20]}", flush=True)
            obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        obs = obs.clamp(-self.cfg.clip_obs, self.cfg.clip_obs)
        return {"policy": obs}

    # ---------------------------------------------------------------- rewards
    def _layer_centers(self):
        s, d = self.cfg.cube_size, self.cube.data
        o = self.scene.env_origins

        def offset(body, z):
            q = d.body_quat_w[:, body]
            return d.body_pos_w[:, body] - o + math_utils.quat_apply(q, torch.tensor([0.0, 0.0, z], device=self.device).expand(q.shape[0], 3))

        return offset(self.core_id, -s / 6), offset(self.top_id, s / 3)

    def _update_roll_nominal(self):
        """nominal roll angle [rad] of the twisting hand: ramp to the move angle, ramp back once the layer is done."""
        dt, T = self.step_dt, max(self.cfg.roll_time, 1e-3)
        t = self.episode_length_buf.float() * dt
        target = self.move_roll[self.move_id]
        go = (t - self.cfg.roll_delay).clamp(min=0.0) / T
        # after the layer reached the target: decay towards 0 at the same speed
        back = torch.where(self.layer_done_latch, self.roll_back + dt / T, torch.zeros_like(self.roll_back))
        self.roll_back = back
        frac = torch.where(self.layer_done_latch, (self.roll_frac_at_latch - back).clamp(min=0.0), go.clamp(max=1.0))
        self.roll_frac_at_latch = torch.where(self.layer_done_latch, self.roll_frac_at_latch, frac)
        self.roll_nominal = frac * target

    def _ee_home_error(self, side):
        """EE position error (N,3) and rotation error axis-angle (N,3) w.r.t. the grasp pose of this episode."""
        p, q = self.ctrl[side].ee_pose_w()
        dp = p - self.scene.env_origins - self.ee_home_pos[side]
        dr = math_utils.axis_angle_from_quat(math_utils.quat_mul(q, math_utils.quat_inv(self.ee_home_quat[side])))
        return dp, dr

    def _tactile_obs(self, hand_idx):
        """delivered Wave SE F/T of one hand (N, 30): forces and moments scaled to ~[-1, 1]."""
        ft = self.tactile.ft[:, hand_idx]
        return torch.cat([ft[..., :3] * self.cfg.tactile_force_scale, ft[..., 3:] * self.cfg.tactile_moment_scale], -1).flatten(1)

    def _update_task_state(self):
        """called from _get_dones (runs before _get_rewards in DirectRLEnv.step)."""
        _, cube_quat, twist, twist_vel = self._cube_state()
        layer_err = math_utils.wrap_to_pi(self.move_target[self.move_id] - twist)  # unlimited joint: wrap
        # whole-cube rotations: progress = orientation error to the goal; the layer itself must stay aligned
        rot_err = math_utils.quat_error_magnitude(cube_quat, self.cube_quat_goal)
        is_rot = self.move_is_rot[self.move_id]
        self.signed_err = torch.where(is_rot, rot_err, layer_err)
        settled = (twist_vel.abs() < self.cfg.move_success_vel) & (self.cube.data.body_ang_vel_w[:, self.core_id].norm(dim=-1) < getattr(self.cfg, "move_core_ang_speed", 1.0))
        tol = torch.where(is_rot, torch.full_like(rot_err, self.cfg.rot_success_tol), torch.full_like(rot_err, self.cfg.move_success_tol))
        self.move_done = (self.signed_err.abs() < tol) & settled & (~is_rot | (layer_err.abs() < self.cfg.move_success_tol))
        self.first_reach = self.move_done & ~self.layer_done_latch
        self.layer_done_latch |= self.move_done
        home_ok = torch.ones_like(self.move_done)
        self.home_err = torch.zeros_like(self.signed_err)
        for side in SIDES:
            dp, dr = self._ee_home_error(side)
            dp, dr = torch.linalg.norm(dp, dim=-1), torch.linalg.norm(dr, dim=-1)
            home_ok &= (dp < self.cfg.home_pos_tol) & (dr < self.cfg.home_rot_tol)
            self.home_err += 10.0 * dp + dr
        self.layer_f = self.tactile.layer_forces()
        pads = (self.layer_f.sum(-1) > self.cfg.tactile_contact_threshold).sum(-1)  # (N, 2)
        self.primitive_done = self.move_done & home_ok & (pads >= self.cfg.home_min_pads).all(-1)

    def _get_rewards(self) -> torch.Tensor:
        core_pos, cube_quat, twist, twist_vel = self._cube_state()
        signed_err = self.signed_err
        err = signed_err.abs()
        progress = (self.prev_twist_err - err).clamp(-0.1, 0.1)
        self.prev_twist_err = err

        # the twisting hand's fingertips go to the moving layer, the holding hand's to the core;
        # for finger-trick moves (no own-side hand) both hands hold the core and the closest tip should reach the layer
        core_c, top_c = self._layer_centers()
        eh = self.ep_hands
        twist_hand = torch.where((eh[:, 1] > 0) & (eh[:, 0] == 0), 1, torch.where((eh[:, 0] > 0) & (eh[:, 1] == 0), 0, -1))
        tip_cost = torch.zeros_like(err)
        top_d_min = torch.full_like(err, 1.0)
        for i, side in enumerate(SIDES):
            tips = self._tips(side)
            d_core = torch.linalg.norm(tips - core_c.unsqueeze(1), dim=-1)
            d_top = torch.linalg.norm(tips - top_c.unsqueeze(1), dim=-1)
            tip_cost += torch.where(twist_hand == i, d_top.mean(-1), d_core.mean(-1))
            top_d_min = torch.min(top_d_min, d_top.min(-1).values)
        tip_cost += torch.where(twist_hand < 0, top_d_min, torch.zeros_like(top_d_min))

        # layer-selective pad contacts
        c = self.layer_f > self.cfg.tactile_contact_threshold  # (N, 2 hands, 5 pads, [core, top])
        sel = torch.zeros_like(err)
        for i in range(2):
            mine = (twist_hand == i).unsqueeze(-1)
            on_core, on_top = c[:, i, :, 0], c[:, i, :, 1]
            good = torch.where(mine, on_top & ~on_core, on_core & ~on_top).float().sum(-1)
            bad = torch.where(mine, on_core, on_top & (twist_hand >= 0).unsqueeze(-1)).float().sum(-1)
            sel += good - bad
        sel += torch.where(twist_hand < 0, c[..., 1].any(-1).any(-1).float(), torch.zeros_like(sel))

        dropped = self._dropped(core_pos)
        self._gravity_curriculum(dropped)
        # cube-motion penalties ramp up over the first steps: the finger must discover the push before being told to be gentle
        warm = 1.0
        if self.cfg.penalty_warmup_steps > 0:
            warm = min(1.0, self.cfg.penalty_warmup_min + (1.0 - self.cfg.penalty_warmup_min) * self.common_step_counter / self.cfg.penalty_warmup_steps)
        # the trick finger should push gently: with position-controlled (compliant) holding fingers a 10-25 N shove bends
        # them and turns the whole cube; the layer itself needs < 1 N
        push_f = torch.zeros_like(err)
        if self.flick_mode:
            rows = torch.arange(self.num_envs, device=self.device)
            top_f = self.cube_contacts.forces()[:, 1]  # (N, 2 hands, 27 links) force on the moving layer
            push_f = (top_f[rows, self.flick_hand[self.move_id].clamp(min=0)] * self.flick_links[self.move_id]).sum(-1)
        # calm-success bonus: a completed move pays more the less the whole cube moved during the episode (peak values)
        rot_now = math_utils.quat_error_magnitude(cube_quat, self.cube_quat_goal)
        pos_now = torch.linalg.norm(core_pos - self.cube_center, dim=-1)
        self.peak_rot = torch.max(self.peak_rot, rot_now)
        self.peak_pos = torch.max(self.peak_pos, pos_now)
        wobble = torch.max(self.peak_rot / math.radians(self.cfg.calm_rot_deg), self.peak_pos / (self.cfg.calm_pos_mm * 1e-3))
        calm = (1.0 - wobble).clamp(min=0.0, max=1.0)
        terms = {
            "calm_done": self.cfg.rew_calm_done * calm * self.primitive_done.float(),
            "push_force": warm * self.cfg.rew_push_force * (push_f - self.cfg.push_force_free).clamp(min=0),
            "progress": self.cfg.rew_twist_progress * progress,
            "done": self.cfg.rew_move_done * self.move_done.float(),
            "tip": self.cfg.rew_tip_dist * tip_cost,
            "cube_pos": warm * self.cfg.rew_cube_pos * torch.linalg.norm(core_pos - self.cube_center, dim=-1),
            "cube_rot": warm * self.cfg.rew_cube_rot * math_utils.quat_error_magnitude(cube_quat, self.cube_quat_goal),
            "action_rate": self.cfg.rew_action_rate * (self.actions - self.prev_actions).square().sum(-1),
            "drop": self.cfg.rew_drop * dropped.float(),
            "return": self.cfg.rew_return * self.home_err * self.layer_done_latch.float(),
            "primitive_done": self.cfg.rew_primitive_done * self.primitive_done.float(),
            "layer_reached": self.cfg.rew_layer_reached * self.first_reach.float(),
            "contact_selective": self.cfg.rew_contact_selective * sel,
        }
        rew = torch.nan_to_num(sum(terms.values()), nan=0.0, posinf=0.0, neginf=0.0)

        # labels for the completion detector
        layer_f = self.layer_f
        self.extras["move_done"] = self.move_done
        self.extras["primitive_done"] = self.primitive_done
        self.extras["twist_error"] = signed_err
        self.extras["tactile_layer_force"] = layer_f
        self.extras["tactile_ft"] = self.tactile.ft
        if self.tactile.image is not None:
            self.extras["tactile_image"] = self.tactile.image
        if self.record:
            self._snapshot(dropped)
        in_contact = layer_f > self.cfg.tactile_contact_threshold
        self.extras["log"] = {
            "twist_err_deg": torch.rad2deg(err).mean(),
            "move_done_rate": self.move_done.float().mean(),
            "layer_done_latch_rate": self.layer_done_latch.float().mean(),
            "primitive_done_rate": self.primitive_done.float().mean(),
            "home_err": self.home_err.mean(),
            "gravity": torch.tensor(self.gravity),
            "penalty_warm": torch.tensor(warm),
            "push_force": push_f.mean(),
            "peak_rot_deg": torch.rad2deg(self.peak_rot).mean(),
            "peak_pos_mm": 1e3 * self.peak_pos.mean(),
            "broken_envs_total": torch.tensor(float(self.broken_count)),
            "nonfinite_obs_total": torch.tensor(float(self.nonfinite_obs_count)),
            "tip_cost": tip_cost.mean(),
            "drop_rate": dropped.float().mean(),
            "tips_on_core": in_contact[..., 0].float().sum((1, 2)).mean(),
            "tips_on_layer": in_contact[..., 1].float().sum((1, 2)).mean(),
            **{f"rew_{k}": v.mean() for k, v in terms.items()},
        }
        return rew

    def _gravity_curriculum(self, dropped):
        self.drop_rate_ema = 0.99 * self.drop_rate_ema + 0.01 * dropped.float().mean().item()
        if not self.cfg.gravity_curriculum or self.common_step_counter < self.cfg.gravity_warmup_steps:
            return
        if self.drop_rate_ema < self.cfg.gravity_drop_thresh and self.gravity < 9.81:
            import carb

            self.gravity = min(9.81, self.gravity + self.cfg.gravity_step)
            self.sim.physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, -self.gravity))

    @staticmethod
    def _aligned_base_quat(axis, wrist_quat):
        """base orientation with x = axis and z = the wrist's finger direction projected orthogonal to it."""
        x = torch.nn.functional.normalize(axis, dim=-1)
        zw = math_utils.quat_apply(wrist_quat, torch.tensor([0.0, 0.0, 1.0], device=axis.device).expand_as(axis))
        z = zw - (zw * x).sum(-1, keepdim=True) * x
        alt = math_utils.quat_apply(wrist_quat, torch.tensor([0.0, 1.0, 0.0], device=axis.device).expand_as(axis))
        alt = alt - (alt * x).sum(-1, keepdim=True) * x
        z = torch.where(z.norm(dim=-1, keepdim=True) > 1e-3, z, alt)  # fingers along the axis: use the wrist y instead
        z = torch.nn.functional.normalize(z, dim=-1)
        y = torch.cross(z, x, dim=-1)
        return math_utils.quat_from_matrix(torch.stack([x, y, z], -1))

    def _snapshot(self, dropped):
        """pre-reset copy of everything a completion-detector dataset needs (DirectRLEnv resets right after rewards)."""
        snap = {
            "tactile_ft": self.tactile.ft,
            "finger_pos": torch.stack([self.hands[s].data.joint_pos[:, self.finger_ids[s]] for s in SIDES], 1),
            "finger_tgt": torch.stack([self.finger_tgt[s] for s in SIDES], 1),
            "ee_pose": torch.stack([torch.cat(self.ctrl[s].ee_pose_w(), -1) for s in SIDES], 1),
            "move_id": self.move_id,
            "twist_error": self.signed_err,
            "move_done": self.move_done,
            "layer_done_latch": self.layer_done_latch,
            "primitive_done": self.primitive_done,
            "dropped": dropped,
            "tactile_layer_force": self.layer_f,
            "episode_step": self.episode_length_buf,
            "grasp_idx": self.episode_grasp_idx,
        }
        # All diagnostic measurements must be taken before DirectRLEnv resets.
        core_pos, cube_quat, twist, twist_vel = self._cube_state()
        wrist_errors = [self._ee_home_error(side) for side in SIDES]
        broken = ~torch.isfinite(self.cube.data.root_state_w).all(-1)
        for hand in self.hands.values():
            broken |= ~torch.isfinite(hand.data.joint_pos).all(-1) | ~torch.isfinite(hand.data.joint_vel).all(-1)
        snap.update({
            "cube_pos": core_pos,
            "cube_quat": cube_quat,
            "twist": twist,
            "twist_velocity": twist_vel,
            "cube_pos_error": (core_pos - self.cube_center).norm(dim=-1),
            "cube_rot_error": math_utils.quat_error_magnitude(cube_quat, self.cube_quat_goal),
            "peak_cube_pos": self.peak_pos,
            "peak_cube_rot": self.peak_rot,
            "wrist_pos_error": torch.stack([dp.norm(dim=-1) for dp, _ in wrist_errors], -1),
            "wrist_rot_error": torch.stack([dr.norm(dim=-1) for _, dr in wrist_errors], -1),
            "timeout": self.episode_length_buf >= self.max_episode_length - 1,
            "broken": broken,
        })
        if self.tactile.image is not None:
            snap["tactile_image"] = self.tactile.image
        self.snapshot = {k: v.clone() for k, v in snap.items()}

    def _dropped(self, core_pos):
        return core_pos[:, 2] < self.cfg.cube_drop_height

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_task_state()
        core_pos = self._cube_state()[0]
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # NaN guard: an env whose physics state blew up is reset (and counted)
        broken = ~torch.isfinite(self.cube.data.root_state_w).all(-1)
        for hand in self.hands.values():
            broken |= ~torch.isfinite(hand.data.joint_pos).all(-1) | ~torch.isfinite(hand.data.joint_vel).all(-1)
        self.broken_count += int(broken.sum())
        return self._dropped(core_pos) | self.primitive_done | broken, time_out

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.cube._ALL_INDICES
        super()._reset_idx(env_ids)
        self.tactile.reset(env_ids)
        n, dev = len(env_ids), self.device
        origins = self.scene.env_origins[env_ids]
        noise = lambda shape, scale: scale * (2 * torch.rand(shape, device=dev) - 1)  # noqa: E731

        self.move_id[env_ids] = self.move_pool[torch.randint(len(self.move_pool), (n,), device=dev)]
        if self.forced_grasp_idx is not None and self.grasp_cache is not None and "move_id" in self.grasp_cache:
            # review: keep each forced grasp's own internal twist axis (re-splitting the layers lets them slip)
            fidx = self.forced_grasp_idx[env_ids] if len(self.forced_grasp_idx) == self.num_envs else self.forced_grasp_idx[:n]
            self.move_id[env_ids] = self.grasp_cache["move_id"][fidx].long()
        if self.forced_move_id is not None:  # evaluation: this move per env (overrides the cache's move id)
            self.move_id[env_ids] = self.forced_move_id[env_ids]
        g = None
        if self.grasp_cache is not None:
            # one grasp per env for all fields, generated with the same internal twist axis as this episode's move
            # (re-splitting the layers under a settled grasp lets the layer slip)
            idx = torch.multinomial(self._grasp_move_mask[self.move_id[env_ids]], 1).squeeze(-1)
            if self.forced_grasp_idx is not None:  # review / debugging: use these cache entries
                idx = self.forced_grasp_idx[env_ids] if len(self.forced_grasp_idx) == self.num_envs else self.forced_grasp_idx[: len(env_ids)]
            g = {k: v[idx] for k, v in self.grasp_cache.items()}
            self.episode_grasp_idx[env_ids] = idx
        mid = self.move_id[env_ids]
        hands = self.move_hands[mid].clone()
        shift = self.move_shift[mid].clone()
        if g is not None and "state" in g:
            edge = torch.where(g["state"] > 0.5, 0, 1)  # A: right is the edge hand, B: left
            is_u = torch.tensor([m[0] == "U" for m in ALL_MOVES], device=dev)[mid]
            up = shift.norm(dim=-1).max(-1).values  # the U spec's shift length
            hands = torch.where(is_u.unsqueeze(-1), torch.nn.functional.one_hot(edge, 2).float(), hands)
            u_shift = torch.zeros_like(shift)
            u_shift[torch.arange(n, device=dev), edge, 2] = up
            shift = torch.where(is_u.view(-1, 1, 1), u_shift, shift)
            self.ep_edge[env_ids] = edge
        self.ep_hands[env_ids] = hands
        self.ep_shift[env_ids] = shift
        self.wrist_mask[env_ids] = hands
        # regrip targets: the start grasp, except x which ends in the paired grasp of the other grip state
        tg = g
        if g is not None and "partner" in g:
            is_x = self.move_is_rot[mid]
            pidx = torch.where(is_x & (g["partner"] >= 0), g["partner"].long(), idx)
            tg = {k: v[pidx] for k, v in self.grasp_cache.items()}
        for side in SIDES:
            if tg is not None:
                self.regrip_pos[side][env_ids] = tg[f"{side}_ee_pos"]
                self.regrip_quat[side][env_ids] = tg[f"{side}_ee_quat"]
                self.regrip_fingers[side][env_ids] = tg[f"{side}_fingers"]
        self._regrip_act = tg
        logical = g["cube_quat"] if g is not None else torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).repeat(n, 1)
        axis_env = math_utils.quat_apply(logical, self.move_axis[self.move_id[env_ids]])
        self.roll_axis[env_ids] = axis_env
        roll_q = math_utils.quat_from_angle_axis(self.move_roll[self.move_id[env_ids]] * self.move_is_rot[self.move_id[env_ids]], axis_env)
        self.cube_quat_goal[env_ids] = math_utils.quat_mul(roll_q, logical)

        for h_idx, (side, hand) in enumerate(self.hands.items()):
            # targets (*) and actual state (*_act) differ in a grasp: the PD offset is what squeezes the cube
            if g is not None:
                pos, quat, fingers = g[f"{side}_ee_pos"].clone(), g[f"{side}_ee_quat"], g[f"{side}_fingers"]
                pos_act, quat_act, fingers_act = g[f"{side}_ee_pos_act"].clone(), g[f"{side}_ee_quat_act"], g[f"{side}_fingers_act"]
            else:
                pos = torch.tensor(self.home[side][0], device=dev).repeat(n, 1)
                quat = torch.tensor(self.home[side][1], device=dev).repeat(n, 1)
                fingers = torch.tensor(self.cfg.finger_init, device=dev).repeat(n, 1)
                fingers += noise(fingers.shape, self.cfg.finger_init_noise)
                pos_act, quat_act, fingers_act = pos.clone(), quat, fingers
            # noise on the targets only: the cached actual state is a settled contact state, perturbing it would
            # make the hands penetrate the cube and PhysX would launch it
            if g is not None and self.cfg.wrist_stiffness is not None:
                # the cached target-actual offset was settled with the USD's 3000 N/m: scale it to keep the same squeeze
                pos = pos_act + (pos - pos_act) * (3000.0 / self.cfg.wrist_stiffness)
            pos = pos + noise((n, 3), self.cfg.ee_init_pos_noise)
            if g is None:
                pos_act = pos.clone()
            lo, hi = self.finger_lower[side][env_ids], self.finger_upper[side][env_ids]
            fingers = torch.max(torch.min(fingers, hi), lo)
            self.ee_pos_tgt[side][env_ids] = pos
            self.ee_quat_tgt[side][env_ids] = quat
            self.ee_grasp_tgt[side][env_ids] = pos
            self.ee_grasp_quat[side][env_ids] = quat
            self.finger_grasp[side][env_ids] = fingers
            tg = self._regrip_act
            self.ee_home_pos[side][env_ids] = tg[f"{side}_ee_pos_act"] if tg is not None else pos_act
            self.ee_home_quat[side][env_ids] = tg[f"{side}_ee_quat_act"] if tg is not None else quat_act
            self.finger_tgt[side][env_ids] = fingers

            # fixed base: at the grasp's base pose if the cache has one (synthesized grasps put it at the wrist target, so
            # the virtual joints start near 0 for any wrist orientation), else at the default home pose
            if g is not None and f"{side}_base_pos" in g:
                bp, bq = g[f"{side}_base_pos"], g[f"{side}_base_quat"]
            else:
                bp = torch.tensor(self.home[side][0], device=dev).repeat(n, 1)
                bq = torch.tensor(self.home[side][1], device=dev).repeat(n, 1)
            # a rotating wrist gets its base x axis on the rotation axis: the twist is then a pure roll (first virtual
            # joint) and the pitch stays at its start value (no Euler singularity for any grasp orientation)
            rot_me = self.wrist_mask[env_ids][:, h_idx] > 0
            if rot_me.any():
                bq = torch.where(rot_me.unsqueeze(-1), self._aligned_base_quat(axis_env, quat), bq)
            self.ctrl[side].set_base_pose(bp + origins, bq, env_ids)
            zeros6 = torch.zeros(n, 6, device=dev)
            base = self.ctrl[side].joint_targets(pos + origins, quat, joint_pos=zeros6, env_ids=env_ids)
            base_act = self.ctrl[side].joint_targets(pos_act + origins, quat_act, joint_pos=zeros6, env_ids=env_ids)
            if g is not None and f"{side}_joint_pos_act" in g:
                joint_pos = g[f"{side}_joint_pos_act"].clone()  # exact finger restore of the settled grasp
            else:
                joint_pos = hand.data.default_joint_pos[env_ids].clone()
                joint_pos[:, self.finger_ids[side]] = fingers_act
            joint_pos[:, self.ctrl[side].joint_ids] = base_act  # virtual joints for the (possibly re-aligned) base
            hand.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
            joint_tgt = joint_pos.clone()
            joint_tgt[:, self.ctrl[side].joint_ids] = base
            joint_tgt[:, self.finger_ids[side]] = fingers
            hand.set_joint_position_target(joint_tgt, env_ids=env_ids)
            self._base_tgt[side][env_ids] = base

        # cube: logical pose from the grasp (or nominal), physical root rotated so the twist axis matches the move
        root = self.cube.data.default_root_state[env_ids].clone()
        if g is not None:
            root[:, :3] = g["cube_pos"]
        root[:, :3] += origins + (0.0 if g is not None else noise((n, 3), self.cfg.cube_init_pos_noise))
        root[:, 3:7] = math_utils.quat_mul(logical, self.move_face_quat[self.move_id[env_ids]])
        root[:, 7:] = 0.0
        self.cube.write_root_state_to_sim(root, env_ids=env_ids)
        twist0 = g["cube_twist"].unsqueeze(-1) if g is not None and "cube_twist" in g else torch.zeros(n, 1, device=dev)
        self.reset_face_match[env_ids] = (
            (self.move_face_quat[g["move_id"].long()] - self.move_face_quat[self.move_id[env_ids]]).abs().sum(-1) < 1e-4
            if g is not None and "move_id" in g
            else False
        )
        self.cube.write_joint_state_to_sim(twist0, torch.zeros_like(twist0), joint_ids=[self.twist_id], env_ids=env_ids)
        self.cube_center[env_ids] = root[:, :3] - origins
        self.cube_quat0[env_ids] = logical

        mid = self.move_id[env_ids]
        self.prev_twist_err[env_ids] = torch.where(
            self.move_is_rot[mid], self.move_roll[mid].abs(), math_utils.wrap_to_pi(self.move_target[mid] - twist0[:, 0]).abs()
        )
        self.move_done[env_ids] = False
        self.layer_done_latch[env_ids] = False
        self.primitive_done[env_ids] = False
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.action_smooth[env_ids] = 0.0
        if self.flick_mode:
            self.flick_lift[env_ids] = 0.0
            self.flick_lift_valid[env_ids] = False
        self.peak_rot[env_ids] = 0.0
        self.peak_pos[env_ids] = 0.0
        self.roll_nominal[env_ids] = 0.0
        self.roll_back[env_ids] = 0.0
        self.roll_frac_at_latch[env_ids] = 0.0
        self.first_reach[env_ids] = False
