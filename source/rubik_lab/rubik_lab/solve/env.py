"""Skill environment for a full blind solve: one physical skill per episode (training) or chained skills (executor).

Hands-fixed frame = env frame: the ready grip is the same env-frame pose for every skill; rolls and regrips move the
wrists about the hand's face normal through the nominal cube center. The cube is a two-body articulation whose free
slab is re-bound to the skill's face at every skill start (rubik_lab.solve.binding).

Observation groups:
    "policy": privileged teacher input (proprioception, tactile F/T, cube state, skill, phase)
    "student": deployable input only (proprioception, delivered tactile F/T, commanded targets, skill, phase)
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as mu
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from rubik_lab.assets import ASSET_DIR, CONTACT_MODELS, FINGER_JOINTS, FINGERTIPS, sharpa_fr3_cfg, sharpa_hand_cfg
from rubik_lab.cube_contacts import LINKS, CubeContacts
from rubik_lab.ee_pose_controller import FloatBaseEEController
from rubik_lab.fr3_arm import FR3EEController, load_layout
from rubik_lab.wave_se_tactile import WaveSETactile

from . import skills as S
from .binding import cube_group, face_index, quarter_residual, rebind_rotation
from .env_cfg import SolveEnvCfg

SIDES = S.HANDS
PROJECT_DIR = os.path.dirname(ASSET_DIR)
FACE_NORMAL = {"R": (0.0, -1.0, 0.0), "L": (0.0, 1.0, 0.0), "U": (0.0, 0.0, 1.0), "D": (0.0, 0.0, -1.0),
               "F": (-1.0, 0.0, 0.0), "B": (1.0, 0.0, 0.0)}  # fmt: skip
FINGER_OF = [j.split("_")[0] for j in FINGER_JOINTS]
IS_FLEX = [("FE" in j or j.endswith(("PIP", "DIP", "IP"))) for j in FINGER_JOINTS]


class SolveEnv(DirectRLEnv):
    cfg: SolveEnvCfg

    def __init__(self, cfg: SolveEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.sim.dt = 1.0 / cfg.physics_hz
        cfg.sim.render_interval = cfg.decimation
        cm = CONTACT_MODELS[cfg.contact_model]
        self._fr3 = None
        if cfg.arm == "fr3":  # arm bases relative to the cached cube position
            c0 = np.load(os.path.join(PROJECT_DIR, cfg.ready_cache))["cube_pos"].mean(0).tolist()
            self._fr3 = {s: load_layout(os.path.join(PROJECT_DIR, cfg.fr3_layout), s, c0) for s in SIDES}
        for side in SIDES:
            if self._fr3 is not None:
                hc = sharpa_fr3_cfg(side, *self._fr3[side], contact_model=cfg.contact_model, arm_stiffness=cfg.fr3_arm_stiffness,
                                    arm_collision=cfg.fr3_arm_collision, usd_dir=cfg.fr3_usd_dir)
            else:
                hc = sharpa_hand_cfg(side, contact_model=cfg.contact_model, wrist_stiffness=cfg.wrist_stiffness)
                # arm-like wrist speed limits (the USD has 2 m/s / 2 rad/s)
                hc.actuators["float_base"].velocity_limit_sim = {f"{side}_[xyz]_joint": cfg.wrist_lin_vel_limit, f"{side}_(roll|pitch|yaw)_joint": cfg.wrist_ang_vel_limit}
            setattr(cfg, f"{side}_hand_cfg", hc)
        if cfg.cube_3body:
            cfg.cube_cfg = cfg.cube_cfg.copy()
            cfg.cube_cfg.spawn.usd_path = cfg.cube_cfg.spawn.usd_path.replace("rubik_cube_2body", "rubik_cube_3body")
            cfg.cube_cfg.actuators["twist"].joint_names_expr = ["twist_joint", "mid_joint"]
            cc = cfg.tactile.contact_cfg
            cc.filter_prim_paths_expr = list(cc.filter_prim_paths_expr) + ["/World/envs/env_.*/cube/mid"]  # after [core, top]
            cfg.tactile.image_targets = tuple(cfg.tactile.image_targets) + ("/World/envs/env_.*/cube/mid/collision",)
        cfg.cube_cfg.spawn.collision_props.contact_offset = cm["obj_contact_offset"]
        cfg.cube_cfg.spawn.collision_props.rest_offset = cm["obj_rest_offset"]
        cfg.action_space = 56
        self._n_skill = len(S.SKILLS)
        # teacher: per hand q, qd, q_tgt (66) + wrist err (6) + F/T (30) = 102; cube 3+6+3+6+2 = 20; skill 13 + phase 2
        # + roll 2 + actions 56
        cfg.observation_space = 2 * 102 + 20 + self._n_skill + 2 + 2 + 56 + (self._n_skill if cfg.obs_next_skill else 0)
        cfg.observation_space += 56 * cfg.obs_act_hist
        self._ready_loaded = False
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device
        self.group = cube_group(dev)

        self.ctrl, self.finger_ids = {}, {}
        self.finger_lo, self.finger_hi = {}, {}
        for side, hand in self.hands.items():
            if self._fr3 is not None:
                bp, bq, q0 = self._fr3[side]
                self.ctrl[side] = FR3EEController(hand, side, self.scene.env_origins + torch.tensor(bp, device=dev), bq, q0,
                                                  urdf_dir=os.path.join(PROJECT_DIR, cfg.fr3_urdf_dir) if cfg.fr3_urdf_dir else None)
            else:
                self.ctrl[side] = FloatBaseEEController(hand, side)
            self.finger_ids[side] = hand.find_joints([f"{side}_{j}" for j in FINGER_JOINTS], preserve_order=True)[0]
            lim = hand.data.soft_joint_pos_limits[0, self.finger_ids[side]]
            self.finger_lo[side], self.finger_hi[side] = lim[:, 0], lim[:, 1]
        self.tactile.initialize(self.hands, n, self.physics_dt, dev)
        self.core_id = self.cube.find_bodies("core")[0][0]
        self.top_id = self.cube.find_bodies("top")[0][0]
        self.twist_id = self.cube.find_joints("twist_joint")[0][0]
        self.mid_id = self.cube.find_joints("mid_joint")[0][0] if cfg.cube_3body else None
        if cfg.use_joint_friction:
            self.cube.write_joint_friction_coefficient_to_sim(
                cfg.twist_static_friction, cfg.twist_dynamic_friction, cfg.twist_viscous_friction, joint_ids=[self.twist_id]
            )
            if self.mid_id is not None:
                m = cfg.mid_friction_scale
                self.cube.write_joint_friction_coefficient_to_sim(
                    cfg.twist_static_friction * m, cfg.twist_dynamic_friction * m, cfg.twist_viscous_friction * m, joint_ids=[self.mid_id]
                )

        # skill tables
        self.skill_pool = torch.tensor([S.SKILL_INDEX[s] for s in cfg.skills], device=dev)
        w = cfg.skill_weights or [1.0] * len(cfg.skills)
        self.skill_w = torch.tensor(w, device=dev, dtype=torch.float32)
        K = self._n_skill
        self.k_bind_normal = torch.tensor([FACE_NORMAL[S.BIND_FACE[s]] for s in S.SKILLS], device=dev)
        self.k_layer_target = torch.tensor([S.LAYER_TARGET.get(s, 0.0) for s in S.SKILLS], device=dev)
        self.k_roll = torch.zeros(K, 2, device=dev)
        for s, d in S.ROLL_DELTA.items():
            for h, v in d.items():
                self.k_roll[S.SKILL_INDEX[s], SIDES.index(h)] = v
        self.k_cube_q = torch.tensor([S.CUBE_QUARTERS_ABOUT_R.get(s, 0) for s in S.SKILLS], device=dev, dtype=torch.float32)
        self.k_regrip = torch.full((K,), -1, dtype=torch.long, device=dev)
        for s, h in S.REGRIP_HAND.items():
            self.k_regrip[S.SKILL_INDEX[s]] = SIDES.index(h)
        self.k_flick_hand = torch.full((K,), -1, dtype=torch.long, device=dev)
        self.k_flick_cols = torch.zeros(K, 4, dtype=torch.long, device=dev)
        self.k_flick_links = torch.zeros(K, len(LINKS), device=dev)
        self.k_flick_pad = torch.zeros(K, dtype=torch.long, device=dev)
        for s, (h, f) in S.FLICK.items():
            k = S.SKILL_INDEX[s]
            self.k_flick_hand[k] = SIDES.index(h)
            self.k_flick_cols[k] = torch.tensor([i for i, j in enumerate(FINGER_JOINTS) if j.startswith(f)][:4], device=dev)
            self.k_flick_links[k] = torch.tensor([ln.startswith(f) for ln in LINKS], device=dev).float()
            self.k_flick_pad[k] = FINGERTIPS.index(f)
        self.k_is_flick = self.k_flick_hand >= 0
        self.k_flick_d = torch.tensor([s in S.FLICK and S.BIND_FACE[s] == "D" for s in S.SKILLS], device=dev)
        self.k_is_roll = self.k_roll.abs().sum(-1) > 0
        self.k_is_x = self.k_cube_q != 0
        self.k_is_regrip = self.k_regrip >= 0
        self.k_duration = torch.tensor([S.nominal_duration(s) for s in S.SKILLS], device=dev)
        self.flex = {f: torch.tensor([FINGER_OF[i] == f and IS_FLEX[i] for i in range(22)], device=dev).float() for f in FINGERTIPS}
        self.flex_all = torch.tensor(IS_FLEX, device=dev).float()
        self.hold_flex = (self.flex["thumb"] + self.flex["middle"] + self.flex["ring"]).clamp(max=1.0)
        # (2 hands, roll -1 / 0 / +1, 22): flexion joints of a rolled hand's digits on the U layer (lifted during a flick)
        self.on_u = torch.zeros(2, 3, 22, device=dev)
        for (side, r), digits in S.ROLLED_ON_U.items():
            for f in digits:
                self.on_u[SIDES.index(side), r + 1] += self.flex[f]
        self.hand_normal = torch.tensor([S.HAND_NORMAL[s] for s in SIDES], device=dev)  # (2, 3)
        # per-skill action scale (in units of the cfg ranges): flicks mostly move the trick finger
        self.k_act_scale = torch.ones(K, 56, device=dev)
        for k in range(K):
            if self.k_is_flick[k]:
                self.k_act_scale[k] = cfg.flick_other_scale
                cols = self.k_flick_hand[k] * 28 + 6 + self.k_flick_cols[k]
                self.k_act_scale[k, cols] = cfg.flick_trick_scale
            elif cfg.wrist_act is not None and S.SKILLS[k] != "hold":  # wrist family: its own residual ranges
                for h in range(2):
                    self.k_act_scale[k, h * 28 : h * 28 + 3] = cfg.wrist_act[0] / cfg.act_wrist_pos
                    self.k_act_scale[k, h * 28 + 3 : h * 28 + 6] = cfg.wrist_act[1] / cfg.act_wrist_rot
                    self.k_act_scale[k, h * 28 + 6 : h * 28 + 28] = cfg.wrist_act[2] / cfg.act_finger

        # nominal flick: median trick-finger target offsets of successful demonstrations (per control step)
        self.flick_nom = None
        fpath = os.path.join(PROJECT_DIR, cfg.flick_nominal) if cfg.flick_nominal else ""
        if fpath and os.path.exists(fpath):
            fz = np.load(fpath)
            T = max(v.shape[0] for v in fz.values())
            self.flick_nom = torch.zeros(K, T, 4, device=dev)
            for s_, kk in (("U", "U"), ("U'", "Up"), ("D", "D"), ("D'", "Dp")):
                if kk in fz:
                    self.flick_nom[S.SKILL_INDEX[s_]] = torch.tensor(fz[kk], device=dev)
            self.k_duration[self.k_is_flick] = cfg.flick_nominal_s
            print(f"[solve] flick nominal from {fpath}: {list(fz.keys())}", flush=True)
        # ready grip
        self._load_ready()
        if self._fr3 is not None and os.environ.get("FR3_ROLL_TABLE", "1") == "1":
            self._build_roll_tables()

        # per-env state
        z = lambda *s, **kw: torch.zeros(n, *s, device=dev, **kw)  # noqa: E731
        self.skill = z(dtype=torch.long)
        self.t_skill = z()
        self.roll = z(2)  # quarter turns of each hand about its face normal, at the current time (nominal)
        self.roll0 = z(2)  # at skill start
        self.servo_p, self.servo_aa = z(3), z(3)  # regrip servo: cube deviation (position, axis-angle), frozen on approach
        self.recenter_p, self.recenter_aa = z(3), z(3)  # recentering regrip: latched cube deviation
        self.hand_dbg = torch.zeros(n, 2, 4, device=dev)  # per hand: wrist error mm, deg, max finger error, speed
        self.ready_idx = z(dtype=torch.long)
        self.grip_mode = z(2, dtype=torch.long)  # per hand: 0 ready (wide) grip, 1 alternative (edge) grip
        self.grip_mode_next = z(2, dtype=torch.long)  # a regrip closes into this mode (set by the caller after start_skill)
        self.alt_idx = z(2, dtype=torch.long)
        self.c0 = z(3)  # nominal cube center (env frame)
        self.cube_goal = z(4)  # logical cube orientation the skill should end in (core frame of the ready binding)
        self.cube_ref = z(4)  # orientation the cube is supposed to have before the skill's own motion
        self.layer_target = z()
        self.bind_S = torch.eye(3, device=dev).repeat(n, 1, 1)  # R_core = R_logical @ bind_S
        self.bind_residual = z()  # layer angle snapped away by the last re-binding [rad]
        # cfg.cube_jam_deg: per logical face (+x -x +y -y +z -z) its residual off the middle slab [rad, about the face's outward
        # normal]; the free face's lives in the twist joint (and, 3-body, its opposite face's in the middle joint)
        self.face_res = z(6)
        self.jam = z(dtype=torch.bool)  # the free layer is locked by a misaligned layer on another axis
        self.actions = z(56)
        self.prev_actions = z(56)
        self._act_hist = z(self.cfg.obs_act_hist, 56)  # cfg.obs_act_hist: newest first
        self.act_smooth = z(56)
        self.wrist_tgt_pos = z(2, 3)
        self.wrist_tgt_quat = z(2, 4)
        self.finger_tgt = z(2, 22)
        # absolute finger targets that replace nominal + residual where finite (external skill controllers, e.g. a
        # legacy flick policy); cleared at every skill start
        self.finger_override = torch.full((n, 2, 22), float("nan"), device=dev)
        self.peak_pos = z()
        self.peak_rot = z()
        self.stable_steps = z(dtype=torch.long)
        self.win_pos = z(3)
        self.win_quat = z(4)
        self.win_twist = z()
        self.success = z(dtype=torch.bool)
        self.success_t = z()
        self.timeout_adv = z(dtype=torch.bool)  # chain: timed-out skills that advance anyway
        self.early_adv = z(dtype=torch.bool)  # chain: this skill advances as soon as its completion window opens (premature)
        self.violation = z(dtype=torch.bool)
        self.prev_err = z()
        self.tilt = z()  # angle of the cube to the nearest axis-aligned orientation [rad]
        self.buf, self.buf_head, self.buf_count, self._started = None, 0, 0, False
        self.extra_twist_effort = z()
        self._last_success = z(dtype=torch.bool)
        self._last_violation = z(dtype=torch.bool)
        self._last_dropped = z(dtype=torch.bool)
        self._last_t = z()
        self._last_twist = z()
        self.stat_n = torch.zeros(self._n_skill, device=dev)
        self.stat_ok = torch.zeros(self._n_skill, device=dev)
        self.stat_len = torch.zeros(self._n_skill, device=dev)  # diagnostic torque on the free layer [Nm]
        self.chain_mode = False  # executor: no auto-reset on success, skills started explicitly
        self.next_skill = torch.full((n,), -1, dtype=torch.long, device=dev)  # -1: unknown / none (obs_next_skill)
        # blind student control (cfg.student_mode): teacher's raw action / its smoothed shadow / previous raw action,
        # the label (teacher targets in the student's action space), the DAgger mask, the student's executed action
        self.teacher_raw, self.teacher_smooth, self.teacher_prev = z(56), z(56), z(56)
        self.teacher_free, self.student_prev = z(56), z(56)
        self.teacher_mask = z(dtype=torch.bool)
        # blind state estimation (cfg.cube_est): the cube state the controllers act on, set every step by the estimator
        self.est_cube = z(17)
        self._goal_fresh = torch.ones(n, dtype=torch.bool, device=dev)
        if cfg.chain_train:
            self._build_chain_bank()
        self.flick_legacy, self.legacy_mine, self.legacy_full = None, None, None
        if cfg.flick_legacy_ckpt:
            from .policies import LegacyFlickPolicy

            self.flick_legacy = LegacyFlickPolicy(self, os.path.join(PROJECT_DIR, cfg.flick_legacy_ckpt), **cfg.flick_legacy_kw)
        self._init_sim2real()
        self._init_vision()
        # built-in blind estimator (cfg.cube_est_ckpt): stepped once per control step inside _get_observations
        self.estimator, self._est_due = None, True
        if cfg.cube_est_ckpt:
            from .estimator import Estimator

            cfg.cube_est = True
            self.estimator = Estimator(os.path.join(PROJECT_DIR, cfg.cube_est_ckpt), n, dev)

    # ------------------------------------------------------------------ setup
    def _setup_scene(self):
        self.hands = {s: Articulation(getattr(self.cfg, f"{s}_hand_cfg")) for s in SIDES}
        self.cube = Articulation(self.cfg.cube_cfg)
        self.tactile = WaveSETactile(self.cfg.tactile, self.scene)
        # per-link cube contacts (54 filtered links per body) are expensive; pad sensors suffice for training
        self.cube_contacts = CubeContacts(self.scene) if self.cfg.link_contacts else None
        # with arms the floor is the table the bases stand on
        floor = min(self._fr3[s_][0][2] for s_ in SIDES) if self._fr3 is not None else 0.0
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, floor))
        self.scene.clone_environments(copy_from_source=False)
        for side in SIDES:
            self.scene.articulations[f"{side}_hand"] = self.hands[side]
        self.scene.articulations["cube"] = self.cube
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    def _load_ready(self):
        path = os.path.join(PROJECT_DIR, self.cfg.ready_cache)
        g = {k: torch.tensor(v, device=self.device, dtype=torch.float32) for k, v in np.load(path).items()}
        if self.cfg.ready_index >= 0:
            g = {k: v[self.cfg.ready_index : self.cfg.ready_index + 1] for k, v in g.items()}
        if self.cfg.wrist_stiffness is not None:
            # the cached target-actual wrist offset was settled with the USD's 3000 N/m: keep the same squeeze force
            for side in SIDES:
                act = g[f"{side}_ee_pos_act"]
                g[f"{side}_ee_pos"] = act + (g[f"{side}_ee_pos"] - act) * (3000.0 / self.cfg.wrist_stiffness)
        self.ready = g
        self.n_ready = len(g["cube_pos"])
        lift = torch.zeros(22, device=self.device)
        for f, a in self.cfg.idle_lift.items():
            lift -= a * self.flex[f]
        self.idle_lift_vec = -lift * self.cfg.regrip_power_grip  # re-flexion that undoes the idle lift
        press = self.cfg.hold_press * (self.flex["thumb"] + self.flex["middle"]).clamp(max=1.0)
        self.ready_fingers = {}
        self.ready_fingers_act = {side: g[f"{side}_fingers_act"] for side in SIDES}
        for side in SIDES:
            tgt = g[f"{side}_fingers"] + lift + press
            self.ready_fingers[side] = torch.max(torch.min(tgt, self.finger_hi[side]), self.finger_lo[side])
        print(f"[solve] ready grip: {self.n_ready} states from {path}", flush=True)
        # grip modes: 0 = the ready grip above (wide: the hand holds its own + the middle column), 1 = an alternative
        # grasp of one hand (edge: its own column only, for the wrist rolls R / L), from grip_alt_cache[side]
        self.alt = {}
        for side in SIDES:
            ap = self.cfg.grip_alt_cache.get(side, "")
            if not ap:
                self.alt[side] = None
                continue
            a = {k: torch.tensor(v, device=self.device, dtype=torch.float32) for k, v in np.load(os.path.join(PROJECT_DIR, ap)).items()}
            tgt = a[f"{side}_fingers"] + lift + press
            self.alt[side] = dict(pos=a[f"{side}_ee_pos"] - a["cube_pos"], quat=a[f"{side}_ee_quat"], fingers_act=a[f"{side}_fingers_act"],
                                  fingers=torch.max(torch.min(tgt, self.finger_hi[side]), self.finger_lo[side]), n=len(a["cube_pos"]))
            print(f"[solve] {side} alternative grip: {self.alt[side]['n']} grasps from {ap}", flush=True)

    def _hand_ready(self, h, side, mode):
        """per-env ready wrist pose and finger targets (squeezing / settled) of one hand in grip mode `mode` (N,)."""
        i = self.ready_idx
        rp, rq = self.ready[f"{side}_ee_pos"][i], self.ready[f"{side}_ee_quat"][i]
        f, fa = self.ready_fingers[side][i], self.ready_fingers_act[side][i]
        a = self.alt[side]
        if a is None:
            return rp, rq, f, fa
        j, m = self.alt_idx[:, h], (mode == 1).unsqueeze(-1)
        return (torch.where(m, self.c0 + a["pos"][j], rp), torch.where(m, a["quat"][j], rq),
                torch.where(m, a["fingers"][j], f), torch.where(m, a["fingers_act"][j], fa))

    # ------------------------------------------------------------- skills
    def start_skill(self, env_ids: torch.Tensor, skill_ids: torch.Tensor, retry: bool = False):
        """Begin a skill in these envs from their CURRENT physical state (no teleport except the face re-binding).

        retry: repeat a face turn that stopped short, keeping the binding and the layer target (no snapping)."""
        if len(env_ids) == 0:
            return
        self.finger_override[env_ids] = float("nan")
        if retry:
            self.t_skill[env_ids] = 0.0
            self.prev_err[env_ids] = self._task_error(env_ids)
            self.peak_pos[env_ids] = 0.0
            self.peak_rot[env_ids] = 0.0
            self.stable_steps[env_ids] = 0
            self.success[env_ids] = False
            self.success_t[env_ids] = 0.0
            self.violation[env_ids] = False
            if self._vis is not None:
                self.stable_vis[env_ids], self.success_vis[env_ids] = 0, False
            return
        self.skill[env_ids] = skill_ids
        self.t_skill[env_ids] = 0.0
        if self.cfg.chain_early_advance_p > 0:
            self.early_adv[env_ids] = torch.rand(len(env_ids), device=self.device) < self.cfg.chain_early_advance_p
            self.timeout_adv[env_ids] = False  # no advance pending into the new skill
        self.roll0[env_ids] = self.roll[env_ids]
        self.grip_mode[env_ids] = self.grip_mode_next[env_ids]
        self._rebind(env_ids, self.k_bind_normal[skill_ids])
        # cube goal: x / x' rotate the whole cube about the R normal; others keep it
        ang = self.k_cube_q[skill_ids] * (0.5 * math.pi)
        axis = torch.tensor(FACE_NORMAL["R"], device=self.device).expand(len(env_ids), 3)
        dq = mu.quat_from_angle_axis(ang, axis)
        logical = self._logical_quat(env_ids)
        if self.cfg.goal_from_perceived and self.cfg.cube_est:  # the blind estimate's orientation, chained over skills
            keep = ~self._goal_fresh[env_ids]  # (an episode starts from the known placement in the ready grip)
            logical = torch.where(keep.unsqueeze(-1), self.perceived()["lq"][env_ids], logical)
        self._goal_fresh[env_ids] = False
        if self._vis is not None:  # the plan's reference orientation is the camera's (fresh episodes: a first frame now)
            v = self._vis
            f = env_ids[v["fresh"][env_ids]]
            if len(f):
                d = self.cube.data
                v["pos"][f], v["lq"][f], v["tw"][f] = self._vis_frame(d.root_pos_w[f] - self.scene.env_origins[f] - self.c0[f], self._logical_quat(f),
                                                                       d.joint_pos[f, self.twist_id], f)
            if self.cfg.vis_control:
                logical = v["lq"][env_ids]
            self.stable_vis[env_ids], self.success_vis[env_ids] = 0, False
        self.cube_ref[env_ids] = logical
        self.cube_goal[env_ids] = mu.quat_mul(dq, logical)
        self.layer_target[env_ids] = self.k_layer_target[skill_ids]
        self.prev_err[env_ids] = self._task_error(env_ids)
        self.peak_pos[env_ids] = 0.0
        self.peak_rot[env_ids] = 0.0
        self.stable_steps[env_ids] = 0
        self.success[env_ids] = False
        self.success_t[env_ids] = 0.0
        self.violation[env_ids] = False

    def _logical_quat(self, env_ids=None):
        """continuous cube orientation (env frame), independent of the face binding: R_core = R_logical @ bind_S."""
        R = mu.matrix_from_quat(self._core_quat()) @ self.bind_S.transpose(-1, -2)
        q = mu.quat_from_matrix(R)
        return q if env_ids is None else q[env_ids]

    def _cube_deviation(self, perceived=False):
        """(dp, dq): rigid transform taking the ready-grip cube pose to the current one, modulo cube symmetry
        (perceived: of the cube state the controllers act on, see perceived())."""
        if perceived:
            per = self.perceived()
            p, lq = per["pos"] + self.c0, per["lq"]
        else:
            p, lq = self.cube.data.root_pos_w - self.scene.env_origins, self._logical_quat()
        R_l = mu.matrix_from_quat(lq)
        R_r = mu.matrix_from_quat(self.ready["cube_quat"][self.ready_idx])
        rel = R_r.transpose(-1, -2) @ R_l  # ~ a group element G when the cube sits as in the ready grip
        cand = rel.unsqueeze(1) @ self.group.transpose(-1, -2).unsqueeze(0)
        best = cand.diagonal(dim1=-2, dim2=-1).sum(-1).argmax(-1)
        G = self.group[best]
        D = R_l @ G.transpose(-1, -2) @ R_r.transpose(-1, -2)
        return p - self.c0, mu.quat_from_matrix(D)

    def _core_quat(self):
        return self.cube.data.root_quat_w  # root body = core (valid right after a state write, unlike body_*_w)

    def _rebind(self, env_ids, normals):
        d = self.cube.data
        twist = d.joint_pos[env_ids, self.twist_id]
        res = quarter_residual(twist)
        self.bind_residual[env_ids] = res
        f_old = self.bind_S[env_ids][:, :, 2].clone()  # logical normal of the free layer (R_core e_z = R_logical S e_z)
        root = d.root_state_w[env_ids].clone()
        R_core = mu.matrix_from_quat(root[:, 3:7])
        R_new = rebind_rotation(R_core, normals, self.group)
        # R_core = R_logical @ S  ->  R_new = R_core @ G = R_logical @ (S @ G)
        self.bind_S[env_ids] = self.bind_S[env_ids] @ (R_core.transpose(-1, -2) @ R_new)
        root[:, 3:7] = mu.quat_from_matrix(R_new)
        self.cube.write_root_state_to_sim(root, env_ids=env_ids)
        zero = torch.zeros(len(env_ids), 1, device=self.device)
        tw0, m0 = zero, zero
        if self.cfg.cube_jam_deg is not None:
            # the free joints go back to the face slots, the new free face (and its opposite: the middle joint) come out of
            # them; a turn about another axis snaps every layer within the corner cutting into line
            r = torch.arange(len(env_ids), device=self.device)
            i_old, i_new = face_index(f_old), face_index(self.bind_S[env_ids][:, :, 2])
            fr = self.face_res[env_ids].clone()
            fr[r, i_old] = res
            if self.mid_id is not None:
                fr[r, i_old ^ 1] = quarter_residual(d.joint_pos[env_ids, self.mid_id])
            ortho = i_old // 2 != i_new // 2
            th = math.radians(self.cfg.cube_jam_deg)
            fr = torch.where(ortho.unsqueeze(-1) & (fr.abs() <= th), torch.zeros_like(fr), fr)
            tw0 = fr[r, i_new].clone().unsqueeze(-1)
            fr[r, i_new] = 0.0
            if self.mid_id is not None:
                m0 = fr[r, i_new ^ 1].clone().unsqueeze(-1)
                fr[r, i_new ^ 1] = 0.0
            self.face_res[env_ids] = fr
            self._set_jam(env_ids, self._jam_of(env_ids))
        if self.mid_id is not None:
            self.cube.write_joint_state_to_sim(m0, zero, joint_ids=[self.mid_id], env_ids=env_ids)
        self.cube.write_joint_state_to_sim(tw0, zero, joint_ids=[self.twist_id], env_ids=env_ids)
        if self._vis is not None:
            self._vis_rebind(env_ids)

    def _jam_of(self, env_ids):
        """(n,) the free layer is blocked: a layer on another axis is out of line by more than cfg.cube_jam_deg."""
        axis = face_index(self.bind_S[env_ids][:, :, 2]) // 2
        other = torch.arange(6, device=self.device).unsqueeze(0) // 2 != axis.unsqueeze(-1)
        return ((self.face_res[env_ids].abs() > math.radians(self.cfg.cube_jam_deg)) & other).any(-1)

    def _set_jam(self, env_ids, jam):
        """lock (stiff implicit drive at the bound angle 0) or free the twist joint of these envs."""
        self.jam[env_ids] = jam
        if not hasattr(self, "_twist_d0"):
            self._twist_d0 = self.cube.data.joint_damping[:, self.twist_id].clone()
            self.cube.set_joint_position_target(torch.zeros(self.num_envs, 1, device=self.device), joint_ids=[self.twist_id])
        k = torch.where(jam, 50.0, 0.0).unsqueeze(-1)
        c = torch.where(jam, torch.full_like(self._twist_d0[env_ids], 0.05), self._twist_d0[env_ids]).unsqueeze(-1)
        self.cube.write_joint_stiffness_to_sim(k, joint_ids=[self.twist_id], env_ids=env_ids)
        self.cube.write_joint_damping_to_sim(c, joint_ids=[self.twist_id], env_ids=env_ids)

    @property
    def misaligned(self):
        """(N,) a layer is out of line by more than cfg.cube_jam_deg (the free one or the remembered one)."""
        th = math.radians(self.cfg.cube_jam_deg or 15.0)
        bad = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.mid_id is not None:
            bad |= quarter_residual(self.cube.data.joint_pos[:, self.mid_id]).abs() > th
        if self.cfg.cube_jam_deg is None:
            return bad
        free = quarter_residual(self.cube.data.joint_pos[:, self.twist_id]).abs() > th
        return bad | free | (self.face_res.abs() > th).any(-1)

    # ------------------------------------------------------------- control
    def _nominal(self):
        """nominal wrist poses (N,2,3)/(N,2,4) and finger targets (N,2,22) at the current skill time."""
        n, dev = self.num_envs, self.device
        k = self.skill
        t = self.t_skill
        prog = torch.zeros(n, device=dev)
        roll_frac = S.roll_progress(t)
        opening, backoff, unroll = S.regrip_profile(t)
        pos, quat, fing = [], [], []
        # sighted servo: a regripping hand re-grasps the cube where it actually is (pose deviation from the ready grip,
        # modulo the cube symmetry); the blind student will have to infer this offset from touch
        if S.REGRIP_RECENTER:
            # integral servo of the holding hand: the correction accumulates the remaining cube deviation (it follows
            # the hand), from the start until the regripping hand begins to close; it fades out over the restore
            hp, hq = self._cube_deviation(perceived=True)
            t_open, t_back, t_unroll, t_appr, _ = S.REGRIP_PHASES
            fresh = (t <= self.step_dt * 0.5).unsqueeze(-1)
            self.recenter_p = torch.where(fresh, torch.zeros_like(hp), self.recenter_p)
            self.recenter_aa = torch.where(fresh, torch.zeros_like(hp), self.recenter_aa)
            live_c = (t < t_open + t_back + t_unroll + t_appr).float().unsqueeze(-1) * S.REGRIP_RECENTER_GAIN
            self.recenter_p = self.recenter_p + live_c * hp
            self.recenter_aa = self.recenter_aa + live_c * mu.axis_angle_from_quat(hq)
            w_c = (1.0 - S.smoothstep((t - sum(S.REGRIP_PHASES)) / S.REGRIP_RESTORE)).unsqueeze(-1)
            rc_p = w_c * self.recenter_p
            rc_aa = -w_c * self.recenter_aa
            rc_ang = rc_aa.norm(dim=-1)
            rc_inv = mu.quat_from_angle_axis(rc_ang, rc_aa / rc_ang.clamp(min=1e-9).unsqueeze(-1))
        if self.cfg.regrip_servo and not S.REGRIP_RECENTER:
            dev_p, dev_q = self._cube_deviation(perceived=True)
            w, live = S.regrip_servo_weight(t)
            live = live.unsqueeze(-1)
            self.servo_p = torch.where(live, dev_p, self.servo_p)
            self.servo_aa = torch.where(live, mu.axis_angle_from_quat(dev_q), self.servo_aa)
            dev_p = w.unsqueeze(-1) * self.servo_p
            aa = w.unsqueeze(-1) * self.servo_aa
            ang = aa.norm(dim=-1)
            dev_q = mu.quat_from_angle_axis(ang, aa / ang.clamp(min=1e-9).unsqueeze(-1))
        for h, side in enumerate(SIDES):
            reg = self.k_regrip[k] == h
            # roll angle in quarters: rolls move from roll0 to roll0 + delta, regrips from roll0 to 0
            q_roll = self.roll0[:, h] + self.k_roll[k, h] * roll_frac
            q_roll = torch.where(reg, self.roll0[:, h] * (1.0 - unroll), q_roll)
            self.roll[:, h] = q_roll
            nrm = self.hand_normal[h].expand(n, 3)
            dq = mu.quat_from_angle_axis(q_roll * (0.5 * math.pi), nrm)
            rp, rq, fr, _ = self._hand_ready(h, side, self.grip_mode[:, h])
            if self.alt[side] is not None:  # a regrip may close into the other grip mode: switch while unrolling
                tp, tq, tf, _ = self._hand_ready(h, side, self.grip_mode_next[:, h])
                u = (reg.float() * unroll).unsqueeze(-1)
                rp, fr = rp + u * (tp - rp), fr + u * (tf - fr)
                tq = torch.where(((rq * tq).sum(-1, keepdim=True) < 0), -tq, tq)
                rq = mu.normalize(rq + u * (tq - rq))
            p = self.c0 + mu.quat_apply(dq, rp - self.c0) + (reg.float() * backoff * S.REGRIP_BACKOFF).unsqueeze(-1) * nrm
            solo = (self.k_roll[k, h] != 0) & (self.k_roll[k, 1 - h] == 0)  # single-hand roll (R / L, not x)
            p = p + (solo.float() * S.roll_shift(t)).unsqueeze(-1) * nrm
            q = mu.quat_mul(dq, rq)
            if S.REGRIP_RECENTER:  # the holding hand carries the cube back by the inverse deviation (about c0)
                hold = ((self.k_regrip[k] >= 0) & ~reg).unsqueeze(-1)
                p = torch.where(hold, self.c0 + mu.quat_apply(rc_inv, p - self.c0 - rc_p), p)
                q = torch.where(hold, mu.quat_mul(rc_inv, q), q)
            elif self.cfg.regrip_servo:
                p_s = self.c0 + dev_p + mu.quat_apply(dev_q, p - self.c0)
                q_s = mu.quat_mul(dev_q, q)
                p = torch.where(reg.unsqueeze(-1), p_s, p)
                q = torch.where(reg.unsqueeze(-1), q_s, q)
            pos.append(p)
            quat.append(q)
            f = fr - (reg.float() * opening * S.REGRIP_OPEN).unsqueeze(-1) * self.flex_all
            # the other hand holds the cube alone meanwhile: squeeze a little harder
            other = (self.k_regrip[k] >= 0) & ~reg
            # power grip: the idle (lifted) index / pinky close onto the cube as well, the holders squeeze harder
            f = f + (other.float() * opening).unsqueeze(-1) * (self.idle_lift_vec + S.REGRIP_HOLD_PRESS * self.hold_flex)
            if S.REGRIP_RESTORE_SQUEEZE > 0:
                f = f + (reg.float() * S.regrip_loosen(t) * S.REGRIP_RESTORE_SQUEEZE).unsqueeze(-1) * self.hold_flex
            if S.REGRIP_RESTORE_LOOSEN > 0:
                f = f - (other.float() * S.regrip_loosen(t) * S.REGRIP_RESTORE_LOOSEN).unsqueeze(-1) * self.flex_all
            if self.flick_nom is not None:
                step = (t / self.step_dt).round().long().clamp(max=self.flick_nom.shape[1] - 1)
                off = self.flick_nom[k, step]  # (N, 4)
                mine = (self.k_flick_hand[k] == h).float().unsqueeze(-1)
                f = f.scatter_add(1, self.k_flick_cols[k], off * mine)
            if self.legacy_mine is not None:  # legacy flick policy: its finger targets are the nominal
                f = torch.where(self.legacy_mine.unsqueeze(-1), self.legacy_full[:, h], f)
            # single-hand roll: the rolling hand's ring lifts off (it tends to rest on the core and drag it)
            roll_w = (solo.float() * S.roll_shift(t) / max(S.ROLL_SHIFT, 1e-9)) if S.ROLL_SHIFT > 0 else solo.float() * 0.0
            f = f - (roll_w * S.ROLL_RING_LIFT).unsqueeze(-1) * self.flex["ring"]
            if S.ROLL_HOLD_PRESS > 0:  # the other hand holds the rest of the cube during a single-hand roll
                hold_solo = (self.k_roll[k, 1 - h] != 0) & (self.k_roll[k, h] == 0)
                w_h = hold_solo.float() * S.roll_shift(t) / max(S.ROLL_SHIFT, 1e-9) if S.ROLL_SHIFT > 0 else hold_solo.float()
                f = f + (w_h * S.ROLL_HOLD_PRESS).unsqueeze(-1) * self.hold_flex
            if S.FLICK_PARTNER_LIFT > 0:  # flick with this (partner) hand rolled: its digits on the U layer lift off
                partner = self.k_is_flick[k] & (self.k_flick_hand[k] != h)
                err = mu.wrap_to_pi(self.layer_target - self.cube.data.joint_pos[:, self.twist_id]).abs()
                w = partner.float() * S.smoothstep(t / S.FLICK_PARTNER_RAMP) * S.smoothstep((err - S.FLICK_PARTNER_CLOSE[0]) / S.FLICK_PARTNER_CLOSE[1])
                r = self.roll0[:, h].round().long().clamp(-1, 1) + 1
                f = f - (w * S.FLICK_PARTNER_LIFT).unsqueeze(-1) * self.on_u[h][r]
            if S.FLICK_D_LIFT > 0:  # D / D': the digits wrapping onto the D row lift off
                err = mu.wrap_to_pi(self.layer_target - self.cube.data.joint_pos[:, self.twist_id]).abs()
                w = self.k_flick_d[k].float() * S.smoothstep(t / S.FLICK_PARTNER_RAMP) * S.smoothstep((err - S.FLICK_PARTNER_CLOSE[0]) / S.FLICK_PARTNER_CLOSE[1])
                lift = sum(self.flex[d_] for d_ in S.ON_D)
                f = f - (w * S.FLICK_D_LIFT).unsqueeze(-1) * lift
            fing.append(f)
        del prog
        return torch.stack(pos, 1), torch.stack(quat, 1), torch.stack(fing, 1)

    def _blind_nominal(self):
        """nominal targets without the sighted parts (regrip servo, legacy flick policy): what a blind controller has."""
        servo, legacy = self.cfg.regrip_servo, self.legacy_mine
        self.cfg.regrip_servo, self.legacy_mine = False, None
        try:
            return self._nominal()
        finally:
            self.cfg.regrip_servo, self.legacy_mine = servo, legacy

    def _student_pre_physics(self, actions):
        """student_mode: targets = blind nominal + action x student_scale; the teacher's targets give the label."""
        sp, sr, sf = self.cfg.student_scale
        a_s = actions.clamp(-1.0, 1.0)
        warm = self.chain_warm if self.cfg.chain_train else torch.zeros_like(self.teacher_mask)
        if self.flick_legacy is not None:
            self.legacy_mine, self.legacy_full = self.flick_legacy.targets()
        # the sighted teacher system exactly as in teacher mode, with its own smoothed action (shadow)
        p_s, q_s, f_s = self._nominal()
        a_t = torch.where(warm.unsqueeze(-1), torch.zeros_like(self.teacher_raw), self.teacher_raw.clamp(-1.0, 1.0))
        self.teacher_prev = a_t
        self.teacher_smooth = self.cfg.action_ema * a_t * self.k_act_scale[self.skill] + (1.0 - self.cfg.action_ema) * self.teacher_smooth
        at = self.teacher_smooth.view(-1, 2, 28)
        p_t = p_s + at[..., 0:3] * self.cfg.act_wrist_pos
        dr = mu.quat_from_angle_axis(at[..., 3:6].norm(dim=-1).reshape(-1) * self.cfg.act_wrist_rot, at[..., 3:6].reshape(-1, 3))
        q_t = mu.quat_mul(dr, q_s.reshape(-1, 4)).view(-1, 2, 4)
        f_t = f_s + at[..., 6:] * self.cfg.act_finger
        f_t = torch.where(torch.isfinite(self.finger_override), self.finger_override, f_t)
        # blind nominal and the label in the student's action space
        p_b, q_b, f_b = self._blind_nominal()
        rot = mu.axis_angle_from_quat(mu.quat_mul(q_t.reshape(-1, 4), mu.quat_inv(q_b.reshape(-1, 4)))).view(-1, 2, 3)
        free = torch.cat([(p_t - p_b) / sp, rot / sr, (f_t - f_b) / sf], -1).clamp(-1.0, 1.0)
        self.teacher_free = torch.where(warm.unsqueeze(-1), torch.zeros_like(free.view(-1, 56)), free.view(-1, 56))
        a = torch.where(self.teacher_mask.unsqueeze(-1), self.teacher_free, a_s)
        a = torch.where(warm.unsqueeze(-1), torch.zeros_like(a), a)
        self.prev_actions[:] = self.actions
        self.actions = a
        self.student_prev = a
        a = a.view(-1, 2, 28)
        self.wrist_tgt_pos = p_b + a[..., 0:3] * sp
        drot = mu.quat_from_angle_axis(a[..., 3:6].norm(dim=-1).reshape(-1) * sr, a[..., 3:6].reshape(-1, 3)).view(-1, 2, 4)
        self.wrist_tgt_quat = mu.quat_mul(drot.reshape(-1, 4), q_b.reshape(-1, 4)).view(-1, 2, 4)
        return f_b + a[..., 6:] * sf

    def _pre_physics_step(self, actions: torch.Tensor):
        self._est_due = True
        self._new_step, self._sub = True, 0
        if self.cfg.dr_slip is not None:
            self._slip_step()
        if self.cfg.dr:
            self._held = ({s_: t_.clone() for s_, t_ in self._base_tgt.items()}, {s_: t_.clone() for s_, t_ in self._base_vel.items()},
                          self.finger_tgt.clone())
        if self.cfg.student_mode:
            f = self._student_pre_physics(actions)
            self._set_joint_targets(f)
            return
        if self.cfg.obs_act_hist > 0:
            self._act_hist = torch.cat([self.actions.unsqueeze(1), self._act_hist[:, :-1]], 1)
        self.prev_actions[:] = self.actions
        self.actions = actions.clamp(-1.0, 1.0)
        if self.cfg.chain_train:  # warm-up hold after a chain restart: zero residual
            self.actions = torch.where(self.chain_warm.unsqueeze(-1), torch.zeros_like(self.actions), self.actions)
        a = self.actions * self.k_act_scale[self.skill]
        self.act_smooth = self.cfg.action_ema * a + (1.0 - self.cfg.action_ema) * self.act_smooth
        if self.flick_legacy is not None:  # one legacy step per control step (from the state after the last step)
            self.legacy_mine, self.legacy_full = self.flick_legacy.targets()
        p_nom, q_nom, f_nom = self._nominal()
        a = self.act_smooth.view(-1, 2, 28)
        self.wrist_tgt_pos = p_nom + a[..., 0:3] * self.cfg.act_wrist_pos
        drot = mu.quat_from_angle_axis(a[..., 3:6].norm(dim=-1).reshape(-1) * self.cfg.act_wrist_rot, a[..., 3:6].reshape(-1, 3)).view(-1, 2, 4)
        self.wrist_tgt_quat = mu.quat_mul(drot.reshape(-1, 4), q_nom.reshape(-1, 4)).view(-1, 2, 4)
        f = f_nom + a[..., 6:] * self.cfg.act_finger
        f = torch.where(torch.isfinite(self.finger_override), self.finger_override, f)
        self._set_joint_targets(f)

    def _build_roll_tables(self):
        """FR3: each arm's reference configurations along its wrist roll (FR3EEController.build_roll_table), from the
        nominal roll of ready state 0 (rotation about the hand's face normal through the cube, as in the skills)."""
        rolls = torch.linspace(-1.0, 1.0, 37, device=self.device)
        c0 = self.ready["cube_pos"][0]
        for h, side in enumerate(SIDES):
            rp, rq = self.ready[f"{side}_ee_pos"][0], self.ready[f"{side}_ee_quat"][0]
            dq = mu.quat_from_angle_axis(rolls * (0.5 * math.pi), self.hand_normal[h].expand(len(rolls), 3))
            p = c0 + mu.quat_apply(dq, (rp - c0).expand(len(rolls), 3)) + self.scene.env_origins[0]
            q = mu.quat_mul(dq, rq.expand(len(rolls), 4))
            self.ctrl[side].build_roll_table(p, q, rolls)
            t = self.ctrl[side].roll_table
            print(f"[solve] FR3 {side} roll table: joint range over the roll (deg) "
                  f"{torch.rad2deg(t.max(0).values - t.min(0).values).round().int().tolist()}", flush=True)

    def _set_joint_targets(self, f):
        """finger targets (clamped) and the wrist joint targets / velocity feed-forward from self.wrist_tgt_*."""
        origins = self.scene.env_origins
        for h, side in enumerate(SIDES):
            self.finger_tgt[:, h] = torch.max(torch.min(f[:, h], self.finger_hi[side]), self.finger_lo[side])
            prev = self._base_tgt[side].clone()
            if self._fr3 is not None:  # the arm IK starts from the reference configuration at the nominal roll
                self.ctrl[side].roll_ref = self.roll[:, h]
            self._base_tgt[side] = self.ctrl[side].joint_targets(self.wrist_tgt_pos[:, h] + origins, self.wrist_tgt_quat[:, h])
            # velocity feed-forward of the wrist ("arm") trajectory: the PD then tracks without a velocity lag
            fresh = self.t_skill <= self.step_dt * 0.5  # every skill start (as at a training episode start)
            vel = (self._base_tgt[side] - prev) / self.step_dt
            self._base_vel[side] = torch.where(fresh.unsqueeze(-1), torch.zeros_like(vel), vel) * float(self.cfg.wrist_vel_ff)

    # ---------------------------------------------------------------- sim2real
    def _init_sim2real(self):
        """cfg.obs_real: measurement buffers; cfg.dr: per-env randomization (pad friction, cube mass once here)."""
        cfg, n, dev = self.cfg, self.num_envs, self.device
        self._ctrl_step, self._meas_step, self._new_step = 0, -1, False
        self._q_meas, self._qd_meas = torch.zeros(n, 2, 22, device=dev), torch.zeros(n, 2, 22, device=dev)
        self._meas_fresh = torch.ones(n, dtype=torch.bool, device=dev)
        self._q_bias = torch.zeros(n, 2, 22, device=dev)
        self._tac_g, self._tac_b = torch.ones(n, 2, 5, device=dev), torch.zeros(n, 2, 5, 3, device=dev)
        self._img_g = torch.ones(n, 10, device=dev)
        self._detent_s = torch.ones(n, 1, device=dev)
        self._act_delay = torch.zeros(n, dtype=torch.long, device=dev)
        self._sub, self._held = 0, None
        self._fbuf = None
        if cfg.dr and cfg.dr_hand_delay is not None:  # finger target ring buffer (see cfg.dr_hand_delay)
            self._fbuf = torch.zeros(cfg.dr_hand_delay[1] + 1, *self.finger_tgt.shape, device=dev)
            self._fptr, self._fbuf_fill = 0, torch.ones(n, dtype=torch.bool, device=dev)
            self._hand_delay = torch.zeros(n, dtype=torch.long, device=dev)
        self._gain0 = {s_: (h_.data.joint_stiffness.clone(), h_.data.joint_damping.clone()) for s_, h_ in self.hands.items()}
        # the fingers are an explicit (IdealPD) actuator: their gains live in the actuator model, the PhysX drives are 0
        self._fgain0 = {s_: (h_.actuators["fingers"].stiffness.clone(), h_.actuators["fingers"].damping.clone()) for s_, h_ in self.hands.items()}
        self._dr_on = torch.rand(n, device=dev) < cfg.dr_frac  # cfg.dr_frac: the other envs run nominal
        self._slip_left = torch.zeros(n, dtype=torch.long, device=dev)  # control steps left of the current pulse
        self._slip_since = torch.full((n,), 1000, dtype=torch.long, device=dev)  # control steps since a pulse
        self._slip_f, self._slip_t = torch.zeros(n, 1, 3, device=dev), torch.zeros(n, 1, 3, device=dev)
        self._tac_ring = None
        if cfg.tac_lag > 0 or cfg.tac_rate_p < 1.0:
            assert cfg.obs_real, "tac_rate_p / tac_lag act on the measured tactile (obs_real)"
            self._tac_ring, self._tac_ptr, self._img_ring = torch.zeros(cfg.tac_lag + 1, n, 2, 5, 6, device=dev), 0, None
        if not cfg.dr:
            return
        ids = torch.arange(n)
        for side, hand in self.hands.items():  # pad friction per env and pad (shape indexing as isaaclab's material event)
            view = hand.root_physx_view
            per_body = [hand._physics_sim_view.create_rigid_body_view(lp).max_shapes for lp in view.link_paths[0]]
            assert sum(per_body) == view.max_shapes
            mat = view.get_material_properties()
            for f in FINGERTIPS:
                b = hand.find_bodies(f"{side}_{f}_elastomer")[0][0]
                a0 = sum(per_body[:b])
                mu_ = torch.where(self._dr_on.cpu().unsqueeze(-1), torch.empty(n, 1).uniform_(*cfg.dr_pad_friction), 1.0)
                mat[:, a0 : a0 + per_body[b], 0] = mu_
                mat[:, a0 : a0 + per_body[b], 1] = mu_
            view.set_material_properties(mat, ids)
        view = self.cube.root_physx_view  # cube mass / inertia (uniform density)
        k = torch.where(self._dr_on.cpu().unsqueeze(-1), torch.empty(n, 1).uniform_(*cfg.dr_cube_mass), 1.0)
        view.set_masses(view.get_masses() * k, ids)
        view.set_inertias(view.get_inertias() * k.unsqueeze(-1), ids)

    def _randomize(self, env_ids):
        """per full reset: observation offsets / gains (cfg.obs_real) and dynamics / latency (cfg.dr)."""
        cfg, n, dev = self.cfg, len(env_ids), self.device
        on = self._dr_on[env_ids]

        def U(r, *sh):  # nominal envs (cfg.dr_frac): 1, or 0 for symmetric offset ranges
            v = torch.empty(n, *sh, device=dev).uniform_(*r)
            return torch.where(on.view(n, *([1] * len(sh))), v, 0.0 if r[0] == -r[1] else 1.0)
        self._meas_fresh[env_ids] = True
        self._slip_left[env_ids], self._slip_since[env_ids] = 0, 1000
        if cfg.obs_real:
            self._q_bias[env_ids] = U((-cfg.joint_pos_bias, cfg.joint_pos_bias), 2, 22)
            self._tac_g[env_ids] = U(cfg.tac_gain, 2, 5)
            self._tac_b[env_ids] = U((-cfg.tac_bias, cfg.tac_bias), 2, 5, 3)
            self._img_g[env_ids] = U(cfg.tac_img_gain, 10)
        if not cfg.dr:
            return
        for side, hand in self.hands.items():
            k0, d0 = self._gain0[side]
            k, d = k0[env_ids].clone(), d0[env_ids].clone()
            fa, (fk0, fd0) = hand.actuators["fingers"], self._fgain0[side]  # (before 10-27 this scaled the zero PhysX gains)
            fa.stiffness[env_ids] = fk0[env_ids] * U(cfg.dr_finger_gain, 1)
            fa.damping[env_ids] = fd0[env_ids] * U(cfg.dr_finger_gain, 1)
            if self._fr3 is not None:
                ai = self.ctrl[side].joint_ids
                g = U(cfg.dr_arm_gain, 1)
                k[:, ai] *= g
                d[:, ai] *= g.sqrt()  # the damping ratio stays
            hand.write_joint_stiffness_to_sim(k, env_ids=env_ids)
            hand.write_joint_damping_to_sim(d, env_ids=env_ids)
        if cfg.use_joint_friction:
            f = U(cfg.dr_twist_friction, 1)
            self.cube.write_joint_friction_coefficient_to_sim(cfg.twist_static_friction * f, cfg.twist_dynamic_friction * f,
                                                             cfg.twist_viscous_friction * f, joint_ids=[self.twist_id], env_ids=env_ids)
            if self.mid_id is not None:  # the same cube tension on both layers
                f = f * cfg.mid_friction_scale
                self.cube.write_joint_friction_coefficient_to_sim(cfg.twist_static_friction * f, cfg.twist_dynamic_friction * f,
                                                                 cfg.twist_viscous_friction * f, joint_ids=[self.mid_id], env_ids=env_ids)
        self._detent_s[env_ids] = U(cfg.dr_detent, 1)
        self._act_delay[env_ids] = torch.randint(cfg.dr_act_delay[0], cfg.dr_act_delay[1] + 1, (n,), device=dev) * on.view(n)
        if self._fbuf is not None:
            sc = min(1.0, self.common_step_counter / cfg.dr_hand_delay_ramp) if cfg.dr_hand_delay_ramp > 0 else 1.0
            lo, hi = round(cfg.dr_hand_delay[0] * sc), round(cfg.dr_hand_delay[1] * sc)
            self._hand_delay[env_ids] = torch.randint(lo, hi + 1, (n,), device=dev) * on.view(n)
            self._fbuf_fill[env_ids] = True  # a reset jumps the targets: no stale pre-reset ones

    def _measure(self):
        """cfg.obs_real: finger joint angles (noise + encoder offset), their finite-difference velocities and the pad
        F/T (gain + offset) as the real robot reads them, once per control step (repeated reads return the same)."""
        if self._meas_step == self._ctrl_step:
            return
        self._meas_step = self._ctrl_step
        for h, side in enumerate(SIDES):
            q = self.hands[side].data.joint_pos[:, self.finger_ids[side]]
            qm = q + self._q_bias[:, h] + self.cfg.joint_pos_noise * torch.randn_like(q) * self._dr_on.unsqueeze(-1)
            prev = torch.where(self._meas_fresh.unsqueeze(-1), qm, self._q_meas[:, h])
            self._qd_meas[:, h] = (qm - prev) / self.step_dt
            self._q_meas[:, h] = qm
        fresh = self._meas_fresh.clone()
        self._meas_fresh[:] = False
        ft = self.tactile.ft
        self._ft_meas = torch.cat([ft[..., :3] * self._tac_g.unsqueeze(-1) + self._tac_b, ft[..., 3:] * self._tac_g.unsqueeze(-1)], -1)
        if self._tac_ring is not None:
            self._tac_deliver(fresh)

    def _tac_deliver(self, fresh):
        """cfg.tac_rate_p / tac_lag: the pad F/T (and images) the robot has at this control step: tac_lag steps old,
        refreshed with probability tac_rate_p per step, held in between (a reset fills the history with the present)."""
        cfg, n = self.cfg, self.num_envs
        L = self._tac_ring.shape[0]
        img = None if self.tactile.image is None else self.tactile.image.reshape(n, 10, 60, 60)
        if img is not None and self._img_ring is None:
            self._img_ring = torch.zeros(L, *img.shape, dtype=img.dtype, device=img.device)
            self._img_held = img.clone()
        self._tac_ptr = (self._tac_ptr + 1) % L
        self._tac_ring[self._tac_ptr] = self._ft_meas
        self._tac_ring[:, fresh] = self._ft_meas[fresh]
        if img is not None:
            self._img_ring[self._tac_ptr] = img
            self._img_ring[:, fresh] = img[fresh]
        old = (self._tac_ptr - cfg.tac_lag) % L
        new = (torch.rand(n, device=self.device) < cfg.tac_rate_p) | fresh
        if not hasattr(self, "_ft_held"):
            self._ft_held = self._ft_meas.clone()
        self._ft_held = torch.where(new.view(n, 1, 1, 1), self._tac_ring[old], self._ft_held)
        self._ft_meas = self._ft_held
        if img is not None:
            self._img_held = torch.where(new.view(n, 1, 1, 1), self._img_ring[old], self._img_held)

    def _slip_step(self):
        """cfg.dr_slip: start / hold / end random wrench pulses on the cube core (once per control step)."""
        rate, fmax, tmax, (d0, d1) = self.cfg.dr_slip
        n, dev, dt = self.num_envs, self.device, self.step_dt
        act = self._slip_left > 0
        self._slip_left -= act.long()
        self._slip_since = torch.where(act, 0, self._slip_since + 1)
        start = (self._slip_left == 0) & (self.t_skill > 0.25) & (torch.rand(n, device=dev) < rate * dt)
        k = int(start.sum())
        if k:
            nrm = torch.nn.functional.normalize
            self._slip_f[start, 0] = nrm(torch.randn(k, 3, device=dev), dim=-1) * torch.rand(k, 1, device=dev) * fmax
            self._slip_t[start, 0] = nrm(torch.randn(k, 3, device=dev), dim=-1) * torch.rand(k, 1, device=dev) * tmax
            self._slip_left[start] = torch.randint(max(1, round(d0 / dt)), max(1, round(d1 / dt)) + 1, (k,), device=dev)
        on = (self._slip_left > 0).view(n, 1, 1).float()
        self.cube.set_external_force_and_torque(self._slip_f * on, self._slip_t * on, body_ids=[self.core_id])

    @property
    def slip_recent(self):
        """cfg.dr_slip: a pulse is on or ended less than 0.5 s ago"""
        return (self._slip_left > 0) | (self._slip_since < round(0.5 / self.step_dt))

    def meas_q(self, h):
        if not self.cfg.obs_real:
            return self.hands[SIDES[h]].data.joint_pos[:, self.finger_ids[SIDES[h]]]
        self._measure()
        return self._q_meas[:, h]

    def meas_qd(self, h):
        if not self.cfg.obs_real:
            return self.hands[SIDES[h]].data.joint_vel[:, self.finger_ids[SIDES[h]]]
        self._measure()
        return self._qd_meas[:, h]

    def meas_ft(self, h):
        if not self.cfg.obs_real:
            return self.tactile.ft[:, h]
        self._measure()
        return self._ft_meas[:, h]

    def _apply_action(self):
        if self._new_step:  # first physics step of this control step: later sensor reads are new measurements
            self._new_step = False
            self._ctrl_step += 1
        late = None
        if self.cfg.dr and self._held is not None:  # command latency: the previous targets until the delay has passed
            late = (self._sub < self._act_delay).unsqueeze(-1)
            self._sub += 1
        f_tgt = self.finger_tgt
        if self._fbuf is not None:  # hand latency: the finger targets of _hand_delay physics steps ago
            L = self._fbuf.shape[0]
            if self._fbuf_fill.any():
                self._fbuf[:, self._fbuf_fill] = f_tgt[self._fbuf_fill]
                self._fbuf_fill[:] = False
            self._fptr = (self._fptr + 1) % L
            self._fbuf[self._fptr] = f_tgt
            f_tgt = self._fbuf[(self._fptr - self._hand_delay) % L, torch.arange(f_tgt.shape[0], device=f_tgt.device)]
        for h, (side, hand) in enumerate(self.hands.items()):
            bt, bv, ft_ = self._base_tgt[side], self._base_vel[side], f_tgt[:, h]
            if late is not None:
                bt = torch.where(late, self._held[0][side], bt)
                bv = torch.where(late, self._held[1][side], bv)
                if self._fbuf is None:
                    ft_ = torch.where(late, self._held[2][:, h], ft_)
            hand.set_joint_position_target(bt, joint_ids=self.ctrl[side].joint_ids)
            hand.set_joint_velocity_target(bv, joint_ids=self.ctrl[side].joint_ids)
            hand.set_joint_position_target(ft_, joint_ids=self.finger_ids[side])
        self.tactile.step()
        tau = self.extra_twist_effort.unsqueeze(-1).clone()
        if not self.cfg.use_joint_friction:
            omega = self.cube.data.joint_vel[:, self.twist_id : self.twist_id + 1]
            tau -= 0.01 * torch.tanh(omega / 0.05)
        if self.cfg.detent_torque > 0:
            # magnetic positioning of a speedcube: a restoring torque only within detent_width of a quarter turn
            q = self.cube.data.joint_pos[:, self.twist_id : self.twist_id + 1]
            e = q - torch.round(q / (0.5 * math.pi)) * (0.5 * math.pi)
            w = self.cfg.detent_width
            tau -= torch.where(e.abs() < w, self.cfg.detent_torque * self._detent_s * torch.sin(math.pi * e / w), torch.zeros_like(e))
        self.cube.set_joint_effort_target(tau, joint_ids=[self.twist_id])
        if self.mid_id is not None and self.cfg.detent_torque > 0:
            q = self.cube.data.joint_pos[:, self.mid_id : self.mid_id + 1]
            e = q - torch.round(q / (0.5 * math.pi)) * (0.5 * math.pi)
            w = self.cfg.detent_width
            tm = -torch.where(e.abs() < w, self.cfg.detent_torque * self._detent_s * torch.sin(math.pi * e / w), torch.zeros_like(e))
            self.cube.set_joint_effort_target(tm, joint_ids=[self.mid_id])

    # -------------------------------------------------------- observations
    def cube_state_true(self):
        """(N, 17) the simulator's cube state in the teacher's units: position off c0 x20, rotation off the skill goal
        (rot6), layer error to the target, core linear / 0.1 x angular velocity, 0.1 x layer velocity (estimator label)."""
        d = self.cube.data
        pos = d.root_pos_w - self.scene.env_origins - self.c0
        rel = mu.quat_mul(self._logical_quat(), mu.quat_inv(self.cube_goal))
        rot6 = mu.matrix_from_quat(rel)[..., :2].reshape(-1, 6)
        err = mu.wrap_to_pi(self.layer_target - d.joint_pos[:, self.twist_id])
        return torch.cat([pos * 20.0, rot6, err.unsqueeze(-1), d.body_lin_vel_w[:, self.core_id], 0.1 * d.body_ang_vel_w[:, self.core_id],
                          0.1 * d.joint_vel[:, self.twist_id : self.twist_id + 1]], -1)

    def perceived(self):
        """the cube state the controllers act on: the simulator's, or with cfg.cube_est the estimate self.est_cube.
        dict: pos (off c0), lq (logical quat), twist, twist_vel, lin_vel, ang_vel. With cfg.cube_vis: the camera's."""
        if self._vis is not None and self.cfg.vis_control and not self._vis_truth:
            v = self._vis
            return dict(pos=v["pos"], lq=v["lq"], twist=v["tw"], twist_vel=v["tw_vel"], lin_vel=v["lin_vel"], ang_vel=v["ang_vel"])
        if not self.cfg.cube_est:
            d = self.cube.data
            return dict(pos=d.root_pos_w - self.scene.env_origins - self.c0, lq=self._logical_quat(), twist=d.joint_pos[:, self.twist_id],
                        twist_vel=d.joint_vel[:, self.twist_id], lin_vel=d.body_lin_vel_w[:, self.core_id], ang_vel=d.body_ang_vel_w[:, self.core_id])
        from .estimator import rot6_to_matrix

        e = self.est_cube
        lq = mu.quat_mul(mu.quat_from_matrix(rot6_to_matrix(e[:, 3:9])), self.cube_goal)
        return dict(pos=e[:, 0:3] / 20.0, lq=lq, twist=self.layer_target - e[:, 9], twist_vel=10.0 * e[:, 16],
                    lin_vel=e[:, 10:13], ang_vel=10.0 * e[:, 13:16])

    def _init_vision(self):
        """cfg.cube_vis: buffers of the camera measurement model (see env_cfg)."""
        cfg, n, dev = self.cfg, self.num_envs, self.device
        self._vis, self._vis_truth = None, False
        if not cfg.cube_vis:
            return
        assert not (cfg.cube_est or cfg.cube_est_ckpt), "cube_vis and the blind estimator are alternatives"
        assert not cfg.handoff_required, "the camera completion has no handoff criterion"
        L = cfg.vis_lag + 1
        self._vis = dict(r_pos=torch.zeros(L, n, 3, device=dev), r_lq=torch.zeros(L, n, 4, device=dev), r_tw=torch.zeros(L, n, device=dev),
                         pos=torch.zeros(n, 3, device=dev), lq=torch.zeros(n, 4, device=dev), tw=torch.zeros(n, device=dev),
                         lin_vel=torch.zeros(n, 3, device=dev), ang_vel=torch.zeros(n, 3, device=dev), tw_vel=torch.zeros(n, device=dev),
                         age=torch.zeros(n, device=dev), occl=torch.zeros(n, dtype=torch.bool, device=dev),
                         fresh=torch.ones(n, dtype=torch.bool, device=dev), b_pos=torch.zeros(n, 3, device=dev),
                         b_rot=torch.zeros(n, 3, device=dev), b_tw=torch.zeros(n, device=dev), ptr=0)
        self._vis["lq"][:, 0] = 1.0
        self.stable_vis = torch.zeros(n, dtype=torch.long, device=dev)
        self.success_vis = torch.zeros(n, dtype=torch.bool, device=dev)
        self.win_vis = (torch.zeros(n, 3, device=dev), torch.zeros(n, 4, device=dev), torch.zeros(n, device=dev))

    def _vis_reset(self, env_ids):
        """full reset: new per-episode bias, the camera history filled with the present state at the next update."""
        v, k = self._vis, len(env_ids)
        bp, br, bt = self.cfg.vis_bias
        u = lambda *sh: 2.0 * torch.rand(k, *sh, device=self.device) - 1.0  # noqa: E731
        v["b_pos"][env_ids], v["b_rot"][env_ids], v["b_tw"][env_ids] = bp * u(3), math.radians(br) * u(3), math.radians(bt) * u()
        v["fresh"][env_ids] = True

    def _vis_frame(self, pos, lq, tw, ids=None):
        """what the camera reports for these true states: bias + white noise."""
        v, (sp, sr, st) = self._vis, self.cfg.vis_noise
        sl = slice(None) if ids is None else ids
        rv = v["b_rot"][sl] + math.radians(sr) * torch.randn_like(pos)
        ang = rv.norm(dim=-1).clamp(min=1e-9)
        lq_m = mu.quat_mul(mu.quat_from_angle_axis(ang, rv / ang.unsqueeze(-1)), lq)
        return pos + v["b_pos"][sl] + sp * torch.randn_like(pos), lq_m, tw + v["b_tw"][sl] + math.radians(st) * torch.randn_like(tw)

    def _vis_update(self):
        """once per control step: record the true state; the tracker (alpha-beta filter, cfg.vis_filter) predicts and,
        where a frame arrives, corrects with that frame, captured vis_lag control steps ago."""
        cfg, v, d, n, dt = self.cfg, self._vis, self.cube.data, self.num_envs, self.step_dt
        pos = d.root_pos_w - self.scene.env_origins - self.c0
        lq, tw = self._logical_quat(), d.joint_pos[:, self.twist_id]
        L = v["r_pos"].shape[0]
        v["ptr"] = (v["ptr"] + 1) % L
        f = v["fresh"]
        for key, x in (("r_pos", pos), ("r_lq", lq), ("r_tw", tw)):
            v[key][v["ptr"]] = x
            v[key][:, f] = x[f]
        p_in, p_out = cfg.vis_occlusion
        r = torch.rand(n, device=self.device)
        v["occl"] = torch.where(v["occl"], r >= p_out, r < p_in) & ~f
        new = ((torch.rand(n, device=self.device) < cfg.vis_rate_p) & ~v["occl"]) | f
        old = (v["ptr"] - cfg.vis_lag) % L
        mp, mq, mt = self._vis_frame(v["r_pos"][old], v["r_lq"][old], v["r_tw"][old])
        # predict
        v["pos"] = v["pos"] + v["lin_vel"] * dt
        v["tw"] = v["tw"] + v["tw_vel"] * dt
        w = v["ang_vel"] * dt
        wa = w.norm(dim=-1).clamp(min=1e-9)
        v["lq"] = mu.quat_mul(mu.quat_from_angle_axis(wa, w / wa.unsqueeze(-1)), v["lq"])
        v["age"] += dt
        # correct where a frame arrived
        al, be = cfg.vis_filter
        T = v["age"].unsqueeze(-1)
        rp = mp - v["pos"]
        rq = mu.axis_angle_from_quat(mu.quat_mul(mq, mu.quat_inv(v["lq"])))
        rt = mu.wrap_to_pi(mt - v["tw"])
        ra = (al * rq).norm(dim=-1).clamp(min=1e-9)
        cor = dict(pos=v["pos"] + al * rp, lq=mu.quat_mul(mu.quat_from_angle_axis(ra, al * rq / ra.unsqueeze(-1)), v["lq"]), tw=v["tw"] + al * rt,
                   lin_vel=v["lin_vel"] + be * rp / T, ang_vel=v["ang_vel"] + be * rq / T, tw_vel=v["tw_vel"] + be * rt / T.squeeze(-1))
        first = dict(pos=mp, lq=mq, tw=mt, lin_vel=torch.zeros_like(mp), ang_vel=torch.zeros_like(mp), tw_vel=torch.zeros_like(mt))
        for key in cor:
            sh = (-1,) + (1,) * (cor[key].dim() - 1)
            v[key] = torch.where(f.view(sh), first[key], torch.where(new.view(sh), cor[key], v[key]))
        v["age"] = torch.where(new, torch.zeros_like(v["age"]), v["age"])
        v["fresh"][:] = False

    def _vis_rebind(self, env_ids):
        """the free layer changed (or snapped): the camera sees the new layer at once, its history re-expressed."""
        v = self._vis
        tw = self.cube.data.joint_pos[env_ids, self.twist_id]
        v["r_tw"][:, env_ids] = tw
        v["tw"][env_ids] = self._vis_frame(v["pos"][env_ids], v["lq"][env_ids], tw, env_ids)[2]
        v["tw_vel"][env_ids] = 0.0

    def _vis_ready(self, done_nominal, hands_ok):
        """env.success_vis: the completion criteria on what the robot measures: the cube's on the camera tracker (with
        noise-aware quiet thresholds cfg.vis_quiet and a fresh frame, age <= cfg.vis_max_age), holding / release on the
        delivered pad forces (cfg.vis_touch); nominal time and hand posture as before (proprioception)."""
        cfg, v, n = self.cfg, self._vis, self.num_envs
        pos, lq, twist = v["pos"] + self.c0, v["lq"], v["tw"]
        layer_ok = mu.wrap_to_pi(self.layer_target - twist).abs() < cfg.layer_tol
        rot_ok = mu.quat_error_magnitude(lq, self.cube_goal) < cfg.cube_rot_goal_tol
        cube_ok = layer_ok & torch.where(self.k_is_x[self.skill], rot_ok, torch.ones_like(rot_ok))
        Rl = mu.matrix_from_quat(self.ready["cube_quat"][self.ready_idx]).transpose(-1, -2) @ mu.matrix_from_quat(lq)
        tr = (Rl.unsqueeze(1) @ self.group.transpose(-1, -2).unsqueeze(0)).diagonal(dim1=-2, dim2=-1).sum(-1).amax(-1)
        tilt_ok = torch.acos(((tr - 1.0) / 2.0).clamp(-1.0, 1.0)) < cfg.tilt_tol
        wp, wq, wt = self.win_vis
        begin = (self.stable_vis == 0).unsqueeze(-1)
        wp, wq, wt = torch.where(begin, pos, wp), torch.where(begin, lq, wq), torch.where(begin.squeeze(-1), twist, wt)
        self.win_vis = (wp, wq, wt)
        qp, qr, ql = cfg.vis_quiet
        quiet = ((pos - wp).norm(dim=-1) < qp) & (mu.quat_error_magnitude(lq, wq) < math.radians(qr)) & ((twist - wt).abs() < math.radians(ql))
        fresh = v["age"] <= cfg.vis_max_age
        # touch: delivered pad forces (N, 5) per hand
        fn = torch.stack([self.meas_ft(h)[..., :3].norm(dim=-1) for h in range(2)], 1)
        t_hold, t_rel = cfg.vis_touch
        holding = ((fn > t_hold).sum(-1) >= cfg.min_pads).all(-1)
        rows = torch.arange(n, device=self.device)
        h_fl = self.k_flick_hand[self.skill].clamp(min=0)
        released = ~self.k_is_flick[self.skill] | (fn[rows, h_fl, self.k_flick_pad[self.skill]] < t_rel)
        ready = cube_ok & done_nominal & hands_ok & released & holding & quiet & tilt_ok & fresh & ~self.dropped
        self.crit_vis = {"cube": cube_ok, "released": released, "holding": holding, "quiet": quiet, "tilt": tilt_ok, "fresh": fresh}
        self.stable_vis = torch.where(ready, self.stable_vis + 1, torch.zeros_like(self.stable_vis))
        self.success_vis |= self.stable_vis >= max(1, round(cfg.stable_s / self.step_dt))

    def _cube_obs(self):
        per = self.perceived()
        rel = mu.quat_mul(per["lq"], mu.quat_inv(self.cube_goal))
        rot6 = mu.matrix_from_quat(rel)[..., :2].reshape(-1, 6)
        twist = per["twist"]
        err = mu.wrap_to_pi(self.layer_target - twist)
        return torch.cat([per["pos"] * 20.0, rot6, torch.stack([err, twist.sin(), twist.cos()], -1),
                          per["lin_vel"], 0.1 * per["ang_vel"], 0.1 * per["twist_vel"].unsqueeze(-1),
                          (self.t_skill > 0).float().unsqueeze(-1)], -1)

    def _est_obs(self):
        """the estimator's input, sensing only: finger joint angles / velocities and delivered pad F/T per hand, the
        commanded skill / phase / rolls / next skill (no targets, tracking errors or actions: in sighted rollouts these
        carry the controllers' knowledge of the cube)."""
        parts = []
        for h, side in enumerate(SIDES):
            ft = self.meas_ft(h)
            parts += [self.meas_q(h), 0.1 * self.meas_qd(h), torch.cat([ft[..., :3] * 0.2, ft[..., 3:] * 20.0], -1).flatten(1)]
        nxt = self.next_skill
        parts += [self._common(), torch.nn.functional.one_hot(nxt.clamp(min=0), self._n_skill).float() * (nxt >= 0).float().unsqueeze(-1)]
        return torch.nan_to_num(torch.cat(parts, -1)).clamp(-10.0, 10.0)

    def _proprio(self, h, side):
        q = self.meas_q(h)
        p, quat = self.ctrl[side].ee_pose_w()
        dp = (p - self.scene.env_origins - self.wrist_tgt_pos[:, h]) * 50.0
        dr = mu.axis_angle_from_quat(mu.quat_mul(quat, mu.quat_inv(self.wrist_tgt_quat[:, h])))
        ft = self.meas_ft(h)
        ft = torch.cat([ft[..., :3] * 0.2, ft[..., 3:] * 20.0], -1).flatten(1)
        return torch.cat([q, 0.1 * self.meas_qd(h), self.finger_tgt[:, h], dp, dr, ft], -1)

    def _common(self):
        k1 = torch.nn.functional.one_hot(self.skill, self._n_skill).float()
        dur = self.k_duration[self.skill].clamp(min=1e-3)
        # skill time over the TRAINING episode length (cfg.phase_horizon_s), also in chains whose episodes never end
        phase = torch.stack([(self.t_skill / dur).clamp(max=2.0), (self.t_skill / self.cfg.phase_horizon_s)], -1)
        return torch.cat([k1, phase, self.roll], -1)

    def _get_observations(self) -> dict:
        prop = [self._proprio(h, s) for h, s in enumerate(SIDES)]
        com = self._common()
        # previous action: the teacher sees its own (raw) one, the student its executed one (student_mode)
        tail_t = [self.teacher_prev if self.cfg.student_mode else self.actions]
        tail_s = [self.student_prev if self.cfg.student_mode else self.actions]
        if self.cfg.obs_next_skill:  # at the end: older checkpoints extend by zero input columns (mix_policy)
            nxt = self.next_skill
            nx = torch.nn.functional.one_hot(nxt.clamp(min=0), self._n_skill).float() * (nxt >= 0).float().unsqueeze(-1)
            tail_t.append(nx)
            tail_s.append(nx)
        if self.cfg.obs_act_hist > 0:
            tail_t.append(self._act_hist.flatten(1))
            tail_s.append(self._act_hist.flatten(1))
        student = torch.nan_to_num(torch.cat(prop + [com] + tail_s, -1)).clamp(-10.0, 10.0)
        est_obs = self._est_obs()
        ee = []  # actual wrist poses (forward kinematics) off the grip center: x20, rot6
        for h, side in enumerate(SIDES):
            p, q = self.ctrl[side].ee_pose_w()
            ee += [(p - self.scene.env_origins - self.c0) * 20.0, mu.matrix_from_quat(q)[..., :2].reshape(-1, 6)]
        out = {"student": student, "est": est_obs, "student_ee": torch.cat([student] + ee, -1)}
        if self.tactile.image is not None:  # delivered 60x60 TacMap images, 2x2 average-pooled: (N, 10 pads, 30, 30) uint8
            src = self._img_held if self._tac_ring is not None and self._img_ring is not None else self.tactile.image
            img = src.reshape(self.num_envs * 10, 1, 60, 60).float()
            if self.cfg.obs_real:  # per-pad intensity gain
                img = (img * self._img_g.view(-1, 1, 1, 1)).clamp(0.0, 255.0)
            out["tac_img"] = torch.nn.functional.avg_pool2d(img, 2).round().to(torch.uint8).view(self.num_envs, 10, 30, 30)
        if self.estimator is not None:
            if self._est_due:  # a new control step (chains restarted after a failure start from a fresh hidden state)
                self._est_due = False
                self.estimator.reset(self.reset_terminated.nonzero().squeeze(-1))
                self.est_cube = self.estimator.step(out)
            self.cfg.cube_est = False  # the critic sees the simulator's cube state
            out["critic"] = torch.nan_to_num(torch.cat(prop + [self._cube_obs(), com] + tail_t, -1)).clamp(-10.0, 10.0)
            self.cfg.cube_est = True
        if self._vis is not None:  # the critic sees the simulator's cube state
            self._vis_truth = True
            out["critic"] = torch.nan_to_num(torch.cat(prop + [self._cube_obs(), com] + tail_t, -1)).clamp(-10.0, 10.0)
            self._vis_truth = False
        teacher = torch.cat(prop[:1] + prop[1:] + [self._cube_obs(), com] + tail_t, -1)
        out["policy"] = torch.nan_to_num(teacher).clamp(-10.0, 10.0)
        return out

    # ------------------------------------------------------------ task state
    def _task_error(self, env_ids=None):
        """skill-specific scalar error [rad]: layer error for face turns, orientation error for x, layer drift else."""
        d = self.cube.data
        twist = d.joint_pos[:, self.twist_id]
        layer_err = mu.wrap_to_pi(self.layer_target - twist).abs()
        rot_err = mu.quat_error_magnitude(self._logical_quat(), self.cube_goal)
        err = torch.where(self.k_is_x[self.skill], rot_err + layer_err, layer_err)
        return err if env_ids is None else err[env_ids]

    def _update_state(self):
        cfg = self.cfg
        d = self.cube.data
        n = self.num_envs
        rows = torch.arange(n, device=self.device)
        if self._vis is not None:
            self._vis_update()
        pos = d.root_pos_w - self.scene.env_origins
        lq = self._logical_quat()
        twist = d.joint_pos[:, self.twist_id]
        self.err = self._task_error()
        layer_err = mu.wrap_to_pi(self.layer_target - twist).abs()
        self.peak_pos = torch.max(self.peak_pos, (pos - self.c0).norm(dim=-1))
        # deviation from the planned orientation: the x motion itself is allowed (bounded by progress between ref/goal)
        dev_ref = mu.quat_error_magnitude(lq, self.cube_ref)
        dev_goal = mu.quat_error_magnitude(lq, self.cube_goal)
        self.rot_dev = torch.where(self.k_is_x[self.skill], (dev_ref + dev_goal - 0.5 * math.pi).clamp(min=0.0), dev_ref)
        self.peak_rot = torch.max(self.peak_rot, self.rot_dev)
        # a regrip may let the cube sway in the single holding hand: the close + restore bring it back, the end state counts
        reg = self.k_is_regrip[self.skill]
        pos_b = torch.where(reg, cfg.regrip_pos_bound, cfg.cube_pos_bound)
        rot_b = torch.where(reg, cfg.regrip_rot_bound, cfg.cube_rot_bound)
        self.violation = (self.peak_pos > pos_b) | (self.peak_rot > rot_b)
        # tilt: deviation from the ready grip's cube orientation, modulo the cube's symmetry
        Rl = mu.matrix_from_quat(self.ready["cube_quat"][self.ready_idx]).transpose(-1, -2) @ mu.matrix_from_quat(lq)
        cand = Rl.unsqueeze(1) @ self.group.transpose(-1, -2).unsqueeze(0)
        tr = cand.diagonal(dim1=-2, dim2=-1).sum(-1).amax(-1)
        self.tilt = torch.acos(((tr - 1.0) / 2.0).clamp(-1.0, 1.0))
        g = torch.tensor(cfg.sim.gravity, device=self.device)
        self.dropped = ((pos - self.c0) @ (g / g.norm())) > cfg.drop_height  # fallen along gravity (any direction)

        # completion conditions
        # settled layer: angle within tolerance and its excursion over the stability window bounded (an instantaneous
        # speed gate fails on sub-degree contact jitter)
        layer_ok = layer_err < cfg.layer_tol
        rot_ok = dev_goal < cfg.cube_rot_goal_tol
        cube_ok = layer_ok & torch.where(self.k_is_x[self.skill], rot_ok, torch.ones_like(rot_ok))
        # nominal motion finished and tracked
        done_nominal = self.t_skill >= self.k_duration[self.skill]
        p_nom, q_nom, f_nom = self._nominal_end()
        hands_ok = torch.ones(n, dtype=torch.bool, device=self.device)
        handoff_hands = torch.ones(n, dtype=torch.bool, device=self.device)
        self.finger_err = torch.zeros(n, device=self.device)
        for h, side in enumerate(SIDES):
            p, q = self.ctrl[side].ee_pose_w()
            dp = (p - self.scene.env_origins - p_nom[:, h]).norm(dim=-1)
            dr = mu.quat_error_magnitude(q, q_nom[:, h])
            hands_ok &= (dp < cfg.wrist_track_tol[0]) & (dr < cfg.wrist_track_tol[1])
            handoff_hands &= (dp < cfg.handoff_wrist_tol[0]) & (dr < cfg.handoff_wrist_tol[1])
            qf = self.hands[side].data.joint_pos[:, self.finger_ids[side]]
            qd = self.hands[side].data.joint_vel[:, self.finger_ids[side]]
            # against the settled ready posture (targets differ from the actual grasp by the squeeze offset)
            e = (qf - self._hand_ready(h, side, self.grip_mode_next[:, h])[3]).abs().max(-1).values
            v = qd.abs().max(-1).values
            # all fingers of a regripping or flicking hand must be back at the nominal ready targets
            active = (self.k_regrip[self.skill] == h) | (self.k_flick_hand[self.skill] == h)
            self.finger_err = torch.max(self.finger_err, torch.where(active, e, torch.zeros_like(e)))
            tol = torch.where(self.k_regrip[self.skill] == h, cfg.regrip_finger_tol, cfg.finger_ready_tol)
            hands_ok &= ~active | ((e < tol) & (v < cfg.finger_speed_tol))
            handoff_hands &= e < cfg.handoff_finger_tol
            self.hand_dbg[:, h] = torch.stack([dp * 1000, torch.rad2deg(dr), e, v], -1)  # wrist mm, deg, finger rad, rad/s
        # flick finger released, every hand touching with at least one pad
        h_fl = self.k_flick_hand[self.skill].clamp(min=0)
        if self.cube_contacts is not None:
            cf = self.cube_contacts.forces()  # (N, 2 bodies, 2 hands, 27)
            trick_f = (cf.sum(1)[rows, h_fl] * self.k_flick_links[self.skill]).sum(-1)
        else:  # trick finger's pad only (rigid phalanx contacts are not sensed then)
            trick_f = self.tactile.layer_forces()[rows, h_fl, self.k_flick_pad[self.skill]].sum(-1)
        released = ~self.k_is_flick[self.skill] | (trick_f < cfg.release_force)
        lf = self.tactile.layer_forces()  # (N, 2 hands, 5 pads, [core, free layer])
        pads = (lf.sum(-1) > cfg.pad_threshold).sum(-1)  # (N, 2)
        self.trick_on_layer = self.k_is_flick[self.skill] & (lf[rows, h_fl, self.k_flick_pad[self.skill], 1] > cfg.pad_threshold)
        holding = (pads >= cfg.min_pads).all(-1)
        begin = self.stable_steps == 0
        self.win_pos = torch.where(begin.unsqueeze(-1), pos, self.win_pos)
        self.win_quat = torch.where(begin.unsqueeze(-1), lq, self.win_quat)
        self.win_twist = torch.where(begin, twist, self.win_twist)
        quiet = ((pos - self.win_pos).norm(dim=-1) < cfg.stable_pos) & (mu.quat_error_magnitude(lq, self.win_quat) < cfg.stable_rot)
        quiet &= (twist - self.win_twist).abs() < cfg.stable_layer
        tilt_ok = self.tilt < cfg.tilt_tol
        # handoff-ready: the cube back at its ready-grip pose (modulo symmetry), layer aligned, hands at the ready posture
        dev_p, dev_q = self._cube_deviation()
        self.handoff_pos = dev_p.norm(dim=-1)
        self.handoff_rot = 2.0 * torch.acos(dev_q[:, 0].abs().clamp(max=1.0))
        frac = min(1.0, self.common_step_counter / cfg.handoff_anneal_steps) if cfg.handoff_anneal_steps > 0 else 1.0
        tol_p = cfg.handoff_start[0] + frac * (cfg.handoff_pos_tol - cfg.handoff_start[0])
        tol_r = cfg.handoff_start[1] + frac * (cfg.handoff_rot_tol - cfg.handoff_start[1])
        handoff_ok = (self.handoff_pos < tol_p) & (self.handoff_rot < tol_r) & handoff_hands
        handoff_ok &= quarter_residual(twist).abs() < cfg.handoff_layer_tol
        ready = cube_ok & done_nominal & hands_ok & released & holding & quiet & ~self.violation & ~self.dropped & tilt_ok
        if cfg.handoff_required:
            ready &= handoff_ok
        self.crit = {"cube": cube_ok, "nominal": done_nominal, "hands": hands_ok, "released": released,
                     "holding": holding, "quiet": quiet, "tilt": tilt_ok, "handoff": handoff_ok}
        if self._vis is not None:
            self._vis_ready(done_nominal, hands_ok)
        self.stable_steps = torch.where(ready, self.stable_steps + 1, torch.zeros_like(self.stable_steps))
        need = max(1, round(cfg.stable_s / self.step_dt))
        self.success |= self.stable_steps >= need
        self.layer_err, self.trick_force, self.holding = layer_err, trick_f, holding

    def _handoff_penalty(self):
        """-rew_handoff x (cube offset / pos tol + tilt / rot tol) from the ready-grip pose; an x skill only once its
        nominal motion has ended (the whole cube turns through 45 deg from every symmetric pose)."""
        cfg = self.cfg
        if cfg.rew_handoff == 0.0:
            return 0.0
        dev = (self.handoff_pos / cfg.handoff_pos_tol).clamp(max=5.0) + (self.handoff_rot / cfg.handoff_rot_tol).clamp(max=5.0)
        live = ~self.k_is_x[self.skill] | (self.t_skill >= self.k_duration[self.skill])
        return -cfg.rew_handoff * dev * live.float()

    def _nominal_end(self):
        """nominal targets at the END of the current skill."""
        t = self.t_skill
        self.t_skill = torch.full_like(t, 100.0)
        roll = self.roll.clone()
        out = self._nominal()
        self.t_skill, self.roll = t, roll
        return out

    def _get_dones(self):
        self.t_skill += self.step_dt
        self._update_state()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        broken = ~torch.isfinite(self.cube.data.root_state_w).all(-1)
        for hand in self.hands.values():
            broken |= ~torch.isfinite(hand.data.joint_pos).all(-1)
        self.broken = broken
        if self.chain_mode:
            return self.dropped | broken, torch.zeros_like(time_out)
        term = self.dropped | broken | self.violation
        if self.cfg.chain_train:  # the next skill starts on success (_chain_advance), a stuck skill fails
            # the warm-up hold after a restart only lets the freshly written grip settle (its first steps can kick the
            # cube past the violation bounds): only a drop ends it, the first skill starts with fresh peaks
            term = torch.where(self.chain_warm, self.dropped | broken, term)
            self.success_t = torch.where(self.success, self.success_t + self.step_dt, torch.zeros_like(self.success_t))
            stuck = (self.t_skill > self.cfg.chain_skill_timeout) & ~self.success
            if self.cfg.chain_timeout_advance_p > 0:  # some time-outs go on from wherever the skill left the cube
                self.timeout_adv = stuck & (torch.rand_like(self.t_skill) < self.cfg.chain_timeout_advance_p)
                stuck &= ~self.timeout_adv
            if self.cfg.chain_early_advance_p > 0:  # a premature detector: the next skill starts while the window runs
                early = self.early_adv & (self.stable_steps > 0) & ~self.success & (self.t_skill >= self.k_duration[self.skill])
                self.timeout_adv = self.timeout_adv | early
            return term | stuck, time_out
        if self.cfg.terminate_on_success:
            term |= self.success
        elif self.cfg.success_linger_s > 0:  # stay in the completed state a little (detector labels), then end
            self.success_t = torch.where(self.success, self.success_t + self.step_dt, torch.zeros_like(self.success_t))
            term |= self.success_t >= self.cfg.success_linger_s
        return term, time_out

    def _get_rewards(self):
        cfg = self.cfg
        progress = (self.prev_err - self.err).clamp(-0.2, 0.2)
        self.prev_err = self.err.clone()
        d = self.cube.data
        pos = d.root_pos_w - self.scene.env_origins
        rew = (
            cfg.rew_progress * progress
            + cfg.rew_success * self.success.float()
            + cfg.rew_hold_ready * (self.stable_steps > 0).float()
            + cfg.rew_cube_pos * (pos - self.c0).norm(dim=-1)
            + cfg.rew_cube_rot * self.rot_dev
            + cfg.rew_finger_return * self.finger_err * (self.layer_err < 0.2).float()
            + cfg.rew_action_rate * (self.actions - self.prev_actions).square().sum(-1)
            + cfg.rew_time
            + cfg.rew_trick_contact * (self.trick_on_layer & (self.layer_err > 0.35)).float()
            + self._handoff_penalty()
            + cfg.rew_drop * self.dropped.float()
            + cfg.rew_violation * self.violation.float()
        )
        self.extras["log"] = {
            "success": self.success.float().mean(),
            "layer_err_deg": torch.rad2deg(self.layer_err).mean(),
            "err_deg": torch.rad2deg(self.err).mean(),
            "violation": self.violation.float().mean(),
            "dropped": self.dropped.float().mean(),
            "peak_pos_mm": self.peak_pos.mean() * 1000,
            "peak_rot_deg": torch.rad2deg(self.peak_rot).mean(),
            "finger_err": self.finger_err.mean(),
            "holding": self.holding.float().mean(),
            "handoff_pos_mm": self.handoff_pos.mean() * 1000,
            "handoff_rot_deg": torch.rad2deg(self.handoff_rot).mean(),
        }
        late = self.t_skill > self.k_duration[self.skill]  # criteria pass rates after the nominal motion ended
        fam = {"flick": self.k_is_flick, "roll": self.k_is_roll, "regrip": self.k_is_regrip}
        for fn, fm in fam.items():
            sel = late & fm[self.skill]
            if sel.any():
                for c, v in self.crit.items():
                    self.extras["log"][f"crit_{fn}_{c}"] = v[sel].float().mean()
        # per-skill success rate and mean duration (chains: skills done before a failure) over the last completed
        # window of 24 steps (rsl_rl averages only the keys present at an iteration's first step)
        if self.common_step_counter % 24 == 0:
            ks = range(self._n_skill) if self.cfg.chain_train else self.skill_pool.tolist()
            self._win_log = {}
            for k in ks:
                if self.stat_n[k] > 0:
                    name = S.SKILLS[k].replace("'", "p")
                    self._win_log[f"succ_{name}"] = float(self.stat_ok[k] / self.stat_n[k])
                    self._win_log[f"len_{name}"] = float(self.stat_len[k] / self.stat_n[k])
            if self.cfg.chain_train and self.chain_len_n > 0:
                self._win_log["chain_len"] = self.chain_len_sum / self.chain_len_n
                self.chain_len_sum, self.chain_len_n = 0.0, 0
            self.stat_n.zero_(); self.stat_ok.zero_(); self.stat_len.zero_()
        self.extras["log"].update(getattr(self, "_win_log", {}))
        if self.cfg.chain_train:
            self._chain_advance()
        return torch.nan_to_num(rew)

    # ------------------------------------------------------------ chain training
    def _build_chain_bank(self):
        from .cube_logic import SkillCompiler

        rng = np.random.default_rng(self.cfg.seed)
        seqs = []
        if self.cfg.chain_bank_file:
            import json

            seqs = [[S.SKILL_INDEX[s] for s in q] for q in json.load(open(os.path.join(PROJECT_DIR, self.cfg.chain_bank_file)))["seqs"]]
        for _ in range(0 if seqs else self.cfg.chain_bank):
            comp = SkillCompiler()
            for m in rng.choice(list(self.cfg.chain_moves), self.cfg.chain_seq_moves):
                comp._emit(str(m))
            seqs.append([S.SKILL_INDEX[s] for s in comp.finish()])
        L = max(len(s) for s in seqs)
        self.chain_bank_t = torch.tensor([s + [0] * (L - len(s)) for s in seqs], device=self.device)
        self.chain_lens = torch.tensor([len(s) for s in seqs], device=self.device)
        z = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.chain_seq, self.chain_ptr, self.chain_done = z.clone(), z.clone(), z.clone()
        self.chain_warm = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.chain_len_sum, self.chain_len_n = 0.0, 0
        print(f"[solve] chain bank: {len(seqs)} sequences " + (f"from {self.cfg.chain_bank_file}" if self.cfg.chain_bank_file else
              f"of {self.cfg.chain_seq_moves} moves {self.cfg.chain_moves}") + ", "
              f"{self.chain_lens.float().mean():.0f} skills each", flush=True)

    def _chain_start(self, env_ids):
        """a new chain from the ready grip: random sequence, first skill."""
        self.chain_seq[env_ids] = torch.randint(len(self.chain_lens), (len(env_ids),), device=self.device)
        self.chain_ptr[env_ids] = 0
        self.chain_done[env_ids] = 0
        first = self.chain_bank_t[self.chain_seq[env_ids], 0]
        if self.cfg.chain_warmup_s > 0:  # hold first (zero action), _chain_advance starts the first skill
            self.chain_warm[env_ids] = True
            self.start_skill(env_ids, torch.full_like(first, S.SKILL_INDEX["hold"]))
            self.next_skill[env_ids] = first
            return
        self.start_skill(env_ids, first)
        self._chain_set_next(env_ids)

    def _chain_set_next(self, env_ids):
        seq = self.chain_seq[env_ids]
        self.next_skill[env_ids] = self.chain_bank_t[seq, (self.chain_ptr[env_ids] + 1) % self.chain_lens[seq]]

    def _chain_advance(self):
        """envs whose skill succeeded this step (and did not fail) start the next skill of their sequence."""
        warm = self.chain_warm & (self.t_skill >= self.cfg.chain_warmup_s) & ~self.reset_terminated
        if warm.any():
            w = warm.nonzero().squeeze(-1)
            self.chain_warm[w] = False
            self.start_skill(w, self.chain_bank_t[self.chain_seq[w], 0])
            self._chain_set_next(w)
        # the completed skill is held cfg.chain_linger_s before the next one starts (labels for a completion detector
        # that, as in execution, may fire a little after the success)
        linger = self.success_t >= self.cfg.chain_linger_s - 1e-6
        free = ~self.reset_terminated & ~self.chain_warm & ~warm
        ids = (self.success & linger & free).nonzero().squeeze(-1)
        tids = (self.timeout_adv & ~self.success & free).nonzero().squeeze(-1)
        if len(tids):  # a failure in the statistics, but the chain goes on
            kt = self.skill[tids]
            self.stat_n.index_add_(0, kt, torch.ones_like(kt, dtype=torch.float32))
            self.stat_len.index_add_(0, kt, self.t_skill[tids])
            self.timeout_adv[tids] = False
        if len(ids):
            k = self.skill[ids]
            one = torch.ones_like(k, dtype=torch.float32)
            self.stat_n.index_add_(0, k, one)
            self.stat_ok.index_add_(0, k, one)
            self.stat_len.index_add_(0, k, self.t_skill[ids])
            self.chain_done[ids] += 1
        ids = torch.cat([ids, tids])
        if len(ids) == 0:
            return
        seq = self.chain_seq[ids]
        self.chain_ptr[ids] = (self.chain_ptr[ids] + 1) % self.chain_lens[seq]
        self.start_skill(ids, self.chain_bank_t[seq, self.chain_ptr[ids]])
        self._chain_set_next(ids)

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.cube._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device)
        if self.cfg.chain_train and self._started:
            fail = self.reset_terminated[env_ids]
            cont = ~fail & (torch.rand(len(env_ids), device=self.device) >= self.cfg.chain_restart_prob)
            if fail.any():  # per-skill failure statistics, chain length
                k = self.skill[env_ids[fail]]
                self.stat_n.index_add_(0, k, torch.ones_like(k, dtype=torch.float32))
                self.stat_len.index_add_(0, k, self.t_skill[env_ids[fail]])
                self.chain_len_sum += float(self.chain_done[env_ids[fail]].sum())
                self.chain_len_n += int(fail.sum())
            # a truncated episode goes on from the current physical state, in the middle of its skill
            self.episode_length_buf[env_ids[cont]] = 0
            env_ids = env_ids[~cont]
            if len(env_ids) == 0:
                return
        # successful terminal states become start states of later episodes (skill chaining without teleports)
        if self.cfg.buffer_size > 0 and self._started and not self.cfg.chain_train:
            ok = self.success[env_ids] & ~self.violation[env_ids] & (self.tilt[env_ids] < self.cfg.tilt_tol)
            if ok.any():
                self._buffer_add(self.capture_full(env_ids[ok]))
        self._last_success[env_ids] = self.success[env_ids]
        self._last_violation[env_ids] = self.violation[env_ids]
        self._last_dropped[env_ids] = self.dropped[env_ids] if hasattr(self, "dropped") else False
        self._last_t[env_ids] = self.t_skill[env_ids]
        self._last_twist[env_ids] = self.cube.data.joint_pos[env_ids, self.twist_id]
        if self._started and not self.cfg.chain_train:  # per-skill outcome statistics of the finished episodes
            k = self.skill[env_ids]
            self.stat_n.index_add_(0, k, torch.ones_like(k, dtype=torch.float32))
            self.stat_ok.index_add_(0, k, self.success[env_ids].float())
            self.stat_len.index_add_(0, k, self.t_skill[env_ids])
        super()._reset_idx(env_ids)
        self._goal_fresh[env_ids] = True
        self.tactile.reset(env_ids)
        self._randomize(env_ids)
        if self._vis is not None:
            self._vis_reset(env_ids)
        n, dev = len(env_ids), self.device
        use_buf = torch.zeros(n, dtype=torch.bool, device=dev)
        if self.buf_count > 0:
            use_buf = torch.rand(n, device=dev) < self.cfg.buffer_prob
        use_pool = torch.zeros(n, dtype=torch.bool, device=dev)
        if self.cfg.start_pool:
            if not hasattr(self, "pool"):
                self.pool = {k: v.to(dev) for k, v in torch.load(self.cfg.start_pool).items()}
            use_pool = torch.rand(n, device=dev) < self.cfg.start_pool_prob
            use_buf &= ~use_pool
            if use_pool.any():
                ids = env_ids[use_pool]
                pick = torch.randint(len(self.pool["roll"]), (len(ids),), device=dev)
                self._write_full(ids, {k: v[pick] for k, v in self.pool.items()})
        fresh = env_ids[~use_buf & ~use_pool]
        if len(fresh):
            self._write_ready(fresh)
        if use_buf.any():
            ids = env_ids[use_buf]
            pick = torch.randint(self.buf_count, (len(ids),), device=dev)
            self._write_full(ids, {k: v[pick] for k, v in self.buf.items()})
        self._started = True
        if self.cfg.chain_train:  # (chains restart from the ready grip: no start pool / buffer)
            self._chain_start(env_ids)
            return
        self.start_skill(env_ids, self._sample_skills(env_ids))

    def _sample_skills(self, env_ids):
        """skills valid from each env's hand roll state (flick hands unrolled, rolls stay within +-1 quarter)."""
        roll = self.roll[env_ids].round()  # (n, 2)
        w = torch.zeros(len(env_ids), self._n_skill, device=self.device)
        w[:, self.skill_pool] = self.skill_w
        after = roll.unsqueeze(1) + self.k_roll.unsqueeze(0)  # (n, K, 2)
        ok = (after.abs() <= 1).all(-1)
        fh = self.k_flick_hand.clamp(min=0)
        if self.cfg.solo_roll_other_unrolled:  # as cube_logic.SkillCompiler.SOLO_ROLL_OTHER_UNROLLED
            solo = (self.k_roll != 0).sum(-1) == 1  # (K,)
            other_rolled = (roll.unsqueeze(1) != 0) & (self.k_roll.unsqueeze(0) == 0)  # (n, K, 2)
            ok &= ~solo.unsqueeze(0) | ~other_rolled.any(-1)
        if self.cfg.flick_both_unrolled and S.FLICK_PARTNER_LIFT <= 0:  # as cube_logic.SkillCompiler.FLICK_BOTH_UNROLLED
            ok &= ~self.k_is_flick.unsqueeze(0) | (roll == 0).all(-1, keepdim=True)
        else:  # the flicking hand unrolled, the other one unrolled or in a FLICK_PARTNER_OK state
            partner_ok = torch.zeros(2, 3, dtype=torch.bool, device=self.device)  # (hand, roll + 1)
            partner_ok[:, 1] = True
            for side, r in S.FLICK_PARTNER_OK:
                partner_ok[SIDES.index(side), r + 1] = True
            other = (1 - fh).unsqueeze(0)  # (1, K)
            r_other = roll[:, 1 - fh].long().clamp(-1, 1) + 1  # (n, K)
            ok_other = partner_ok[other.expand_as(r_other), r_other]
            ok &= ~self.k_is_flick.unsqueeze(0) | ((roll[:, fh] == 0) & ok_other)
        # a rolled hand should mostly be regripped
        rolled = roll != 0  # (n, 2)
        boost = torch.ones_like(w)
        for h in range(2):
            boost[:, self.k_regrip == h] = torch.where(rolled[:, h : h + 1], self.cfg.regrip_boost, 1.0)
        w = w * ok.float() * boost
        w[w.sum(-1) == 0, S.SKILL_INDEX["hold"]] = 1.0
        return torch.multinomial(w, 1).squeeze(-1)

    def _write_ready(self, env_ids):
        n, dev = len(env_ids), self.device
        origins = self.scene.env_origins[env_ids]
        idx = torch.randint(self.n_ready, (n,), device=dev)
        if self.cfg.ready_sequential:  # screening: env i starts from ready state i (mod the cache size)
            idx = env_ids % self.n_ready
        self.ready_idx[env_ids] = idx
        self.grip_mode[env_ids] = 0
        self.grip_mode_next[env_ids] = 0
        for h, side in enumerate(SIDES):
            if self.alt[side] is not None:
                self.alt_idx[env_ids, h] = torch.randint(self.alt[side]["n"], (n,), device=dev)
        g = {k: v[idx] for k, v in self.ready.items()}
        self.c0[env_ids] = g["cube_pos"]
        for h, side in enumerate(SIDES):
            hand = self.hands[side]
            self._set_base(env_ids, h, idx)
            if self._fr3 is not None:  # the cache holds float-base joint vectors: arm IK + fingers by name
                base_act = self.ctrl[side].reset_targets(g[f"{side}_ee_pos_act"] + origins, g[f"{side}_ee_quat_act"], env_ids)
                jp = hand.data.default_joint_pos[env_ids].clone()
                jp[:, self.finger_ids[side]] = g[f"{side}_fingers_act"]
            else:
                zeros6 = torch.zeros(n, 6, device=dev)
                base_act = self.ctrl[side].joint_targets(g[f"{side}_ee_pos_act"] + origins, g[f"{side}_ee_quat_act"], joint_pos=zeros6, env_ids=env_ids)
                jp = g[f"{side}_joint_pos_act"].clone()
            jp[:, self.ctrl[side].joint_ids] = base_act
            self._write_hand(env_ids, h, jp)
        root = self.cube.data.default_root_state[env_ids].clone()
        root[:, :3] = g["cube_pos"] + origins
        root[:, 3:7] = g["cube_quat"]  # cached states have the free layer on U: the cached quat is the core's
        root[:, 7:] = 0.0
        self.cube.write_root_state_to_sim(root, env_ids=env_ids)
        self.bind_S[env_ids] = torch.eye(3, device=dev)
        if self.cfg.cube_jam_deg is not None:
            self.face_res[env_ids] = 0.0
            self._set_jam(env_ids, torch.zeros(len(env_ids), dtype=torch.bool, device=dev))
        tw = g["cube_twist"].unsqueeze(-1)
        if self.mid_id is not None:
            self.cube.write_joint_state_to_sim(torch.zeros_like(tw), torch.zeros_like(tw), joint_ids=[self.mid_id], env_ids=env_ids)
        self.cube.write_joint_state_to_sim(tw, torch.zeros_like(tw), joint_ids=[self.twist_id], env_ids=env_ids)
        self.roll[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.act_smooth[env_ids] = 0.0
        self._act_hist[env_ids] = 0.0
        for buf in (self.teacher_smooth, self.teacher_prev, self.student_prev, self.teacher_raw):
            buf[env_ids] = 0.0

    def _set_base(self, env_ids, h, idx):
        side = SIDES[h]
        bp, bq = self.ready[f"{side}_base_pos"][idx], self.ready[f"{side}_base_quat"][idx]
        # base x axis on the hand's face normal: a roll is then a pure first virtual joint (no Euler singularity)
        bq = _aligned_base_quat(self.hand_normal[h].expand(len(env_ids), 3), bq)
        self.ctrl[side].set_base_pose(bp + self.scene.env_origins[env_ids], bq, env_ids)

    def _write_hand(self, env_ids, h, jp):
        side = SIDES[h]
        hand = self.hands[side]
        hand.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=env_ids)
        hand.set_joint_position_target(jp, env_ids=env_ids)
        self._base_tgt[side][env_ids] = jp[:, self.ctrl[side].joint_ids]
        self._base_vel[side][env_ids] = 0.0
        self.finger_tgt[env_ids, h] = jp[:, self.finger_ids[side]]
        p, q = self.ctrl[side].ee_pose_w()
        self.wrist_tgt_pos[env_ids, h] = p[env_ids] - self.scene.env_origins[env_ids]
        self.wrist_tgt_quat[env_ids, h] = q[env_ids]

    def capture_full(self, env_ids) -> dict:
        """complete restorable state (velocities excluded) of these envs at the end of their skill."""
        d = self.cube.data
        return {
            "ready_idx": self.ready_idx[env_ids].float(),
            "roll": self.roll[env_ids].round(),
            "hand_q": torch.stack([self.hands[s].data.joint_pos[env_ids] for s in SIDES], 1),
            "cube_pose": torch.cat([d.root_pos_w[env_ids] - self.scene.env_origins[env_ids], d.root_quat_w[env_ids]], -1),
            "twist": d.joint_pos[env_ids, self.twist_id],
            "mid": d.joint_pos[env_ids, self.mid_id] if self.mid_id is not None else torch.zeros_like(self.roll[env_ids, 0]),
            "face_res": self.face_res[env_ids],
            "bind_S": self.bind_S[env_ids],
            "act_smooth": self.act_smooth[env_ids],
            "actions": self.actions[env_ids],
        }

    def _write_full(self, env_ids, st):
        idx = st["ready_idx"].long()
        self.ready_idx[env_ids] = idx
        self.grip_mode[env_ids] = 0  # chained training starts are captured in the ready grip
        self.grip_mode_next[env_ids] = 0
        self.c0[env_ids] = self.ready["cube_pos"][idx]
        for h in range(2):
            self._set_base(env_ids, h, idx)
            self._write_hand(env_ids, h, st["hand_q"][:, h])
        root = self.cube.data.default_root_state[env_ids].clone()
        root[:, :7] = st["cube_pose"]
        root[:, :3] += self.scene.env_origins[env_ids]
        root[:, 7:] = 0.0
        self.cube.write_root_state_to_sim(root, env_ids=env_ids)
        tw = st["twist"].unsqueeze(-1)
        if self.mid_id is not None:
            mid = st["mid"].unsqueeze(-1) if "mid" in st else torch.zeros_like(tw)
            self.cube.write_joint_state_to_sim(mid, torch.zeros_like(tw), joint_ids=[self.mid_id], env_ids=env_ids)
        self.cube.write_joint_state_to_sim(tw, torch.zeros_like(tw), joint_ids=[self.twist_id], env_ids=env_ids)
        self.bind_S[env_ids] = st["bind_S"]
        if self.cfg.cube_jam_deg is not None:
            self.face_res[env_ids] = st["face_res"] if "face_res" in st else 0.0
            self._set_jam(env_ids, self._jam_of(env_ids))
        self.roll[env_ids] = st["roll"]
        self.act_smooth[env_ids] = st["act_smooth"]
        self.actions[env_ids] = st["actions"]
        self.prev_actions[env_ids] = st["actions"]

    def _buffer_add(self, st):
        if self.buf is None:
            self.buf = {k: torch.zeros(self.cfg.buffer_size, *v.shape[1:], device=self.device) for k, v in st.items()}
        m = len(st["roll"])
        pos = (self.buf_head + torch.arange(m, device=self.device)) % self.cfg.buffer_size
        for k, v in st.items():
            self.buf[k][pos] = v
        self.buf_head = int((self.buf_head + m) % self.cfg.buffer_size)
        self.buf_count = min(self.cfg.buffer_size, self.buf_count + m)

    def capture_state(self, env_ids=None) -> dict:
        """current physical state in the grasp-cache format (ready-grip targets kept), for new ready caches/buffers.

        Only valid for hands at roll 0 (the ready layout): wrist targets are the ready targets of each env.
        """
        ids = slice(None) if env_ids is None else env_ids
        d = self.cube.data
        out = {
            "cube_pos": (d.root_pos_w - self.scene.env_origins)[ids],
            "cube_quat": self._logical_quat()[ids],
            "cube_twist": torch.zeros_like(d.joint_pos[ids, self.twist_id]),
            "move_id": torch.full_like(d.joint_pos[ids, self.twist_id], 6.0),
        }
        out["cube_twist"] += quarter_residual(d.joint_pos[ids, self.twist_id]) * (self.bind_S[ids, 2, 2] > 0.5).float()
        ri = self.ready_idx[ids]
        for h, side in enumerate(SIDES):
            hand = self.hands[side]
            p, q = self.ctrl[side].ee_pose_w()
            out[f"{side}_ee_pos"] = self.ready[f"{side}_ee_pos"][ri]
            out[f"{side}_ee_quat"] = self.ready[f"{side}_ee_quat"][ri]
            out[f"{side}_ee_pos_act"] = (p - self.scene.env_origins)[ids]
            out[f"{side}_ee_quat_act"] = q[ids]
            out[f"{side}_fingers"] = self.ready[f"{side}_fingers"][ri]
            out[f"{side}_fingers_act"] = hand.data.joint_pos[ids][:, self.finger_ids[side]]
            out[f"{side}_joint_pos_act"] = hand.data.joint_pos[ids]
            out[f"{side}_base_pos"] = self.ready[f"{side}_base_pos"][ri]
            out[f"{side}_base_quat"] = self.ready[f"{side}_base_quat"][ri]
        return out

    @property
    def _base_tgt(self):
        if not hasattr(self, "_base_tgt_buf"):
            self._base_tgt_buf = {s: torch.zeros(self.num_envs, len(self.ctrl[s].joint_ids), device=self.device) for s in SIDES}
        return self._base_tgt_buf

    @property
    def _base_vel(self):
        if not hasattr(self, "_base_vel_buf"):
            self._base_vel_buf = {s: torch.zeros(self.num_envs, len(self.ctrl[s].joint_ids), device=self.device) for s in SIDES}
        return self._base_vel_buf


def _aligned_base_quat(axis, wrist_quat):
    """base orientation with x = axis and z = the wrist's finger direction projected orthogonal to it."""
    x = torch.nn.functional.normalize(axis, dim=-1)
    zw = mu.quat_apply(wrist_quat, torch.tensor([0.0, 0.0, 1.0], device=axis.device).expand_as(axis))
    z = zw - (zw * x).sum(-1, keepdim=True) * x
    z = torch.nn.functional.normalize(z, dim=-1)
    y = torch.cross(z, x, dim=-1)
    return mu.quat_from_matrix(torch.stack([x, y, z], -1))
