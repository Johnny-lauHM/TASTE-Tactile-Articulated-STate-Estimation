"""Load trained skill policies for chained execution (teacher MLPs from rsl_rl, student recurrent policies)."""

from __future__ import annotations

import math

import torch
from tensordict import TensorDict

import isaaclab.utils.math as mu

from rubik_lab.assets import FINGERTIPS


class TeacherPolicy:
    """rsl_rl ActorCritic checkpoint; called with the env's observation tensor of its group."""

    def __init__(self, path: str, obs_dim: int, device):
        from rsl_rl.modules import ActorCritic

        from .agent_cfg import SolvePPOCfg
        from .mix_policy import MixActorCritic

        pc = SolvePPOCfg().policy.to_dict()
        pc.pop("class_name", None)
        dummy = TensorDict({"policy": torch.zeros(1, obs_dim, device=device)}, batch_size=[1])
        sd = torch.load(path, map_location=device, weights_only=False)["model_state_dict"]
        cls = MixActorCritic if "fam.0.log_std" in sd else ActorCritic  # both families in one checkpoint (chain training)
        self.net = cls(dummy, {"policy": ["policy"], "critic": ["policy"]}, 56, **pc).to(device)
        self.net.load_state_dict(sd)
        self.net.eval()

    @torch.no_grad()
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net.act_inference(TensorDict({"policy": obs}, batch_size=[obs.shape[0]]))

    def reset(self, env_ids):
        pass


def load_policies(env, paths: dict[str, str], obs_key: str = "policy") -> dict:
    obs = env._get_observations()[obs_key]
    out = {}
    for fam, path in paths.items():
        blob = torch.load(path, map_location=env.device, weights_only=False)
        if "model_state_dict" in blob and "actor.0.weight" in blob["model_state_dict"] and blob["model_state_dict"]["actor.0.weight"].shape[1] == 250:
            out[fam] = LegacyFlickPolicy(env, path)  # old bimanual_twist flick checkpoint (4-dim trick-finger action)
        elif "student_state_dict" in blob:
            from .student import StudentPolicy

            out[fam] = StudentPolicy.from_checkpoint(blob, env.num_envs, env.device)
        else:
            out[fam] = TeacherPolicy(path, obs.shape[-1], env.device)
    return out


class LegacyFlickPolicy:
    """Single-finger flick policy of the old bimanual_twist task (Rubik-Flick-*-Sharpa-v0, 250-dim obs, 4-dim trick-finger
    action) driving the U / U' / D / D' skills of SolveEnv.

    Rebuilds the old observation from SolveEnv state and writes the old finger targets into env.finger_override: the
    trick finger at grasp target + EMA-smoothed action, the other fingers as the old env held them (grasp targets,
    thumbs / middles pressed by `hold_press`, fingers on the moving layer and the trick hand's pinky lifted by
    `lift`; SolveEnv's ready grip instead lifts every index / pinky, which tilts the cube out of the old policy's
    training distribution). Wrists keep SolveEnv's nominal. Once the layer is done by the old criterion (within
    8 deg, < 0.5 rad/s) the policy is no longer queried (its episodes ended there; it never learned a release) and
    the fingers return to SolveEnv's ready targets: the trick finger's MCP flexion first extends `first_extra` past
    its ready target over `first_s` (lifts the tip off the turned layer; uncurling in place drags the layer back
    30-50 deg), then everything moves to the ready targets over `return_s`.
    """

    MOVES = {"U": 6, "U'": 7, "D": 9, "D'": 10}  # rubik_lab.moves.MOVE_INDEX
    N_MOVES = 21
    EMA, LIMIT_SCALE, CLIP_OBS = 0.5, 0.95, 5.0
    DONE_VEL = 0.5

    def __init__(self, env, path: str, return_s: float = 0.3, hold_press: float = 0.15, lift: float = 0.45, done_tol_deg: float = 8.0,
                 retreat_first: int = 0, first_s: float = 0.25, first_extra: float = 0.4, ideal_cube: bool = False,
                 partner_unroll: bool = True, latch_steps: int = 1, latch_min_s: float = 0.0):
        from . import skills as S

        self.env, self.S, dev, n = env, S, env.device, env.num_envs
        self.ideal_cube = ideal_cube  # diagnostic: the old policy sees the cube at its ready-grip pose
        # the old policy only saw the other hand in the ready grip: a rolled partner hand is shown rotated back about its
        # face normal through the nominal cube center (wrist pose and fingertips; joints, forces are roll invariant)
        self.partner_unroll = partner_unroll
        # completion latch: the criterion must hold this many consecutive steps, not before latch_min_s into the skill
        # (blind: one wrong layer estimate would otherwise end the flick before the layer moved)
        self.latch_steps, self.latch_min_s = latch_steps, latch_min_s
        sd = torch.load(path, map_location=dev, weights_only=False)["model_state_dict"]
        layers = sorted({int(k.split(".")[1]) for k in sd if k.startswith("actor.")})
        mods = []
        for i, li in enumerate(layers):
            w = sd[f"actor.{li}.weight"]
            lin = torch.nn.Linear(w.shape[1], w.shape[0]).to(dev)
            lin.weight.data.copy_(w)
            lin.bias.data.copy_(sd[f"actor.{li}.bias"])
            mods += [lin] + ([torch.nn.ELU()] if i < len(layers) - 1 else [])
        self.net = torch.nn.Sequential(*mods).eval()
        self.return_steps = max(1, round(return_s / env.step_dt))
        self.done_tol = math.radians(done_tol_deg)
        # retreat: this trick-finger joint (column 0..3) returns first, alone, over first_s; then the rest (-1: all at once)
        # and first_extra [rad] beyond its ready target (lifts the tip off the turned layer before the finger uncurls)
        self.retreat_first, self.first_steps, self.first_extra = retreat_first, max(1, round(first_s / env.step_dt)), first_extra
        self.k_move = torch.full((len(S.SKILLS),), -1, dtype=torch.long, device=dev)
        for s, mi in self.MOVES.items():
            self.k_move[S.SKILL_INDEX[s]] = mi
        self.tip_ids, self.lo, self.hi = {}, {}, {}
        for side in S.HANDS:
            hand = env.hands[side]
            self.tip_ids[side] = hand.find_bodies([f"{side}_{f}_fingertip" for f in FINGERTIPS], preserve_order=True)[0]
            mid, half = 0.5 * (env.finger_lo[side] + env.finger_hi[side]), 0.5 * (env.finger_hi[side] - env.finger_lo[side]) * self.LIMIT_SCALE
            self.lo[side], self.hi[side] = mid - half, mid + half
        # old holding targets per ready grip (n_ready, 2, 22) and the lifted fingers per skill (K, 2, 22)
        press = hold_press * (env.flex["thumb"] + env.flex["middle"]).clamp(max=1.0)
        self.hold = torch.stack([torch.max(torch.min(env.ready[f"{s}_fingers"] + press, self.hi[s]), self.lo[s]) for s in S.HANDS], 1)
        self.k_lift = torch.zeros(len(S.SKILLS), 2, 22, device=dev)
        for s_, (hand, finger) in S.FLICK.items():
            k = S.SKILL_INDEX[s_]
            self.k_lift[k] = lift * env.flex[finger]  # UDR grip: index on the U row, ring on the D row (the moving layer)
            self.k_lift[k, S.HANDS.index(hand)] = lift * (env.flex[finger] + env.flex["pinky"]).clamp(max=1.0)
        self.full_latch = torch.zeros(n, 2, 22, device=dev)
        self.actions = torch.zeros(n, 4, device=dev)
        self.smooth = torch.zeros(n, 4, device=dev)
        self.latch = torch.zeros(n, dtype=torch.bool, device=dev)
        self.done_count = torch.zeros(n, dtype=torch.long, device=dev)
        self.back = torch.zeros(n, device=dev)  # return-phase progress [steps]
        self.last_t = torch.full((n,), -1.0, device=dev)
        self.last_k = torch.full((n,), -1, dtype=torch.long, device=dev)

    def reset(self, env_ids):
        self.last_t[env_ids] = -1.0

    def _obs(self, mine):
        e, S = self.env, self.S
        o = e.scene.env_origins
        per = e.perceived()  # the simulator's cube state, or the blind estimate (cfg.cube_est)
        core = per["pos"] + e.c0
        obs = []
        for h, side in enumerate(S.HANDS):
            hand = e.hands[side]
            q = e.meas_q(h)
            p, quat = e.ctrl[side].ee_pose_w()
            hp, hq = e.ready[f"{side}_ee_pos_act"][e.ready_idx], e.ready[f"{side}_ee_quat_act"][e.ready_idx]
            ft = e.meas_ft(h)
            tips = hand.data.body_pos_w[:, self.tip_ids[side]] - o.unsqueeze(1)
            if self.partner_unroll:
                partner = mine & (e.k_flick_hand[e.skill] != h)
                ang = -e.roll[:, h] * (0.5 * math.pi) * partner.float()
                dq = mu.quat_from_angle_axis(ang, e.hand_normal[h].expand(e.num_envs, 3))
                c = e.c0 + o
                p = c + mu.quat_apply(dq, p - c)
                quat = mu.quat_mul(dq, quat)
                tips = e.c0.unsqueeze(1) + mu.quat_apply(dq.unsqueeze(1).expand(-1, tips.shape[1], -1), tips - e.c0.unsqueeze(1))
            obs += [
                2.0 * (q - self.lo[side]) / (self.hi[side] - self.lo[side]) - 1.0,
                0.1 * e.meas_qd(h),
                p - o - e.c0,
                mu.quat_unique(quat),
                10.0 * (p - o - hp),
                mu.axis_angle_from_quat(mu.quat_mul(quat, mu.quat_inv(hq))),
                (tips - core.unsqueeze(1)).flatten(1),
                torch.cat([ft[..., :3] * 0.2, ft[..., 3:] * 20.0], -1).flatten(1),
            ]
        twist, tv = per["twist"], per["twist_vel"]
        move = self.k_move[e.skill].clamp(min=0)
        hand_1h = torch.nn.functional.one_hot(e.k_flick_hand[e.skill].clamp(min=0), 2).float() * mine.float().unsqueeze(-1)
        zero1 = torch.zeros(e.num_envs, 1, device=e.device)
        obs += [
            torch.zeros_like(core) if self.ideal_cube else core - e.c0,
            mu.quat_unique(e.ready["cube_quat"][e.ready_idx] if self.ideal_cube else per["lq"]),
            torch.stack([twist.sin(), twist.cos(), 0.1 * tv], -1),
            per["lin_vel"],
            0.1 * per["ang_vel"],
            torch.nn.functional.one_hot(move, self.N_MOVES).float(),
            zero1,  # move_roll (flicks: no wrist roll)
            self.latch.float().unsqueeze(-1),
            hand_1h,  # wrist_mask: the trick hand
            zero1,  # roll nominal
            self.actions,
        ]
        return torch.nan_to_num(torch.cat(obs, -1)).clamp(-self.CLIP_OBS, self.CLIP_OBS)

    @torch.no_grad()
    def __call__(self, obs=None):
        """external controller: write the targets into env.finger_override, no residual action."""
        mine, full = self.targets()
        self.env.finger_override[mine] = full[mine]
        return torch.zeros(self.env.num_envs, 56, device=self.env.device)

    @torch.no_grad()
    def targets(self):
        """advance one control step: (N,) envs in a legacy flick skill, (N, 2, 22) finger targets."""
        e = self.env
        k = e.skill
        mine = self.k_move[k] >= 0
        new = (e.t_skill < self.last_t) | (self.last_t < 0) | (k != self.last_k)
        for buf in (self.actions, self.smooth, self.full_latch):
            buf[new] = 0.0
        self.latch[new] = False
        self.done_count[new] = 0
        self.back[new] = 0.0
        self.last_t, self.last_k = e.t_skill.clone(), k.clone()
        # old completion label (latched): from here on the old policy was never trained
        per = e.perceived()
        err = mu.wrap_to_pi(e.layer_target - per["twist"]).abs()
        done = (err < self.done_tol) & (per["twist_vel"].abs() < self.DONE_VEL) & (per["ang_vel"].norm(dim=-1) < 1.0)
        a = self.net(self._obs(mine)).clamp(-1.0, 1.0)
        live = mine & ~self.latch
        self.actions = torch.where(live.unsqueeze(-1), a, self.actions)
        self.smooth = torch.where(live.unsqueeze(-1), self.EMA * a + (1.0 - self.EMA) * self.smooth, self.smooth)
        rows = torch.arange(e.num_envs, device=e.device)
        h = e.k_flick_hand[k].clamp(min=0)
        cols = e.k_flick_cols[k]
        lo = torch.stack([self.lo[s] for s in self.S.HANDS])[h].gather(1, cols)
        hi = torch.stack([self.hi[s] for s in self.S.HANDS])[h].gather(1, cols)
        grasp = torch.stack([e.ready[f"{s}_fingers"][e.ready_idx] for s in self.S.HANDS], 1)[rows, h].gather(1, cols)
        tgt = torch.max(torch.min(grasp + self.smooth, hi), lo)
        full = self.hold[e.ready_idx] - self.k_lift[k]
        full[rows.unsqueeze(-1), h.unsqueeze(-1), cols] = tgt
        self.done_count = torch.where(done, self.done_count + 1, torch.zeros_like(self.done_count))
        just = mine & (self.done_count >= self.latch_steps) & (e.t_skill >= self.latch_min_s) & ~self.latch
        self.full_latch = torch.where(just.view(-1, 1, 1), full, self.full_latch)
        self.latch |= just
        # return phase: from the targets at completion to SolveEnv's ready targets
        ready = torch.stack([e.ready_fingers[s][e.ready_idx] for s in self.S.HANDS], 1)
        first = self.first_steps if self.retreat_first >= 0 else 0
        self.back = torch.where(self.latch, (self.back + 1.0).clamp(max=first + self.return_steps), self.back)
        w = self.S.smoothstep((self.back - first) / self.return_steps).view(-1, 1, 1)
        start = self.full_latch
        if first:
            c = cols[:, self.retreat_first]
            mid = self.full_latch.clone()
            mid[rows, h, c] = ready[rows, h, c] - self.first_extra  # extension
            w1 = self.S.smoothstep(self.back / first).view(-1, 1, 1)
            start = (1.0 - w1) * self.full_latch + w1 * mid
        full = torch.where(self.latch.view(-1, 1, 1), (1.0 - w) * start + w * ready, full)
        return mine, full
