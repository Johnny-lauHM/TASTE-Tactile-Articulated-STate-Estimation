"""Legacy scripted teacher with wrist-roll and finger-flick branches, using privileged sim state.

Implemented branches are not evidence of accepted full-cycle skills.

Driven by rubik_lab.moves.primitive_spec, per env, in the env's residual action space (env.cfg.wrist_prior = False):
    SETTLE
    [U only]  SH_RELEASE -> SH_MOVE -> SH_GRIP : the twisting hand lets go, moves up one row, pinches the top row
    TWIST     the rotating hands press and rotate together about the move axis (pivot = cube center), small overshoot;
              if the target isn't reached after the full roll: UNDO (roll back gripping) and retry
    per rotating hand, one after the other (x: right, then left):
              RELEASE (open + lift off) -> RETURN (rotate back, undo the shift) -> REGRIP (gentle close, light press)
    DONE      wait for the env's primitive_done
Transitions are closed-loop on the true layer angle / cube orientation (env.signed_err), pad contacts and the
commanded motion.
"""

from __future__ import annotations

import math

import torch

import isaaclab.utils.math as math_utils

from .constants import FINGER_JOINTS
from .cube_contacts import FINGERTIP_LINK

SETTLE, SH_RELEASE, SH_MOVE, SH_GRIP, TWIST, RELEASE, RETURN, REGRIP, DONE, UNDO = range(10)
F_SETTLE, F_PUSH, F_RELEASE, F_WAIT, F_DONE = range(10, 15)  # finger tricks (rubik_lab.flick)
PHASES = ("settle", "sh_release", "sh_move", "sh_grip", "twist", "release", "return", "regrip", "done", "undo",
          "f_settle", "f_push", "f_release", "f_wait", "f_done")  # fmt: skip


class ScriptedTeacher:
    def __init__(self, env, settle_s=0.3, roll_speed=math.radians(90), overshoot=1.10, press=0.15, hold_press=0.10,
                 open_amount=0.70, twist_tol=math.radians(5.0), release_s=0.5, retract=0.010, regrip_s=0.5,
                 regrip_press=0.05, release_tol=math.radians(15.0), max_tries=2, shift_time=0.6, max_overshoot=1.5, flick_lead=3.0, press_max=0.010, lead_max=9.0,
                 min_pads=2, regrip_max_s=1.5, close_rate=0.4, max_press=0.30, wrap_flex=0.6, wrap_palm=0.006):  # fmt: skip
        self.env = env
        n, dev = env.num_envs, env.device
        self.n, self.dev, self.dt = n, dev, env.step_dt
        self.c = dict(settle_s=settle_s, roll_speed=roll_speed, overshoot=overshoot, press=press, hold_press=hold_press,
                      open_amount=open_amount, twist_tol=twist_tol, release_s=release_s, retract=retract,
                      regrip_s=regrip_s, regrip_press=regrip_press, release_tol=release_tol, max_tries=max_tries, flick_lead=flick_lead, press_max=press_max, lead_max=lead_max,
                      shift_time=shift_time, max_overshoot=max_overshoot, min_pads=min_pads, regrip_max_s=regrip_max_s,
                      close_rate=close_rate, max_press=max_press, wrap_flex=wrap_flex, wrap_palm=wrap_palm)  # fmt: skip
        self.phase = torch.zeros(n, dtype=torch.long, device=dev)
        self.t_phase = torch.zeros(n, device=dev)
        self.theta = torch.zeros(n, 2, device=dev)  # commanded rotation per hand [rad] about the move axis
        self.shift_frac = torch.zeros(n, 2, device=dev)  # 0..1 of the move's grip shift per hand
        self.blend = torch.zeros(n, 2, device=dev)  # 0..1 from the start grasp to the regrip target (x: paired grasp)
        self.cur = torch.zeros(n, dtype=torch.long, device=dev)  # index into the regrip order
        self.tries = torch.zeros(n, dtype=torch.long, device=dev)
        self.best_err = torch.full((n,), 10.0, device=dev)  # smallest |err| during the current twist
        self.err_at = torch.full((n, len(PHASES)), float("nan"), device=dev)
        cols = [i for i, j in enumerate(FINGER_JOINTS) if j.split("_")[0] in ("thumb", "index", "middle")
                and ("FE" in j or j.endswith(("PIP", "DIP", "IP")))]  # fmt: skip
        self.flex = torch.zeros(len(FINGER_JOINTS), device=dev)
        self.flex[cols] = 1.0
        # ring / pinky flexion: used to wrap the cube when one hand holds it alone
        wrap_cols = [i for i, j in enumerate(FINGER_JOINTS) if j.split("_")[0] in ("ring", "pinky")
                     and ("FE" in j or j.endswith(("PIP", "DIP")))]  # fmt: skip
        self.wrap = torch.zeros(len(FINGER_JOINTS), device=dev)
        self.wrap[wrap_cols] = 1.0

    def reset(self, env_ids):
        flick = self.env.move_is_flick[self.env.move_id[env_ids]]
        self.phase[env_ids] = torch.where(flick, F_SETTLE, SETTLE)
        if hasattr(self, "f_tries"):
            self.f_tries[env_ids] = 0
            self.f_prog[env_ids] = 0.0
            self.f_kmax[env_ids] = 0
        self.t_phase[env_ids] = 0.0
        self.theta[env_ids] = 0.0
        self.shift_frac[env_ids] = 0.0
        self.blend[env_ids] = 0.0
        self.cur[env_ids] = 0
        self.tries[env_ids] = 0
        self.best_err[env_ids] = 10.0
        self.err_at[env_ids] = float("nan")

    def _goto(self, mask, phase):
        if hasattr(self.env, "signed_err"):
            self.err_at[:, phase] = torch.where(mask, self.env.signed_err, self.err_at[:, phase])
        self.phase = torch.where(mask, torch.full_like(self.phase, phase), self.phase)
        self.t_phase = torch.where(mask, torch.zeros_like(self.t_phase), self.t_phase)

    @torch.no_grad()
    def act(self) -> torch.Tensor:
        a = self._act_main()
        fl = self.env.move_is_flick[self.env.move_id]
        if fl.any():
            a = torch.where(fl.unsqueeze(-1), self._act_flick(fl), a)
        return a

    # ------------------------------------------------------------------ finger tricks
    def _flick_init(self):
        from .flick import FlickPlanner

        n, dev = self.n, self.dev
        offset = self.c.get("flick_contact_offset")
        self.planner = FlickPlanner(dev, lateral_range=None if offset is None else (offset, offset), contact_height=self.c.get("flick_contact_height"))
        self.closed_loop = True  # re-solve the trick finger IK every step from the actual wrist pose
        W = self.planner.W
        self.f_q = torch.zeros(n, W, 22, device=dev)
        self.f_dp = torch.zeros(n, W, 3, device=dev)
        self.f_dr = torch.zeros(n, W, 3, device=dev)
        self.f_kmax = torch.zeros(n, dtype=torch.long, device=dev)
        self.f_prog = torch.zeros(n, device=dev)
        self.f_Rc = torch.eye(3, device=dev).repeat(n, 1, 1)
        self.f_lift = torch.zeros(n, 2, 22, device=dev)
        self.f_tries = torch.zeros(n, dtype=torch.long, device=dev)
        self.f_rel = torch.zeros(n, 22, device=dev)  # extension of the flicking finger while releasing
        self.f_phi0 = torch.zeros(n, device=dev)  # layer angle (move direction) when the push was planned
        self.f_press = torch.zeros(n, device=dev)  # closed-loop press depth [m]; ramps up while the layer stalls
        self.f_lead = torch.zeros(n, device=dev)  # waypoints the commanded point runs ahead of the layer; ramps up on a stall
        self.f_err_buf = torch.zeros(n, 8, device=dev)  # recent layer errors (stall detection)
        finger_of = [j.split("_")[0] for j in FINGER_JOINTS]
        flexj = [("FE" in j or j.endswith(("PIP", "DIP", "IP"))) for j in FINGER_JOINTS]
        self.finger_cols = {f: torch.tensor([finger_of[i] == f and flexj[i] for i in range(22)], device=dev).float()
                            for f in ("thumb", "index", "middle", "ring", "pinky")}  # fmt: skip

    def _flick_progress(self):
        """Layer progress in the commanded direction, positive for both U and U-prime.

        signed_err is target minus measured angle. The previous error-target
        expression only worked for negative-angle (clockwise) commands.
        """
        target = self.env.move_target[self.env.move_id]
        error = getattr(self.env, "signed_err", target)
        direction = torch.where(target < 0, -torch.ones_like(target), torch.ones_like(target))
        return (target - error) * direction

    def _flick_plan(self, mask):
        from .cube_contacts import LINKS
        from .flick import FLICKS, to_cube_frame

        e = self.env
        touch_top = e.cube_contacts.forces()[:, 1] > 0.05  # (N, 2 hands, 27 links) touching the moving layer
        for m, (side, finger, *_) in FLICKS.items():
            from .moves import MOVE_INDEX

            ids = (mask & (e.move_id == MOVE_INDEX[m])).nonzero().flatten()
            if len(ids) == 0:
                continue
            pos_c, R_c, fing, Rc = to_cube_frame(e, side, mode=self.c.get("flick_plan_mode", "cache"))
            retry_mode=self.c.get("flick_retry_plan_mode")
            if retry_mode is not None:
                alternate=to_cube_frame(e,side,mode=retry_mode)
                retry=self.f_tries>0
                pos_c,R_c,fing,Rc=tuple(torch.where(retry.reshape(-1,*([1]*(original.ndim-1))),current,original) for original,current in zip((pos_c,R_c,fing,Rc),alternate))
            with torch.enable_grad():
                phi = self._flick_progress()[ids]  # rotation the layer already has, in the move's direction
                q, dp, dr, ok = self.planner.solve(m, pos_c[ids], R_c[ids], fing[ids], phi=phi)
            self.f_q[ids], self.f_dp[ids], self.f_dr[ids] = q, dp, dr
            if not hasattr(self, "f_pad"):
                self.f_pad = torch.zeros(self.n, self.planner.W, 3, device=self.dev)
            self.f_pad[ids] = self.planner.last_pads
            if not hasattr(self, "f_tn"):
                self.f_tn = torch.zeros(self.n, self.planner.W, 3, device=self.dev)
            self.f_tn[ids] = self.planner.last_tn
            if not hasattr(self, "f_qcmd"):
                self.f_qcmd = torch.zeros(self.n, 22, device=self.dev)
            self.f_qcmd[ids] = q[:, 0]
            self.f_kmax[ids] = (ok.sum(-1) - 1).clamp(min=1)
            self.f_Rc[ids] = Rc[ids]
            h = 0 if side == "left" else 1
            # lift the other fingers that touch the moving layer (else they turn with it / block it) -- except the thumb
            # and the middle finger: they are the grip itself (their pads straddle the layer boundary in the home
            # grip; lifting them loses the cube), the layer just rubs past them
            lift = torch.zeros(len(ids), 2, 22, device=self.dev)
            for hh in range(2):
                for li, ln in enumerate(LINKS):
                    f = ln.split("_")[0]
                    if f not in self.finger_cols or (hh == h and f == finger) or f in ("thumb", "middle"):
                        continue
                    lift[:, hh] += touch_top[ids, hh, li].float().unsqueeze(-1) * self.finger_cols[f]
            lift[:, h] += self.finger_cols["pinky"]  # the flicking hand's pinky gets out of the ring/index finger's way
            self.f_lift[ids] = lift.clamp(max=1.0)
            if getattr(e, "trace_debug", False) and 0 in ids.tolist():
                tt = touch_top[0]
                print("  [plan-dbg] touching layer: " + str({hh_: [LINKS[li_] for li_ in tt[hh_].nonzero().flatten().tolist()] for hh_ in range(2)})
                      + f" | lift L {lift[(ids == 0).nonzero().flatten()[0], 0].nonzero().flatten().tolist()} R {lift[(ids == 0).nonzero().flatten()[0], 1].nonzero().flatten().tolist()}", flush=True)
            self.f_prog[ids] = 0.0
            self.f_rel[ids] = 0.25 * self.finger_cols[finger]
            self.f_phi0[ids] = phi

    def _act_flick(self, fl):
        e, c, dev = self.env, self.c, self.dev
        if not hasattr(self, "planner"):
            self._flick_init()
        ph, t = self.phase, self.t_phase
        self.f_tick = getattr(self, "f_tick", 0) + 1
        # plans are batched every 6 steps (one IK solve for all envs that are ready; each solve costs ~0.5 s)
        start = fl & (ph == F_SETTLE) & (t >= c["settle_s"]) & (self.f_tick % 6 == 0)
        if start.any():
            self._flick_plan(start)
            self._goto(start, F_PUSH)
        # approach: one waypoint per step; push: the commanded waypoint follows the layer (a bit ahead of it), and the
        # release uses directed progress, an optional segment threshold, stall detection or the phase-time cap.
        # The cycle diagnostic separately enforces the unchanged total primitive budget; it has no detent.
        pushing = fl & (self.phase == F_PUSH)
        n_app = self.planner.n_app
        err = e.signed_err if hasattr(e, "signed_err") else e.move_target[e.move_id]  # not set before the first step
        phi_now = self._flick_progress()
        prog = (phi_now - self.f_phi0) / self.planner.thetas[1]  # layer rotation since the plan, in waypoints
        in_app = self.f_prog < n_app
        self.f_prog = torch.where(pushing & in_app, self.f_prog + 1.0, self.f_prog)
        self.f_lead = torch.where(pushing & in_app, torch.full_like(self.f_lead, c["flick_lead"]), self.f_lead)
        self.f_prog = torch.where(pushing & ~in_app, torch.maximum(n_app + prog + self.f_lead, torch.full_like(prog, n_app)), self.f_prog)
        self.f_err_buf = torch.cat([self.f_err_buf[:, 1:], err.unsqueeze(-1)], 1)
        # press harder while the layer does not move (force control by depth: the finger PD is a spring)
        from .flick import PRESS_CL

        moving = (self.f_err_buf[:, -1] - self.f_err_buf[:, -2]).abs() > math.radians(0.3)
        self.f_press = torch.where(pushing & in_app, torch.full_like(self.f_press, PRESS_CL), self.f_press)
        self.f_press = torch.where(pushing & ~in_app & ~moving, (self.f_press + 0.001).clamp(max=c["press_max"]), self.f_press)
        self.f_lead = torch.where(pushing & ~in_app & ~moving, (self.f_lead + 0.5).clamp(max=c["lead_max"]), self.f_lead)  # pull further along the arc
        stalled = (self.t_phase >= 0.5) & ((self.f_err_buf[:, -1] - self.f_err_buf[:, 0]).abs() < math.radians(0.5)) & (self.f_press >= c["press_max"] - 1e-6)
        past = phi_now >= math.radians(87.0)  # no detent: push all the way to 90 deg
        stroke_limit = c.get("flick_stroke_limit_deg")
        segment_done = torch.zeros_like(past) if stroke_limit is None else (phi_now - self.f_phi0 >= math.radians(stroke_limit))
        self._goto(pushing & ~in_app & (past | segment_done | stalled | (self.t_phase >= 6.0)), F_RELEASE)  # a slow ring push can take 4-5 s
        self._goto(fl & (self.phase == F_RELEASE) & (self.t_phase >= 0.25), F_WAIT)
        waited = fl & (self.phase == F_WAIT) & (e.move_done | (self.t_phase >= 0.4))
        retry_ready = getattr(e, "flick_retry_ready", torch.ones_like(waited))
        waited = waited & (e.move_done | retry_ready)
        retry = waited & ~e.move_done & (self.f_tries < 4)  # a move may take several pushes (fingers only)
        self.f_tries = torch.where(retry, self.f_tries + 1, self.f_tries)
        self._goto(retry, F_SETTLE)
        self._goto(waited & ~retry, F_DONE)

        ph = self.phase
        n, P = self.n, e.per_hand
        a = torch.zeros(n, e.cfg.action_space, device=dev)
        k0 = self.f_prog.floor().long().clamp(max=self.f_q.shape[1] - 1)
        k0 = torch.min(k0, self.f_kmax)
        k1 = torch.min(k0 + 1, self.f_kmax)
        w = (self.f_prog - self.f_prog.floor()).clamp(0, 1).unsqueeze(-1)
        gat = lambda X: (X.gather(1, k0.view(-1, 1, 1).expand(-1, 1, X.shape[-1])).squeeze(1) * (1 - w)
                         + X.gather(1, k1.view(-1, 1, 1).expand(-1, 1, X.shape[-1])).squeeze(1) * w)  # noqa: E731
        q_t, dp_c, dr_c = gat(self.f_q), gat(self.f_dp), gat(self.f_dr)
        pushing = ph == F_PUSH
        if self.closed_loop and pushing.any():
            from .flick import FLICKS, to_cube_frame
            from .moves import MOVE_INDEX

            pad_t, tn_t = gat(self.f_pad), gat(self.f_tn)
            for m, (side, *_) in FLICKS.items():
                ids = (pushing & fl & (e.move_id == MOVE_INDEX[m])).nonzero().flatten()
                if len(ids) == 0:
                    continue
                pos_c, R_c, fing, _ = to_cube_frame(e, side)
                with torch.enable_grad():
                    q_cl = self.planner.track(m, pos_c[ids], R_c[ids], fing[ids], self.f_qcmd[ids], pad_t[ids], tn_t[ids], press=self.f_press[ids], align_normal=c.get("flick_track_normal", False))
                q_t[ids] = q_cl
                self.f_qcmd[ids] = q_cl
        releasing = ph == F_RELEASE
        lifted = (ph == F_PUSH) | (ph == F_RELEASE)
        trick_hand = e.move_twist_hand[e.move_id]  # the flicking hand (one-hot hands of a finger trick)
        rot_range, pos_range, fr = e.cfg.residual_rot_range, e.cfg.residual_pos_range, e.cfg.residual_finger_range
        for h, side in enumerate(("left", "right")):
            sl = slice(h * P, (h + 1) * P)
            mine = trick_hand == h
            # fingers: the trick finger follows the IK path, then extends a bit to let go; lifted fingers open
            base = e.finger_grasp[side]
            tgt = torch.where((mine & pushing).unsqueeze(-1), q_t, base)
            tgt = tgt - (mine & releasing).float().unsqueeze(-1) * self.f_rel
            tgt = tgt - lifted.float().unsqueeze(-1) * 0.45 * self.f_lift[:, h]
            a[:, sl][:, 6:] = ((tgt - base) / fr).clamp(-1, 1)
            # wrist assist (cube frame -> env frame), as a residual about the cube center
            dp_e = torch.einsum("bij,bj->bi", self.f_Rc, dp_c) * (mine & pushing).float().unsqueeze(-1)
            dr_e = torch.einsum("bij,bj->bi", self.f_Rc, dr_c) * (mine & pushing).float().unsqueeze(-1)
            p_g = e.ee_grasp_tgt[side]
            r_q = math_utils.quat_from_angle_axis(torch.linalg.norm(dr_e, dim=-1), dr_e)
            d_res = p_g + dp_e - e.cube_center - math_utils.quat_apply(r_q, p_g - e.cube_center)
            a[:, sl][:, 0:3] = d_res / pos_range
            a[:, sl][:, 3:6] = dr_e / rot_range
        return a.clamp(-1, 1)

    def _act_main(self) -> torch.Tensor:
        e, c, dev = self.env, self.c, self.dev
        mv = e.move_id
        hands = e.ep_hands > 0  # (N, 2) rotating hands of this episode (U: the grip state's edge hand)
        active = hands.any(-1)
        angle = e.move_roll[mv]  # signed, about e.roll_axis
        shift = e.ep_shift  # (N, 2, 3) logical
        has_shift = shift.norm(dim=-1).sum(-1) > 0
        # regrip order: single hand -> that hand; both (x) -> the edge hand first (it regrips as the new center hand while
        # the center hand holds the cube), then the former center hand
        both = hands[:, 0] & hands[:, 1]
        single = torch.where(hands[:, 1], 1, 0)
        order = torch.stack([torch.where(both, e.ep_edge, single), torch.where(both, 1 - e.ep_edge, -1)], -1)
        cur_hand = order.gather(1, self.cur.unsqueeze(-1)).squeeze(-1)  # hand being released / regripped
        err = e.signed_err if hasattr(e, "signed_err") else angle.abs()
        touch = e.cube_contacts.forces().sum(1) > 0.05  # (N, 2, 27)
        pads_on = (touch & FINGERTIP_LINK.to(dev)).sum(-1)  # (N, 2)
        shift_hand = torch.where(shift[:, 0].norm(dim=-1) > 0, 0, 1)  # the hand that shifts (U)
        pads_shift = pads_on.gather(1, shift_hand.unsqueeze(-1)).squeeze(-1)
        pads_cur = pads_on.gather(1, cur_hand.clamp(min=0).unsqueeze(-1)).squeeze(-1)

        # ---------------- transitions
        self.t_phase += self.dt
        ph, t = self.phase, self.t_phase
        settle_done = (ph == SETTLE) & (t >= c["settle_s"]) & active
        self._goto(settle_done & has_shift, SH_RELEASE)
        self._goto(settle_done & ~has_shift, TWIST)
        self._goto((self.phase == SH_RELEASE) & ((pads_shift == 0) | (self.t_phase >= c["release_s"])), SH_MOVE)
        self._goto((self.phase == SH_MOVE) & (self.t_phase >= c["shift_time"]), SH_GRIP)
        sh_ok = (pads_shift >= c["min_pads"]) & (self.t_phase >= c["regrip_s"] + 0.2)
        self._goto((self.phase == SH_GRIP) & (sh_ok | (self.t_phase >= c["regrip_max_s"])), TWIST)
        # reached: within tolerance, or past the error minimum (the compliant grip lags ~5 deg, so the minimum can stay
        # just above the tolerance and the error then grows again while the hands keep turning)
        in_twist = self.phase == TWIST
        self.best_err = torch.where(in_twist, torch.min(self.best_err, err.abs()), torch.full_like(self.best_err, 10.0))
        past_min = in_twist & (self.best_err < c["release_tol"]) & (err.abs() > self.best_err + math.radians(2.0))
        reached = (err.abs() < c["twist_tol"]) | past_min
        th = self.theta.abs().max(-1).values
        rolled_out = th >= c["max_overshoot"] * angle.abs() - 1e-4
        self._goto((self.phase == TWIST) & reached, RELEASE)
        failed = (self.phase == TWIST) & rolled_out & ~reached
        near = failed & (err.abs() < c["release_tol"])
        self._goto(near, RELEASE)  # close enough: the detent snaps it
        self.tries = torch.where(failed & ~near, self.tries + 1, self.tries)
        self._goto(failed & ~near, UNDO)
        self._goto((self.phase == UNDO) & (th < 1e-3) & (self.tries <= c["max_tries"]), TWIST)
        self._goto((self.phase == RELEASE) & ((pads_cur == 0) | (self.t_phase >= c["release_s"])), RETURN)
        cur_theta = self.theta.gather(1, cur_hand.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        cur_shift = self.shift_frac.gather(1, cur_hand.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        cur_blend = self.blend.gather(1, cur_hand.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        self._goto((self.phase == RETURN) & (cur_theta.abs() < 1e-3) & (cur_shift < 1e-3) & (cur_blend > 0.999), REGRIP)
        regrip_done = (self.phase == REGRIP) & (
            ((self.t_phase >= c["regrip_s"] + 0.2) & (pads_cur >= c["min_pads"])) | (self.t_phase >= c["regrip_max_s"])
        )
        more = regrip_done & (self.cur == 0) & (order[:, 1] >= 0)
        self.cur = torch.where(more, torch.ones_like(self.cur), self.cur)
        self._goto(more, RELEASE)
        self._goto(regrip_done & ~more, DONE)

        # ---------------- motion commands
        ph = self.phase
        cur_hand = order.gather(1, self.cur.unsqueeze(-1)).squeeze(-1)
        step = c["roll_speed"] * self.dt
        for h in range(2):
            mine = hands[:, h]
            goal = torch.where(ph == TWIST, c["max_overshoot"] * angle, self.theta[:, h])  # stop on `reached`
            goal = torch.where((ph == UNDO) | ((ph == RETURN) & (cur_hand == h)), torch.zeros_like(goal), goal)
            moving = mine & ((ph == TWIST) | (ph == UNDO) | (ph == RETURN))
            self.theta[:, h] = torch.where(moving, self.theta[:, h] + (goal - self.theta[:, h]).clamp(-step, step), self.theta[:, h])
            # grip shift (U): up during SH_MOVE, back down during this hand's RETURN
            sstep = self.dt / c["shift_time"]
            up = (ph == SH_MOVE) & (shift_hand == h) & has_shift
            down = (ph == RETURN) & (cur_hand == h) & has_shift
            self.shift_frac[:, h] = (self.shift_frac[:, h] + sstep * up.float() - sstep * down.float()).clamp(0.0, 1.0)
            span = (c["max_overshoot"] * angle.abs()).clamp(min=1e-3)
            bstep = step / span
            self.blend[:, h] = (self.blend[:, h] + bstep * ((ph == RETURN) & (cur_hand == h)).float()).clamp(0.0, 1.0)

        a = torch.zeros(self.n, e.cfg.action_space, device=dev)
        P = e.per_hand
        cube_disp = e._cube_state()[0] - e.cube_center  # (N, 3) env frame
        rot_range, pos_range, fr = e.cfg.residual_rot_range, e.cfg.residual_pos_range, e.cfg.residual_finger_range
        for h, side in enumerate(("left", "right")):
            sl = slice(h * P, (h + 1) * P)
            # regrip target relative to the episode's grasp reference (zero unless x): rotation r about the cube center
            q0, p0 = e.ee_grasp_quat[side], e.ee_grasp_tgt[side]
            r_vec = math_utils.axis_angle_from_quat(math_utils.quat_mul(e.regrip_quat[side], math_utils.quat_inv(q0)))
            r_q = math_utils.quat_from_angle_axis(torch.linalg.norm(r_vec, dim=-1), r_vec)
            d_pos = e.regrip_pos[side] - e.cube_center - math_utils.quat_apply(r_q, p0 - e.cube_center)
            bl = self.blend[:, h : h + 1]
            a[:, sl][:, 3:6] = (e.roll_axis * self.theta[:, h : h + 1] + bl * r_vec) / rot_range
            # position residual: grip shift (logical -> env) + lift-off along the palm normal when released
            shift_env = math_utils.quat_apply(e.cube_quat0, shift[:, h]) * self.shift_frac[:, h : h + 1]
            # re-grip where the cube is now (it sags / shifts a few mm during a move), not where it was at reset
            gripping = ((ph == REGRIP) & (cur_hand == h)) | ((ph == SH_GRIP) & (shift_hand == h) & has_shift)
            follow = gripping | (((ph == RETURN) & (cur_hand == h)))
            shift_env = shift_env + follow.float().unsqueeze(-1) * cube_disp
            palm_n = math_utils.quat_apply(e.ee_quat_tgt[side], torch.tensor([1.0, 0.0, 0.0], device=dev).expand(self.n, 3))
            released = ((ph == RELEASE) | (ph == RETURN)) & (cur_hand == h)
            sh_rel = ((ph == SH_RELEASE) | (ph == SH_MOVE)) & (shift_hand == h) & has_shift
            lift = (released | sh_rel).float().unsqueeze(-1) * (-palm_n * c["retract"])
            # holding alone (the other hand is released / moving / regripping): wrap - palm closer, ring + pinky curl in
            other_busy = ((((ph == RELEASE) | (ph == RETURN) | (ph == REGRIP)) & (cur_hand == 1 - h))
                          | (((ph == SH_RELEASE) | (ph == SH_MOVE) | (ph == SH_GRIP)) & (shift_hand == 1 - h) & has_shift))  # fmt: skip
            alone = other_busy & ~released & ~sh_rel
            wrap_in = alone.float().unsqueeze(-1) * (palm_n * c["wrap_palm"])
            a[:, sl][:, 0:3] = (shift_env + lift + wrap_in + bl * d_pos) / pos_range
            # fingers
            twisting = hands[:, h] & ((ph == TWIST) | (ph == UNDO))
            amount = torch.where(twisting, c["press"], c["hold_press"])
            amount = torch.where(released | sh_rel, -c["open_amount"], amount)
            ramp_phase = ((ph == REGRIP) & (cur_hand == h)) | ((ph == SH_GRIP) & (shift_hand == h) & has_shift)
            ramp = (self.t_phase / c["regrip_s"]).clamp(max=1.0)
            # force closure: after the ramp keep closing (up to max_press) while fewer than min_pads pads touch
            pads_h = pads_on[:, h]
            extra = ((self.t_phase - c["regrip_s"]).clamp(min=0.0) * c["close_rate"]).clamp(max=c["max_press"])
            close = torch.where(pads_h >= c["min_pads"], torch.full_like(extra, c["regrip_press"]), c["regrip_press"] + extra)
            amount = torch.where(ramp_phase, -c["open_amount"] * (1.0 - ramp) + torch.where(ramp >= 1.0, close, 0.0), amount)
            amount = torch.where(ph == SETTLE, torch.zeros_like(amount), amount)
            d_fing = bl * (e.regrip_fingers[side] - e.finger_grasp[side])
            a[:, sl][:, 6:] = ((d_fing + amount.unsqueeze(-1) * self.flex + alone.float().unsqueeze(-1) * c["wrap_flex"] * self.wrap) / fr).clamp(-1, 1)
        return a.clamp(-1, 1)
