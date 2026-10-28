"""Complete U/U' primitives with a fixed wrist and an explicit release/hold phase.

Actor observations deliberately retain the legacy privileged layout for checkpoint
warm starts. This is training infrastructure, not a deployed blind-solving actor.
"""
from __future__ import annotations

import torch
import isaaclab.utils.math as mu
from rubik_lab.moves import ALL_MOVES, MOVE_INDEX
from rubik_lab.primitive_cycle import CycleMonitor, PoseWindow, limit_finger_command, pad_progress_bonus, participant_finger_metrics
from .env import BimanualTwistEnv, SIDES


class PrimitiveCycleEnv(BimanualTwistEnv):
    def __init__(self, cfg, **kwargs):
        self._cycle_initialized = False
        super().__init__(cfg, **kwargs)
        # Both independent policies start from the same geometry-feasible grasp set.
        if self.grasp_cache is None or "flick_reach" not in self.grasp_cache:
            raise ValueError("Complete-cycle tasks require a verified reach-annotated cache")
        shared = torch.ones(len(self.grasp_cache["cube_pos"]), device=self.device)
        for move in cfg.flick_moves:
            mid = MOVE_INDEX[move]
            shared *= self._grasp_move_mask[mid] * (self.grasp_cache["flick_reach"][:, mid] >= cfg.flick_reach_min)
        if not (shared > 0).any():
            raise ValueError("No common feasible grasp for all cycle directions")
        for move in cfg.flick_moves:
            self._grasp_move_mask[MOVE_INDEX[move]] = shared
        print(f"[cycle] shared feasible grasps: {int((shared > 0).sum())}", flush=True)
        n, dev = self.num_envs, self.device
        self.cycle_monitor = CycleMonitor(n, dev, self.step_dt, cfg.cycle_stable_s)
        self.cycle_pose_window = PoseWindow(n, dev, cfg.cycle_stable_pos, cfg.cycle_stable_rot)
        self.cycle_angle_origin = torch.zeros(n, device=dev)
        self.cycle_wrist_pos_peak = torch.zeros(n, device=dev)
        self.cycle_wrist_rot_peak = torch.zeros(n, device=dev)
        self.cycle_finger_error = torch.zeros(n, device=dev)
        self.cycle_prev_finger_error = torch.zeros(n, device=dev)
        self.cycle_finger_speed = torch.zeros(n, device=dev)
        self.cycle_release_force = torch.zeros(n, device=dev)
        self.cycle_motion_ok = torch.ones(n, dtype=torch.bool, device=dev)
        self.cycle_step = torch.zeros(n, dtype=torch.long, device=dev)
        self.cycle_sequence_index = torch.zeros(n, dtype=torch.long, device=dev)
        self.cycle_best_angle_error = torch.full((n,),torch.pi,device=dev)
        self._cycle_initialized = True

    def _pre_physics_step(self, actions):
        actions=limit_finger_command(self.actions,actions,self.cfg.cycle_command_slew_rad_s,
                                     self.step_dt,self.cfg.residual_finger_range)
        super()._pre_physics_step(actions)

    def _cube_state(self):
        pos, quat, twist, vel = super()._cube_state()
        if self._cycle_initialized:
            twist = twist - self.cycle_angle_origin
        return pos, quat, twist, vel

    def _idle_offsets(self):
        # Fixed, deployable finger roles; no privileged link/layer contact mask.
        return -self.cfg.cycle_idle_lift * (
            self.finger_flex["index"] + self.finger_flex["pinky"]
        ).clamp(max=1.0)

    def _expand_flick_action(self, pol):
        n, p = self.num_envs, self.per_hand
        full = torch.zeros(n, 2 * p, device=self.device)
        idle = self._idle_offsets() / self.cfg.residual_finger_range
        hold = (self.finger_flex["thumb"] + self.finger_flex["middle"]).clamp(max=1.0)
        offsets = idle + hold * self.cfg.flick_hold_press / self.cfg.residual_finger_range
        full[:, 6:p] = offsets
        full[:, p + 6:] = offsets
        rows = torch.arange(n, device=self.device)
        h, cols = self.flick_hand[self.move_id], self.flick_cols[self.move_id]
        full[rows[:, None], (h * p + 6)[:, None] + cols] = pol
        return full.clamp(-1.0, 1.0)

    def _update_task_state(self):
        super()._update_task_state()
        if not self._cycle_initialized:
            return
        cfg = self.cfg
        self.cycle_step += 1
        pos, quat, _, _ = self._cube_state()
        pos_err = (pos - self.cube_center).norm(dim=-1)
        rot_err = mu.quat_error_magnitude(quat, self.cube_quat_goal)
        self.peak_pos = torch.maximum(self.peak_pos, pos_err)
        self.peak_rot = torch.maximum(self.peak_rot, rot_err)
        wp, wr = [], []
        for side in SIDES:
            dp, dr = self._ee_home_error(side)
            wp.append(dp.norm(dim=-1))
            wr.append(dr.norm(dim=-1))
        self.cycle_wrist_pos_peak = torch.maximum(self.cycle_wrist_pos_peak, torch.stack(wp, -1).amax(-1))
        self.cycle_wrist_rot_peak = torch.maximum(self.cycle_wrist_rot_peak, torch.stack(wr, -1).amax(-1))
        self.cycle_motion_ok = (
            (self.peak_pos <= cfg.cycle_cube_pos) & (self.peak_rot <= cfg.cycle_cube_rot)
            & (self.cycle_wrist_pos_peak <= cfg.cycle_wrist_pos)
            & (self.cycle_wrist_rot_peak <= cfg.cycle_wrist_rot)
        )
        q = torch.stack([self.hands[s].data.joint_pos[:, self.finger_ids[s]] for s in SIDES], 1)
        qd = torch.stack([self.hands[s].data.joint_vel[:, self.finger_ids[s]] for s in SIDES], 1)
        ready = torch.stack([self.finger_grasp[s] for s in SIDES], 1) + self._idle_offsets()
        lo = torch.stack([self.finger_lower[s] for s in SIDES], 1)
        hi = torch.stack([self.finger_upper[s] for s in SIDES], 1)
        ready = ready.clamp(min=lo, max=hi)
        forces = self.cube_contacts.forces()[:, 1]
        self.cycle_finger_error, self.cycle_finger_speed, self.cycle_release_force = self._completion_finger_metrics(q, qd, ready, forces)
        fingers_ready = (
            (self.cycle_finger_error <= cfg.cycle_finger_tol)
            & (self.cycle_finger_speed <= cfg.cycle_finger_speed)
            & (self.cycle_release_force <= cfg.cycle_release_force)
        )
        core = self.cube.data
        if cfg.cycle_stability_mode == "pose_window":
            # Contact solver velocities can jitter while the object pose remains
            # steady. Require bounded pose excursion throughout the dwell window.
            begin = self.cycle_monitor.stable_steps == 0
            quiet = self.cycle_pose_window.update(pos, quat, begin)
        else:
            quiet = core.body_lin_vel_w[:, self.core_id].norm(dim=-1) <= cfg.cycle_cube_speed
            quiet &= core.body_ang_vel_w[:, self.core_id].norm(dim=-1) <= cfg.cycle_core_ang_speed
        pads = (self.layer_f.sum(-1) > cfg.tactile_contact_threshold).sum(-1)
        grasp_ok = (pads >= cfg.home_min_pads).all(-1) & quiet
        finite = torch.isfinite(pos).all(-1) & torch.isfinite(self.signed_err)
        self.primitive_done = self.cycle_monitor.update(
            layer_ok=self.move_done, fingers_ready=fingers_ready, grasp_ok=grasp_ok,
            motion_ok=self.cycle_motion_ok, failed=self._dropped(pos) | ~finite,
        )

    def _completion_finger_metrics(self, q, qd, ready, forces):
        # Diagnostic multi-finger controllers override this hook with a persistent
        # participant set; never change move_id or the angle origin to route a hand.
        hands = self.flick_hand[self.move_id, None]
        return participant_finger_metrics(
            q, qd, ready, forces, hands, self.flick_cols[self.move_id, None],
            self.flick_links[self.move_id, None], torch.ones_like(hands, dtype=torch.bool),
        )

    def _get_rewards(self):
        reward = super()._get_rewards()
        if not self._cycle_initialized:
            return reward
        rows=torch.arange(self.num_envs,device=self.device)
        pad_force=self.layer_f[rows,self.flick_hand[self.move_id],self.flick_pad[self.move_id],1]
        self.cycle_best_angle_error,pad_bonus=pad_progress_bonus(self.cycle_best_angle_error,
            self.signed_err,pad_force,self.cycle_release_force,self.cfg.rew_pad_progress)
        # Return shaping is activated only after the layer has reached its target.
        # The actor still controls retraction; there is no scripted success shortcut.
        returning = self.layer_done_latch.float()
        extra = self.cfg.cycle_return_reward * self.cycle_finger_error * returning
        improvement = (self.cycle_prev_finger_error - self.cycle_finger_error).clamp(-0.2, 0.2)
        extra += self.cfg.cycle_return_progress * improvement * returning * (~self.first_reach).float()
        self.cycle_prev_finger_error.copy_(self.cycle_finger_error)
        extra += self.cfg.cycle_hold_reward * self.cycle_monitor.ready.float()
        extra += self.cfg.cycle_violation_penalty * (~self.cycle_motion_ok).float()
        self.extras["log"].update({
            "cycle_ready_rate": self.cycle_monitor.ready.float().mean(),
            "cycle_violation_rate": (~self.cycle_motion_ok).float().mean(),
            "cycle_finger_error": self.cycle_finger_error.mean(),
            "cycle_wrist_pos_mm": self.cycle_wrist_pos_peak.mean() * 1000,
            "cycle_wrist_rot_deg": torch.rad2deg(self.cycle_wrist_rot_peak).mean(),
        })
        return reward + torch.nan_to_num(extra+pad_bonus, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_dones(self):
        terminated, timeout = super()._get_dones()
        if self._cycle_initialized:
            if not self.cfg.cycle_terminate_on_success:
                # Rebuild termination: never mask a drop/broken state just because
                # a success flag happened on that step.
                pos = self._cube_state()[0]
                broken = ~torch.isfinite(self.cube.data.root_state_w).all(-1)
                for hand in self.hands.values():
                    broken |= ~torch.isfinite(hand.data.joint_pos).all(-1) | ~torch.isfinite(hand.data.joint_vel).all(-1)
                terminated = self._dropped(pos) | broken
            terminated |= self.cycle_monitor.failed
        return terminated, timeout

    def _snapshot(self, dropped):
        super()._snapshot(dropped)
        if self._cycle_initialized:
            self.snapshot.update({k: v.clone() for k, v in {
                "tactile_timing": self.tactile.timing,
                "cycle_ready": self.cycle_monitor.ready,
                "cycle_failed": self.cycle_monitor.failed,
                "cycle_stable_steps": self.cycle_monitor.stable_steps,
                "cycle_step": self.cycle_step,
                "sequence_index": self.cycle_sequence_index,
                "finger_ready_error": self.cycle_finger_error,
                "finger_speed": self.cycle_finger_speed,
                "release_force": self.cycle_release_force,
                "peak_wrist_pos": self.cycle_wrist_pos_peak,
                "peak_wrist_rot": self.cycle_wrist_rot_peak,
                "angle_origin": self.cycle_angle_origin,
            }.items()})

    def _reset_idx(self, env_ids=None):
        super()._reset_idx(env_ids)
        if not self._cycle_initialized:
            return
        ids = self.cube._ALL_INDICES if env_ids is None else env_ids
        self.wrist_mask[ids] = 0.0  # both wrists fixed, regardless of legacy move tables
        self.cycle_monitor.reset(ids)
        for field in (self.cycle_angle_origin, self.cycle_wrist_pos_peak,
                      self.cycle_wrist_rot_peak, self.cycle_finger_error,
                      self.cycle_prev_finger_error, self.cycle_finger_speed, self.cycle_release_force,
                      self.cycle_step, self.cycle_sequence_index):
            field[ids] = 0
        self.cycle_motion_ok[ids] = True
        self.cycle_best_angle_error[ids] = self.prev_twist_err[ids]
        if "cycle_ready" in self.grasp_cache:
            # Cache actual states were prepared with both index fingers released.
            # Match PD targets and EMA state, otherwise reset would immediately
            # pull the fingers back onto the layer before the actor can react.
            ready_action = self._idle_offsets()[self.flick_cols[self.move_id]] / self.cfg.residual_finger_range
            full = self._expand_flick_action(ready_action)
            self.action_smooth[ids] = full[ids]
            self.actions[ids] = ready_action[ids]
            self.prev_actions[ids] = ready_action[ids]
            for h, side in enumerate(SIDES):
                target = self.finger_grasp[side][ids] + full[ids, h*self.per_hand+6:(h+1)*self.per_hand] * self.cfg.residual_finger_range
                target = target.clamp(min=self.finger_lower[side][ids], max=self.finger_upper[side][ids])
                self.finger_tgt[side][ids] = target
                self.hands[side].set_joint_position_target(target, joint_ids=self.finger_ids[side], env_ids=ids)

    def advance_cycle(self, ids, move: str):
        """Change same-face goal without a physical reset or a new pose reference.

        Only a diagnostic sequence runner may call this, after ground-truth full
        completion. A tactile executor needs its own validated trigger later.
        """
        if self.cfg.cycle_terminate_on_success:
            raise RuntimeError("Sequence mode requires cycle_terminate_on_success=False")
        mid = MOVE_INDEX[move]
        if move not in self.cfg.flick_moves:
            raise ValueError(f"Move {move} has no finger mapping in this task")
        if not self.cycle_monitor.done[ids].all():
            raise RuntimeError("Cannot advance an incomplete primitive")
        if not (self.move_face_id[self.move_id[ids]] == self.move_face_id[mid]).all():
            raise ValueError("Two-body continuation supports only the same face")
        # Preserve the IDEAL aligned target, not the measured residual angle.
        # Otherwise small errors would silently accumulate across the sequence.
        self.cycle_angle_origin[ids] += self.move_target[self.move_id[ids]]
        self.move_id[ids] = mid
        self.cycle_monitor.reset(ids)
        self.move_done[ids] = False
        self.primitive_done[ids] = False
        self.layer_done_latch[ids] = False
        self.first_reach[ids] = False
        self.cycle_step[ids] = 0
        self.cycle_sequence_index[ids] += 1
        self.prev_twist_err[ids] = mu.wrap_to_pi(
            self.move_target[mid] - self._cube_state()[2][ids]
        ).abs()
        self.cycle_best_angle_error[ids] = self.prev_twist_err[ids]
        # No write_*_to_sim, no reset(), no home-reference or smoothing reset.


class ContinuousPrefixTrainingEnv(PrimitiveCycleEnv):
    """Two physical primitives per episode; frozen prefix is routed by trainer.

    First completion must be handed off explicitly using advance_preserving_state.
    Each primitive keeps the original deadline; only the total episode doubles.
    """
    def __init__(self,cfg,prefix_move='U',**kwargs):
        if prefix_move not in cfg.flick_moves:
            raise ValueError('Prefix move has no configured finger mapping')
        self.primitive_seconds=float(cfg.episode_length_s)
        cfg.episode_length_s=2*self.primitive_seconds
        cfg.cycle_terminate_on_success=False
        super().__init__(cfg,**kwargs)
        import math
        self.primitive_step_limit=math.ceil(self.primitive_seconds/self.step_dt)
        self.forced_move_id=torch.full((self.num_envs,),MOVE_INDEX[prefix_move],dtype=torch.long,device=self.device)

    def _get_dones(self):
        terminated,timeout=super()._get_dones()
        if self._cycle_initialized and hasattr(self,'primitive_step_limit'):
            terminated|=(self.cycle_sequence_index>0)&self.primitive_done
            timeout|=(self.cycle_step>=self.primitive_step_limit)&~self.primitive_done
        return terminated,timeout
