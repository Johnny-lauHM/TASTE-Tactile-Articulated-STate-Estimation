"""Fixed-quota continuous two-action diagnosis; truth gates are diagnostic only."""
import argparse, json
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--out',required=True)
p.add_argument('--record-probe-wrenches',action='store_true',help='Read-only selectedenv raw topbody contact andfriction manifolds')
p.add_argument('--probe-wrench-start',choices=['episode','intervention'],default='episode')
p.add_argument('--live-whole-hand',action='store_true',help='Pause integration to plan from actualstate; privileged, not real-time')
p.add_argument('--live-prefix-degrees',type=int,default=15)
p.add_argument('--live-released-start',action='store_true',help='Startdiagnosis fromreleasedslowfinger withoutrequiringreadyposition; completionrequirements unchanged')
p.add_argument('--live-support-contact',action='store_true',help='Use contact-preserving support geometry in live diagnostic planner')
p.add_argument('--live-support-reserve-m',type=float,default=0.)
p.add_argument('--live-ready-return',action='store_true',help='Audit actualstate indexwithdrawal after4arrivalvalidframes; originalphysicalmonitor decides completion')
p.add_argument('--live-index-feedback',action='store_true',help='After prefix+.25s apply bounded index proprio P feedback; unchanged physical limits')
p.add_argument('--live-contact-transport',action='store_true',help='Require tangential pad transport, independently audited before execution')
p.add_argument('--live-transport-reserve-m',type=float,default=0.)
p.add_argument('--live-contact-lateral-max-m',type=float,help='Narrow contact patch toward face interior; independently audited')
p.add_argument('--live-contact-face',choices=['R','B'])
p.add_argument('--live-hand-pairs-from',type=Path,help='Transfer audited collision pair identities, never prior physical poses')
p.add_argument('--whole-hand-path',type=Path,help='Privileged prefix-only actuator probe; never a completed-action claim')
p.add_argument('--path-trial',type=int,default=0)
p.add_argument('--path-env',type=int,default=2)
p.add_argument('--path-end',type=int,help='Explicit final waypoint of the diagnostic prefix')
p.add_argument('--path-servo-bias',action='store_true',help='Preserve initial actuator references when replaying physical-joint increments')
p.add_argument('--residual-recontact',action='store_true',help='Privileged single opposite-index correction after original withdrawal; persistent two-finger completion')
p.add_argument('--recontact-press',type=float,default=.003)
p.add_argument('--recontact-dense-clearance',action='store_true')
p.add_argument('--second-checkpoint-u')
p.add_argument('--checkpoint-up',default='logs/rsl_rl/rubik_primitive_cycle/Up_blind_ppo_v1/model_200.pt')
p.add_argument('--num-envs',type=int,default=32)
p.add_argument('--rounds',type=int,default=2)
p.add_argument('--seed',type=int,default=1210)
p.add_argument('--handoff-wait-steps',type=int,default=0)
p.add_argument('--after-left-return-move',choices=['U',"U'"],help='Append original actor after complete left withdrawal; retain all physical state and original references')
p.add_argument('--post-return-move',choices=['U',"U'"],help='Append one frozen-actor action to selected audited-return environment; no physical reset')
p.add_argument('--post-return-left-path',type=Path,help='Independently audited left full withdrawal after exact source history; original monitor and all-body release score completion')
p.add_argument('--fourth-recontact-surface-depth-m',type=float,choices=(0.,.0004,.0008),default=0.)
p.add_argument('--coordinated-contact-bind-actual',action='store_true',help='Reuse only exact geometric inputs, retain actual sensor history and independently re-audit before execution')
p.add_argument('--coordinated-contact-path',type=Path,help='Independently audited full-hand contact arc after completed recontact; privileged diagnostic only')
p.add_argument('--contact-entry-delay-frames',type=int,choices=[1,10],default=10,help='Internal pre-window delay; always retain10consecutive stable contact frames')
p.add_argument('--completed-recontact-feedback',action='store_true',help='Gain3 previous-proprioception feedback during completed recontact hold only')
p.add_argument('--completed-recontact-outside-m',type=float,choices=[.0003,.0005],default=.0003,help='Completed-cycle recontact FK point outside offset; same frozen geometry gates')
p.add_argument('--completed-lift-feedback',action='store_true',help='Gain3 previous-proprioception feedback during completed-cycle lift hold only')
p.add_argument('--recycle-completed-stroke',action='store_true',help='Actual completed secondstroke release/recontact diagnostic, preserving original target and clock')
p.add_argument('--second-stroke-degrees',type=float,choices=[5.,10.],default=5.,help='Declared diagnostic secondstroke extent; originalquartertarget retained')
p.add_argument('--second-stroke-feedback-gain',type=float,choices=[1.,3.],default=1.,help='Declared hold feedback gain; same correction/rate/actuatorlimits')
p.add_argument('--second-stroke-feedback',action='store_true',help='Hold-only previous-proprioception feedback after secondstroke duration+.25s, gain1/cap.1rad/1radpersec')
p.add_argument('--second-stroke',action='store_true',help='Second actual-source declared-angle drive after internal recontact, within unchanged original goal/clock')
p.add_argument('--live-stroke-cycle',action='store_true',help='Generate and independently audit internal lift/recontact from each actual causal source')
p.add_argument('--stroke-recontact-path',type=Path,help='Audited outside reset and recontact after internal lift; retains original target/clock')
p.add_argument('--stroke-lift-path',type=Path,help='Audited internal lift after quiet small stroke; retains original target and deadline')
p.add_argument('--contact-turn-depth-reserve-m',type=float,choices=[0.,.0001],default=0.,help='Diagnostic cube-face normal depth reserve, ramped over firstdegree; original contact limits unchanged')
p.add_argument('--contact-turn-path',type=Path,help='Audited diagnostic candidate, requires exact firsteligible actualhistory; no state replay or fullaction credit')
p.add_argument('--contact-turn',action='store_true',help='Audited5degree diagnostic after causal contact entry; no full action credit')
p.add_argument('--fourth-recontact-center-m',type=float,choices=(0.,.004),default=0.)
p.add_argument('--fourth-recontact-press-m',type=float,choices=(.001,.0015),default=.001)
p.add_argument('--fourth-recontact',action='store_true',help='Privileged actual-state fingertip contact component after full third Uprime; no fourth turn completion credit')
p.add_argument('--live-left-withdrawal',action='store_true',help='Truth-only actual-state audited left residual/full withdrawal; no fixed-history replay')
p.add_argument('--diagnostic-preload-escape-path',type=Path,help='Small audited residual-contact displacement after exact recorded preload history; no completion credit')
p.add_argument('--diagnostic-preload-at-error-deg',type=float,help='Truth-only diagnosis: start rate-limited preload removal before target at this remaining angle; no completion credit')
p.add_argument('--post-return-relax-after-steps',type=int,help='Diagnostic preload removal: ramp held index motors to latched actual q at1rad/s; no physical state freeze or completion credit')
p.add_argument('--post-return-escape-after-steps',type=int,help='Diagnostic first withdrawal segment: after N tail frames move held left index joint6 by -.4rad at1rad/s; no completion credit')
p.add_argument('--post-return-hold-after-steps',type=int,help='Diagnostic only: hold previous active index motor target after N third-action frames; no completion credit')
p.add_argument('--second-controller',choices=['policy','replay','cartesian-replay'],default='policy')
p.add_argument('--support-delta',type=float,default=0.0)
p.add_argument('--physics-hz',type=int,default=240)
p.add_argument('--friction-law',choices=['legacy','impulse-cap','implicit-damping'],default='legacy')
p.add_argument('--forces-each-iteration',action='store_true')
p.add_argument('--progress-gate',action='store_true')
p.add_argument('--replay-pad-normal',action='store_true')
p.add_argument('--record-tracking',action='store_true')
p.add_argument('--tracking-iters',type=int,default=12)
p.add_argument('--template-target',choices=['raw','filtered'],default='raw')
p.add_argument('--compensate-output-ema',action='store_true')
p.add_argument('--replay-time-scale',type=float,default=1.0)
p.add_argument('--reversal-withdrawal',action='store_true')
p.add_argument('--withdrawal-concurrent-return',action='store_true',help='Diagnostic: after radial clearance return all active joints concurrently at the existing rate')
p.add_argument('--audit-handoff',action='store_true')
p.add_argument('--record-release-substeps',action='store_true')
p.add_argument('--capture-release-states',action='store_true')
p.add_argument('--release-at-task-tolerance',action='store_true',help='Use unchanged task angle tolerance for withdrawal phase instead of diagnostic3deg margin')
p.add_argument('--hold-then-withdraw',action='store_true',help='After terminal hold reaches abs(error)<3deg and abs(speed)<.25 for4samples, perform existing radial withdrawal')
p.add_argument('--terminal-command-hold',action='store_true',help='Causal diagnostic: latch last command on first UU near-target low-speed state; no withdrawal')
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
if (a.live_support_contact or a.live_support_reserve_m or a.live_contact_face or a.live_hand_pairs_from or a.live_released_start or a.live_contact_transport or a.live_transport_reserve_m or a.live_index_feedback or a.live_ready_return or a.live_contact_lateral_max_m is not None) and not a.live_whole_hand:raise ValueError('Live support options require live whole-hand probe')
out=Path(a.out)
if a.record_probe_wrenches and not (a.live_whole_hand or a.whole_hand_path):raise ValueError('Probe wrenches require explicit probe cohort')
if a.probe_wrench_start!='episode' and not a.record_probe_wrenches:raise ValueError('Wrench start requires wrench recording')
if a.replay_pad_normal and a.second_controller!='cartesian-replay':raise ValueError('Pad-normal replay requires Cartesian replay')
if a.second_checkpoint_u and a.second_controller!='policy':raise ValueError('Separate second U checkpoint requires policy controller')
if a.capture_release_states and not a.record_release_substeps:raise ValueError('State capture requires release substeps')
if a.record_release_substeps and not a.hold_then_withdraw:raise ValueError('Release substeps require hold-then-withdraw')
if a.release_at_task_tolerance and not a.hold_then_withdraw:raise ValueError('Task-tolerance release requires hold-then-withdraw')
if a.hold_then_withdraw and not a.terminal_command_hold:raise ValueError('Hold-then-withdraw requires terminal command hold')
if a.terminal_command_hold and (a.reversal_withdrawal or a.second_controller!='cartesian-replay'):raise ValueError('Terminal hold requires Cartesian replay without withdrawal')
if a.withdrawal_concurrent_return and not a.reversal_withdrawal:raise ValueError('Concurrent return requires reversal withdrawal')
if a.reversal_withdrawal and a.second_controller!='cartesian-replay':raise ValueError('Reversal withdrawal requires Cartesian diagnostic')
if a.replay_time_scale<1 or (a.replay_time_scale!=1 and (a.second_controller!='cartesian-replay' or a.progress_gate)):raise ValueError('Time dilation requires Cartesian replay without progress gate and scale>=1')
if a.progress_gate and a.second_controller!='cartesian-replay':raise ValueError('Progress gate requires Cartesian replay diagnostic')
if a.physics_hz%120:raise ValueError('Physics rate must preserve exact40Hz control and30Hz sensing periods')
if a.num_envs%4 or a.handoff_wait_steps<0:raise ValueError('Four equal sequence groups and nonnegative wait required')
if out.with_suffix('.json').exists() or out.with_suffix('.npz').exists():raise FileExistsError(out)
if a.residual_recontact and not (a.hold_then_withdraw and a.release_at_task_tolerance):raise ValueError('Recontact requires hold-then-withdraw at unchanged task tolerance')
if a.live_whole_hand and (a.whole_hand_path or not 1<=a.live_prefix_degrees<=50 or not 0<=a.path_env<a.num_envs//4 or not 0<=a.path_trial<a.rounds):raise ValueError('Live probe requires validsingleUU source andprefix; not cachedpath')
if a.whole_hand_path and (a.path_end is None or not 0<=a.path_env<a.num_envs//4 or not 0<=a.path_trial<a.rounds):raise ValueError('Wholehand probe requires explicitprefix and validUU trial/env')
if a.post_return_move and not (a.live_whole_hand and a.live_ready_return):raise ValueError('Post-return action requires live audited withdrawal')
if a.completed_recontact_outside_m!=.0003 and not a.recycle_completed_stroke:raise ValueError('Completed recontact offset requires recycle')
if a.contact_entry_delay_frames!=10 and not a.contact_turn:raise ValueError('Contact timing requires contactturn')
if a.completed_recontact_feedback and not a.recycle_completed_stroke:raise ValueError('Recontact feedback requires recycle')
if a.completed_lift_feedback and not a.recycle_completed_stroke:raise ValueError('Lift feedback requires completed recycle')
if a.recycle_completed_stroke and not a.second_stroke_feedback:raise ValueError('Completed stroke recycle requires secondstroke feedback')
if a.second_stroke_degrees!=5. and not a.second_stroke:raise ValueError('Secondstroke extent requires secondstroke')
if a.second_stroke_feedback_gain!=1. and not a.second_stroke_feedback:raise ValueError('Feedback gain requires feedback')
if a.second_stroke_feedback and not a.second_stroke:raise ValueError('Secondstroke feedback requires secondstroke')
if a.second_stroke and not a.live_stroke_cycle:raise ValueError('Second stroke requires live internal cycle')
if a.live_stroke_cycle and (not a.contact_turn or a.stroke_lift_path or a.stroke_recontact_path):raise ValueError('Live internal cycle requires contact turn and excludes prior-run internal candidates')
if a.stroke_recontact_path and not a.stroke_lift_path:raise ValueError('Internal recontact requires lift')
if a.stroke_lift_path and not a.contact_turn:raise ValueError('Internal lift requires contact turn')
if a.contact_turn_depth_reserve_m and (not a.contact_turn or a.contact_turn_path):raise ValueError('Turn reserve requires live contact turn generation')
if a.contact_turn_path and not a.contact_turn:raise ValueError('Candidate requires contact turn')
if a.coordinated_contact_bind_actual and not a.coordinated_contact_path:raise ValueError('Actual binding requires coordinated candidate')
if a.coordinated_contact_path and (not a.completed_recontact_feedback or a.contact_entry_delay_frames!=1):raise ValueError('Coordinated path requires completed recontact feedback and declared entry timing')
if a.contact_turn and (not a.fourth_recontact or a.fourth_recontact_center_m!=.004):raise ValueError('Contact turn requires center4 recontact')
if a.fourth_recontact and (not a.live_left_withdrawal or a.after_left_return_move!="U'"):raise ValueError('Fourth recontact requires liveleft return then Uprime')
if a.live_left_withdrawal and (a.diagnostic_preload_at_error_deg is None or a.diagnostic_preload_escape_path is not None or a.post_return_left_path is not None):raise ValueError('Live left requires early preload and excludes fixed left paths')
if a.diagnostic_preload_escape_path is not None and a.diagnostic_preload_at_error_deg is None:raise ValueError('Residual escape requires preload diagnostic')
if a.diagnostic_preload_at_error_deg is not None and (a.post_return_move!="U'" or not 0<a.diagnostic_preload_at_error_deg<90 or a.post_return_hold_after_steps is not None or a.post_return_relax_after_steps is not None or (a.post_return_left_path is not None and a.diagnostic_preload_escape_path is None)):raise ValueError('Preload braking requires Uprime and excludes fixed-time overlays')
if a.after_left_return_move and a.post_return_left_path is None and not a.live_left_withdrawal:raise ValueError('Fourth action requires audited left return')
if a.post_return_left_path is not None and ((a.post_return_relax_after_steps is None and a.diagnostic_preload_escape_path is None) or not a.live_ready_return):raise ValueError('Left path requires preload relaxation or audited residual escape and audited first return')
if a.post_return_relax_after_steps is not None and (a.post_return_hold_after_steps is None or a.post_return_relax_after_steps<=a.post_return_hold_after_steps or a.post_return_escape_after_steps is not None):raise ValueError('Relax requires earlier hold and excludes escape')
if a.post_return_escape_after_steps is not None and (a.post_return_hold_after_steps is None or a.post_return_escape_after_steps<=a.post_return_hold_after_steps):raise ValueError('Escape requires earlier left motor hold')
if a.post_return_hold_after_steps is not None and (a.post_return_move!="U'" or a.post_return_hold_after_steps<1):raise ValueError('Motor-hold diagnosis requires U-prime tail and positive step count')
app=AppLauncher(a).app
import numpy as np
import torch
from rubik_lab.tasks.bimanual_twist.cycle_cfg import CycleUEnvCfg
from rubik_lab.tasks.bimanual_twist.cycle_env import PrimitiveCycleEnv
from rubik_lab.primitive_config import restore_checkpoint_settings,validate_sequence_policy_settings
from rubik_lab.blind_actor import BlindInference
from rubik_lab.primitive_execution import apply_ready_hold,command_for_filtered_target
from rubik_lab.passive_friction import bounded_friction_torque,friction_damping
from rubik_lab.moves import MOVE_INDEX
paths=['logs/rsl_rl/rubik_primitive_cycle/U_blind_full/model.pt',a.checkpoint_up]
cfg=CycleUEnvCfg();restore_checkpoint_settings(cfg,paths[0]);validate_sequence_policy_settings(cfg,paths[1])
if a.second_checkpoint_u:validate_sequence_policy_settings(cfg,a.second_checkpoint_u)
cfg.scene.num_envs=a.num_envs;cfg.seed=a.seed;cfg.grasp_cache_path='cache/g70s_cycle_demo1_v3.npz'
cfg.moves=('U',"U'");cfg.cycle_terminate_on_success=False;cfg.gravity_curriculum=False
if a.compensate_output_ema and (a.second_controller!='cartesian-replay' or a.template_target!='filtered'):raise ValueError('EMA compensation requires filtered Cartesian template')
if (a.template_target=='filtered' or a.compensate_output_ema) and cfg.cycle_command_slew_rad_s!=0:raise ValueError('Filtered template requires zero slew; include slew transformation before enabling otherwise')
release_angle_tolerance=cfg.move_success_tol if a.release_at_task_tolerance else np.deg2rad(3.)
primitive_seconds=cfg.episode_length_s;cfg.episode_length_s*=4 if a.after_left_return_move else (3 if a.post_return_move else 2)
# Preserve legacy physical sample/hold cadence and quantized20.833ms latency.
legacy_latency=round(cfg.tactile.latency/cfg.sim.dt)*cfg.sim.dt
cfg.sim.dt=1/a.physics_hz;cfg.decimation=a.physics_hz//40;cfg.sim.render_interval=cfg.decimation
cfg.tactile.latency=legacy_latency
cfg.sim.physx.enable_external_forces_every_iteration=a.forces_each_iteration
if a.friction_law!="legacy" and (cfg.detent_torque or cfg.notch_torque):raise ValueError("Friction-only diagnostic requires existing zero detent/notch settings")
def escape_motor_target(original,elapsed,lower,upper):
    if elapsed<0:raise ValueError('Negative escape time')
    target=original.clone();target[1]-=min(float(elapsed),.4)
    if not bool(((target>=lower)&(target<=upper)).all()):raise ValueError('Escape motor target outside original actuator limits')
    return target

def relax_motor_target(original,physical_reference,elapsed,lower,upper):
    if elapsed<0:raise ValueError('Negative relaxation time')
    if not bool(((physical_reference>=lower)&(physical_reference<=upper)).all()):raise ValueError('Latched physical reference outside actuator limits')
    target=original+(physical_reference-original).clamp(min=-float(elapsed),max=float(elapsed))
    if not bool(((target>=lower)&(target<=upper)).all()):raise ValueError('Relax motor target outside actuator limits')
    return target

def preload_brake_trigger(error_rad,layer_speed,threshold_deg,completed_cycle_steps):
    return bool(completed_cycle_steps>0 and np.isfinite([error_rad,layer_speed,threshold_deg]).all() and 0<error_rad<=np.deg2rad(threshold_deg) and layer_speed>0)

class TraceEnv(PrimitiveCycleEnv):
    def _pre_physics_step(self,actions):
        super()._pre_physics_step(actions)
        hold=getattr(self,'post_return_hold_target',None)
        if hold is not None:
            i,side,cols,values=hold
            self.finger_tgt[side][i,cols]=values
        recontact=getattr(self,'fourth_recontact_target',None)
        if recontact is not None:
            i,side,cols,values=recontact
            self.finger_tgt[side][i,cols]=values
        coordinated=getattr(self,'coordinated_wrist_target',None)
        if coordinated is not None:
            i,side,position,quat=coordinated
            if not bool(((position>=self.ws_lo)&(position<=self.ws_hi)).all()):raise ValueError('Coordinated wrist outside workspace')
            self.ee_pos_tgt[side][i]=position;self.ee_quat_tgt[side][i]=quat
            self._base_tgt[side]=self.ctrl[side].joint_targets(self.ee_pos_tgt[side]+self.scene.env_origins,self.ee_quat_tgt[side])
        target=getattr(self,'whole_hand_probe_target',None)
        if target is None:return
        i,value=target
        tensor=lambda v:torch.as_tensor(v,device=self.device,dtype=self.finger_tgt['right'].dtype)
        q=tensor(value['fingers']);position=tensor(value['wrist_position']);quat=tensor(value['wrist_quaternion'])
        if not bool(((q>=self.finger_lower['right'][i])&(q<=self.finger_upper['right'][i])).all()):raise ValueError('Probe jointtarget outside actuatorbounds')
        if not bool(((position>=self.ws_lo)&(position<=self.ws_hi)).all()):raise ValueError('Probe wristtarget outside workspace')
        self.finger_tgt['right'][i]=q
        self.ee_pos_tgt['right'][i]=position;self.ee_quat_tgt['right'][i]=quat
        self._base_tgt['right']=self.ctrl['right'].joint_targets(self.ee_pos_tgt['right']+self.scene.env_origins,self.ee_quat_tgt['right'])
    def _apply_action(self):
        super()._apply_action()
        if a.record_release_substeps and hasattr(self,"diagnostic_withdrawal_stage") and hasattr(self,"diagnostic_live"):
            ids=(self.diagnostic_live & (self.diagnostic_withdrawal_stage>=2) & (torch.arange(self.num_envs,device=self.device)<self.num_envs//4)).nonzero().flatten()
            if len(ids):
                if a.capture_release_states:
                    fresh=ids[~self.diagnostic_captured[ids]]
                    if len(fresh):
                        state=self.scene.get_state(is_relative=True)['articulation']
                        for name,fields in state.items():
                            asset=self.scene[name]
                            fields.update({field:getattr(asset.data,field).clone() for field in ('joint_pos_target','joint_vel_target','joint_effort_target')})
                            for field,value in fields.items():
                                key=f'terminal_state__{name}__{field}'
                                self.release_states.setdefault(key,[]).append(value[fresh].cpu().numpy())
                        meta=torch.stack([torch.full_like(fresh,self.diagnostic_trial),fresh,torch.full_like(fresh,self.tactile.step_count)],-1)
                        self.release_states.setdefault('terminal_state_index',[]).append(meta.cpu().numpy())
                        self.diagnostic_captured[fresh]=True
                q=self.cube.data.joint_pos[ids,self.twist_id]
                v=self.cube.data.joint_vel[ids,self.twist_id]
                contact=self.cube_contacts.forces()[ids,1].sum((-1,-2))
                self.release_substeps.append(torch.stack([torch.full_like(q,self.diagnostic_trial),torch.full_like(q,self.tactile.step_count),ids,q,v,contact,self.cube.data.body_ang_vel_w[ids,self.core_id].norm(dim=-1)],-1).cpu().numpy())
        if a.friction_law=="impulse-cap":
            omega=self.cube.data.joint_vel[:,self.twist_id:self.twist_id+1]
            bound=self.cube.data.joint_armature[:,self.twist_id:self.twist_id+1]
            tau=bounded_friction_torque(omega,self.cfg.twist_friction_torque,0.05,self.physics_dt,bound)
            self.cube.set_joint_effort_target(tau,joint_ids=[self.twist_id])
        elif a.friction_law=="implicit-damping":
            omega=self.cube.data.joint_vel[:,self.twist_id:self.twist_id+1]
            damping=self.diagnostic_base_damping+friction_damping(omega,self.cfg.twist_friction_torque,0.05)
            self.cube.write_joint_damping_to_sim(damping,joint_ids=[self.twist_id])
            self.cube.set_joint_effort_target(torch.zeros_like(omega),joint_ids=[self.twist_id])
    def _expand_flick_action(self,pol):
        full=super()._expand_flick_action(pol)
        if hasattr(self,"diagnostic_support_adjust"):
            hold=(self.finger_flex["thumb"]+self.finger_flex["middle"]).clamp(max=1)
            delta=self.diagnostic_support_adjust[:,None]*hold/self.cfg.residual_finger_range
            for h in range(2):full[:,h*self.per_hand+6:(h+1)*self.per_hand]+=delta
        correction=getattr(self,'residual_recontact',None)
        if correction is not None:
            ids=(correction.used&self.diagnostic_live).nonzero().flatten()
            cols=self.flick_cols[correction.moves[ids]]
            full[ids[:,None],6+cols]=correction.command[ids]
        return full.clamp(-1,1)
    def _completion_finger_metrics(self,q,qd,ready,forces):
        correction=getattr(self,'residual_recontact',None)
        if correction is not None:return correction.completion_metrics(q,qd,ready,forces)
        return super()._completion_finger_metrics(q,qd,ready,forces)
    def _snapshot(self,dropped):
        super()._snapshot(dropped)
        self.snapshot['link_contact_force']=self.cube_contacts.forces().clone()
        self.snapshot['core_angular_speed']=self.cube.data.body_ang_vel_w[:,self.core_id].norm(dim=-1).clone()
        self.snapshot['ee_target_pose']=torch.stack([torch.cat([self.ee_pos_tgt[s],self.ee_quat_tgt[s]],-1) for s in ('left','right')],1).clone()
        self.snapshot['finger_velocity']=torch.stack([self.hands[s].data.joint_vel[:,self.finger_ids[s]] for s in ('left','right')],1).clone()
        recorder=getattr(self,'diagnostic_wrench_recorder',None)
        if recorder is not None and (a.probe_wrench_start=='episode' or getattr(self,'diagnostic_wrench_active',False)):self.snapshot['probe_wrench']=recorder.capture(self)
u=TraceEnv(cfg);u.record=True;u.release_substeps=[];u.release_states={}
wrench_recorder=None
if a.record_probe_wrenches:
    from rubik_lab.diagnostics.contact_wrench import ContactWrenchRecorder
    wrench_recorder=ContactWrenchRecorder(a.path_env);u.diagnostic_wrench_recorder=wrench_recorder
u.diagnostic_base_damping=u.cube.data.joint_damping[:,u.twist_id:u.twist_id+1].clone()
if a.friction_law=="impulse-cap" and (u.cube.data.joint_armature[:,u.twist_id]<=0).any():raise ValueError("Positive inertia lowerbound required")
u.diagnostic_support_adjust=torch.zeros(u.num_envs,device=u.device)
planner=None
if a.second_controller=="cartesian-replay":
    from rubik_lab.flick import FlickPlanner,to_cube_frame
    planner=FlickPlanner(u.device)
actors=[BlindInference(torch.load(path,map_location='cpu',weights_only=False),u.num_envs,u.device) for path in paths]
if a.second_checkpoint_u:actors.append(BlindInference(torch.load(a.second_checkpoint_u,map_location='cpu',weights_only=False),u.num_envs,u.device))
pairs=[('U','U'),('U',"U'"),("U'",'U'),("U'","U'")]
group=torch.arange(u.num_envs,device=u.device)//(u.num_envs//4)
first=torch.tensor([MOVE_INDEX[x[0]] for x in pairs],device=u.device)[group]
handoff_audits=[];post_return_events=[]
recontact_events=[];recontact_trace=[];core_speed_log=[];ready_target_log=[]
motor_target_log=[];wrist_target_log=[]
records=[];traces=[];sensory=[];contact=[];poses=[];actions_log=[];tracking=[];tracking_index=[];withdrawal_log=[];terminal_hold_log=[];template_target_error=0.0;output_target_error=0.0;output_saturated=0;output_values=0
whole_hand_probe=None
if a.whole_hand_path:
    from rubik_lab.diagnostics.whole_hand_path import WholeHandPath,WholeHandProbe
    whole_hand_probe=WholeHandProbe(WholeHandPath.from_result(a.whole_hand_path,a.path_trial,a.path_env,stop_waypoint=a.path_end),a.path_trial,a.path_env,preserve_servo_bias=a.path_servo_bias)
if a.live_whole_hand:
    from rubik_lab.diagnostics.live_whole_hand import LiveWholeHandProbe
    history=dict(trace=traces,sensory=sensory,contact=contact,poses=poses,motor_targets=motor_target_log,wrist_targets=wrist_target_log,ready_targets=ready_target_log,core_angular_speed=core_speed_log)
    whole_hand_probe=LiveWholeHandProbe(a.path_trial,a.path_env,out,history,a.live_prefix_degrees,a.live_support_contact,a.live_support_reserve_m,a.live_contact_face,a.live_hand_pairs_from,a.live_released_start,a.live_contact_transport,a.live_transport_reserve_m,a.live_index_feedback,a.live_contact_lateral_max_m,a.live_ready_return)
try:
    for trial in range(a.rounds):
        u.whole_hand_probe_target=None;u.post_return_hold_target=None;escape_original=None;relax_original=None;relax_reference=None;left_return=None;left_return_elapsed=0.;left_return_release_frames=0;left_return_finished_event=False;preload_brake_step=None;residual_escape=None;live_left=None;fourth_recontact=None;u.fourth_recontact_target=None;coordinated_contact=None;u.coordinated_wrist_target=None
        withdrawal=None
        if a.reversal_withdrawal or a.hold_then_withdraw:
            from rubik_lab.diagnostics.withdrawal import RadialWithdrawal
            withdrawal=RadialWithdrawal(u,proximal_first=not a.withdrawal_concurrent_return)
            u.diagnostic_withdrawal_stage=withdrawal.stage
            u.diagnostic_trial=trial
            u.diagnostic_captured=torch.zeros(u.num_envs,dtype=torch.bool,device=u.device)
        release_latched=torch.zeros(u.num_envs,dtype=torch.bool,device=u.device)
        hold_latched=torch.zeros_like(release_latched)
        hold_stable=torch.zeros(u.num_envs,dtype=torch.long,device=u.device)
        held_command=torch.zeros(u.num_envs,4,device=u.device)
        previous_command=held_command.clone()
        u.diagnostic_support_adjust.zero_()
        u.residual_recontact=None
        u.forced_move_id=first.clone();obs,_=u.reset(seed=a.seed+trial)
        exact_ready=torch.stack([u.finger_grasp[s] for s in ('left','right')],1)+u._idle_offsets()
        exact_ready=exact_ready.clamp(min=torch.stack([u.finger_lower[s] for s in ('left','right')],1),max=torch.stack([u.finger_upper[s] for s in ('left','right')],1))
        ready_target_log.append(exact_ready.cpu().numpy())
        if a.residual_recontact:
            from rubik_lab.diagnostics.recontact import ResidualRecontact
            u.residual_recontact=ResidualRecontact(u,contact_press=a.recontact_press,dense_clearance=a.recontact_dense_clearance)
        bank=torch.zeros(u.num_envs,int(primitive_seconds/u.step_dt)+2,4,device=u.device)
        pad_bank=torch.zeros(*bank.shape[:2],3,device=u.device)
        normal_bank=torch.zeros_like(pad_bank)
        progress_bank=torch.zeros(*bank.shape[:2],device=u.device)
        replay_cursor=torch.full((u.num_envs,),4,dtype=torch.long,device=u.device)
        bank_length=torch.zeros(u.num_envs,dtype=torch.long,device=u.device)
        rows=torch.arange(u.num_envs,device=u.device)
        same_pair=(group==0)|(group==3)
        for actor in actors:actor.reset()
        active=torch.ones(u.num_envs,dtype=torch.bool,device=u.device);stage=torch.zeros(u.num_envs,dtype=torch.long,device=u.device)
        u.diagnostic_live=active
        wait=torch.full_like(stage,4);first_times=torch.full((u.num_envs,),float('nan'),device=u.device)
        for step in range(u.max_episode_length+1):
            old_stage=stage.clone();template_checks=[];output_checks=[]
            with torch.no_grad():
                aa=[actor(obs) for actor in actors]
                command=torch.where((u.move_id==MOVE_INDEX['U'])[:,None],aa[0],aa[1])
                if a.second_checkpoint_u:
                    second_u=(stage==1)&(u.move_id==MOVE_INDEX['U'])
                    command=torch.where(second_u[:,None],aa[2],command)
                ready=u._idle_offsets()[u.flick_cols[u.move_id]]/cfg.residual_finger_range
                if a.second_controller in ("replay","cartesian-replay"):
                    phase=(u.cycle_step.float()/a.replay_time_scale+4).clamp(max=bank.shape[1]-1)
                    at=phase.long()
                    if a.progress_gate:
                        direction=torch.sign(u.move_target[u.move_id])
                        actual_progress=(u.move_target[u.move_id]-getattr(u,"signed_err",u.move_target[u.move_id]))*direction
                        next_at=(replay_cursor+1).clamp(max=bank.shape[1]-1)
                        can_advance=actual_progress>=progress_bank[rows,next_at]-torch.deg2rad(torch.tensor(5.,device=u.device))
                        replay_cursor+=((stage==1)&same_pair&active&can_advance).long()
                        replay_cursor.clamp_(max=bank.shape[1]-1)
                        at=replay_cursor
                    replay=torch.where((at<bank_length)[:,None],bank[rows,at],ready)
                    command=torch.where(((stage==1)&same_pair)[:,None],replay,command)
                    if planner is not None:
                        for move,side in (("U","right"),("U'","left")):
                            ids=(active&(stage==1)&same_pair&(at<bank_length)&(u.move_id==MOVE_INDEX[move])).nonzero().flatten()
                            if not len(ids):continue
                            wp,wr,q,_=to_cube_frame(u,side,mode="actual")
                            guess=u.finger_tgt[side][ids].clone()
                            target=pad_bank[ids,at[ids]]
                            normal=normal_bank[ids,at[ids]]
                            if a.replay_time_scale!=1:
                                next_at=torch.minimum(at[ids]+1,bank_length[ids]-1)
                                weight=(phase[ids]-at[ids])[:,None]
                                target=target+(pad_bank[ids,next_at]-target)*weight
                                normal=normal+(normal_bank[ids,next_at]-normal)*weight
                                normal=normal/normal.norm(dim=-1,keepdim=True).clamp(min=1e-8)
                            with torch.enable_grad():
                                corrected=planner.track(move,wp[ids],wr[ids],q[ids],guess,target,iters=a.tracking_iters,pad_normal=normal if a.replay_pad_normal else None)
                            if a.record_tracking:
                                tracking.append(torch.cat([wp[ids],wr[ids].flatten(1),q[ids],guess,target,corrected],-1).detach().cpu().numpy())
                                tracking_index.append(torch.stack([torch.full_like(ids,trial),torch.full_like(ids,step),ids,torch.full_like(ids,0 if side=='right' else 1),at[ids]],-1).cpu().numpy())
                            cols=u.flick_cols[u.move_id[ids]]
                            desired_action=((corrected-u.finger_grasp[side][ids])/cfg.residual_finger_range).gather(1,cols)
                            command[ids]=desired_action.clamp(-1,1)
                            if a.compensate_output_ema:
                                h=0 if side=='left' else 1
                                previous=u.action_smooth[ids,h*u.per_hand+6:(h+1)*u.per_hand].gather(1,cols)
                                command[ids]=command_for_filtered_target(desired_action,previous,cfg.action_ema)
                                applied=cfg.action_ema*command[ids]+(1-cfg.action_ema)*previous
                                expected=u.finger_grasp[side][ids].gather(1,cols)+applied*cfg.residual_finger_range
                                expected=expected.clamp(min=u.finger_lower[side][ids].gather(1,cols),max=u.finger_upper[side][ids].gather(1,cols))
                                output_saturated+=int(((applied-desired_action).abs()>1e-6).sum());output_values+=applied.numel()
                                output_checks.append((ids.clone(),h,cols.clone(),expected.clone()))
                withdrawal_override=torch.zeros_like(active)
                if withdrawal is not None and not a.hold_then_withdraw:
                    eligible=active&(stage==1)&(group==0)
                    release_latched|=eligible&(getattr(u,'signed_err',torch.full((u.num_envs,),torch.pi,device=u.device)).abs()<torch.deg2rad(torch.tensor(15.,device=u.device)))&(u.cube.data.joint_vel[:,u.twist_id]>1.)
                    withdrawn=withdrawal.apply(command,ready,eligible&~release_latched,planner)
                    withdrawal_override=eligible&(withdrawal.stage>0)
                    command=torch.where(withdrawal_override[:,None],withdrawn,command)
                    # Recompute expected applied targets after the explicit withdrawal override.
                    for j,(ids,h,cols,expected) in enumerate(output_checks):
                        side='left' if h==0 else 'right'
                        previous=u.action_smooth[ids,h*u.per_hand+6:(h+1)*u.per_hand].gather(1,cols)
                        applied=cfg.action_ema*command[ids]+(1-cfg.action_ema)*previous
                        expected=u.finger_grasp[side][ids].gather(1,cols)+applied*cfg.residual_finger_range
                        expected=expected.clamp(min=u.finger_lower[side][ids].gather(1,cols),max=u.finger_upper[side][ids].gather(1,cols))
                        output_checks[j]=(ids,h,cols,expected)
                if a.terminal_command_hold:
                    delta=u.move_target[u.move_id]-u._cube_state()[2]
                    err=torch.atan2(delta.sin(),delta.cos())
                    omega=u.cube.data.joint_vel[:,u.twist_id]
                    begin=active&(stage==1)&(u.cycle_step>0)&(group==0)&~hold_latched&(err.abs()<torch.deg2rad(torch.tensor(15.,device=u.device)))&(omega.abs()<1.)
                    held_command[begin]=previous_command[begin]
                    hold_latched|=begin
                    command=torch.where((hold_latched&active&(stage==1))[:,None],held_command,command)
                    output_checks=[]  # Cartesian candidate was intentionally replaced by held raw command.
                    if a.hold_then_withdraw:
                        eligible=active&(stage==1)&(group==0)
                        stable=eligible&hold_latched&(err.abs()<release_angle_tolerance)&(omega.abs()<.25)
                        hold_stable=torch.where(stable,hold_stable+1,torch.zeros_like(hold_stable))
                        release_latched|=hold_stable>=4
                        if u.residual_recontact is not None:release_latched|=u.residual_recontact.request_original_withdrawal
                        withdrawn=withdrawal.apply(command,ready,eligible&~release_latched,planner)
                        command=torch.where((eligible&(withdrawal.stage>0))[:,None],withdrawn,command)
                if u.residual_recontact is not None:
                    recontact_eligible=active&(stage==1)&(group==0)&hold_latched
                    if whole_hand_probe is not None and whole_hand_probe.started and trial==whole_hand_probe.trial and int(stage[whole_hand_probe.env])==1:recontact_eligible[whole_hand_probe.env]=False
                    u.residual_recontact.apply(recontact_eligible,err,omega,trial,step)
                command=apply_ready_hold(command,ready,wait>0,actors)
                tail_ids=(active&(stage>=2)&(wait==0)).nonzero().flatten()
                if len(tail_ids):
                    expected=torch.where((u.move_id==MOVE_INDEX['U'])[:,None],aa[0],aa[1])
                    if not torch.equal(command[tail_ids],expected[tail_ids]):raise RuntimeError('Tail actor command was overridden')
                if bool((active&(stage==3)).any()) and (u.post_return_hold_target is not None or u.whole_hand_probe_target is not None):raise RuntimeError('Completed diagnostic target leaked into fourth action')
                previous_command=command.clone()
                first_ids=(active&(stage==0)).nonzero().flatten()
                age=u.cycle_step[first_ids].clamp(max=bank.shape[1]-1)
                bank[first_ids,age]=command[first_ids];bank_length[first_ids]=age+1
                progress_bank[first_ids,age]=(u.move_target[u.move_id[first_ids]]-getattr(u,"signed_err",u.move_target[u.move_id])[first_ids])*torch.sign(u.move_target[u.move_id[first_ids]])
                if planner is not None:
                    for move,side in (("U","right"),("U'","left")):
                        ids=(active&(stage==0)&(u.move_id==MOVE_INDEX[move])).nonzero().flatten()
                        if not len(ids):continue
                        wp,wr,q,_=to_cube_frame(u,side,mode="actual")
                        desired=q[ids].clone();cols=u.flick_cols[u.move_id[ids]]
                        template_action=command[ids]
                        if a.template_target=='filtered':
                            h=0 if side=='left' else 1
                            previous=u.action_smooth[ids,h*u.per_hand+6:(h+1)*u.per_hand].gather(1,cols)
                            template_action=cfg.action_ema*template_action.clamp(-cfg.clip_actions,cfg.clip_actions)+(1-cfg.action_ema)*previous
                        requested=u.finger_grasp[side][ids].gather(1,cols)+template_action*cfg.residual_finger_range
                        if a.template_target=='filtered':
                            requested=requested.clamp(min=u.finger_lower[side][ids].gather(1,cols),max=u.finger_upper[side][ids].gather(1,cols))
                            template_checks.append((ids.clone(),h,cols.clone(),requested.clone()))
                        desired.scatter_(1,cols,requested)
                        pad,normal=planner.fk[side].pads(planner.fk[side].forward(wp[ids],wr[ids],desired))["index"]
                        pad_bank[ids,u.cycle_step[ids].clamp(max=bank.shape[1]-1)]=pad
                        normal_bank[ids,u.cycle_step[ids].clamp(max=bank.shape[1]-1)]=normal
                u.diagnostic_support_adjust[:]=a.support_delta*(stage==1).float()*((u.cycle_step+1)/4).clamp(max=1)
                wait=(wait-1).clamp(min=0)
                if whole_hand_probe is not None:whole_hand_probe.prepare(u,trial,step,active&(stage==1),stage)
                if a.post_return_hold_after_steps is not None and trial==a.path_trial and bool(active[a.path_env]) and int(stage[a.path_env])==2 and u.post_return_hold_target is None and int(u.cycle_step[a.path_env])>=a.post_return_hold_after_steps:
                    i=a.path_env;side='left';cols=u.flick_cols[u.move_id[i]].clone();values=u.finger_tgt[side][i,cols].clone()
                    u.post_return_hold_target=(i,side,cols,values)
                    post_return_events.append(dict(event='post_return_motor_hold',trial=trial,step=step,env=i,side=side,joint_indices=cols.tolist(),motor_target=values.tolist(),after_steps=a.post_return_hold_after_steps,scope='Diagnostic motor-target intervention only; no state writes or completion credit'))
                if a.diagnostic_preload_at_error_deg is not None and trial==a.path_trial and bool(active[a.path_env]) and int(stage[a.path_env])==2:
                    i=a.path_env;side='left';cols=u.flick_cols[u.move_id[i]].clone();s0=u.snapshot
                    if preload_brake_step is None and preload_brake_trigger(float(s0['twist_error'][i]),float(s0['twist_velocity'][i]),a.diagnostic_preload_at_error_deg,int(u.cycle_step[i])):
                        preload_brake_step=step;relax_original=u.finger_tgt[side][i,cols].clone();relax_reference=s0['finger_pos'][i,0,cols].clone()
                        post_return_events.append(dict(event='post_return_motor_hold',trial=trial,step=step,env=i,scope='Truth-triggered preload removal, not fixed hold; no completion credit'))
                        post_return_events.append(dict(event='post_return_relax_start',trial=trial,step=step,env=i,source_motor_target=relax_original.tolist(),physical_reference=relax_reference.tolist(),source_error_deg=float(s0['twist_error'][i])*180/np.pi,source_layer_speed=float(s0['twist_velocity'][i]),threshold_deg=a.diagnostic_preload_at_error_deg,scope='Privileged trigger with latched proprio reference;1rad/s motor ramp, no state writes'))
                    if preload_brake_step is not None:
                        values=relax_motor_target(relax_original,relax_reference,(step-preload_brake_step+1)*u.step_dt,u.finger_lower[side][i,cols],u.finger_upper[side][i,cols])
                        u.post_return_hold_target=(i,side,cols,values)
                if a.diagnostic_preload_escape_path is not None and trial==a.path_trial and bool(active[a.path_env]) and int(stage[a.path_env])==2:
                    from rubik_lab.diagnostics.left_return import LeftResidualEscapePath
                    if residual_escape is None:
                        residual_escape=LeftResidualEscapePath(a.diagnostic_preload_escape_path)
                        if (residual_escape.source['trial'],residual_escape.source['env'])!=(trial,a.path_env):raise ValueError('Residual escape cohort mismatch')
                    if step>residual_escape.source['source_step']:
                        if not residual_escape.started:
                            residual_escape.verify_history(history)
                            post_return_events.append(dict(event='preload_residual_escape_start',trial=trial,step=step,env=a.path_env,source=str(a.diagnostic_preload_escape_path),scope='Exact history, audited small motor increment; no state writes or completion credit'))
                        i,side,cols,_=u.post_return_hold_target
                        if cols.tolist()!=[5,6,7,8]:raise RuntimeError('Unexpected residual escape mapping')
                        values=torch.as_tensor(residual_escape.sample((step-residual_escape.source['source_step'])*u.step_dt),device=u.device,dtype=u.finger_tgt[side].dtype)
                        u.post_return_hold_target=(i,side,cols,values)
                if a.post_return_relax_after_steps is not None and trial==a.path_trial and bool(active[a.path_env]) and int(stage[a.path_env])==2 and int(u.cycle_step[a.path_env])>=a.post_return_relax_after_steps:
                    i,side,cols,values=u.post_return_hold_target
                    if cols.tolist()!=[5,6,7,8]:raise RuntimeError('Unexpected left index mapping')
                    if relax_original is None:
                        relax_original=values.clone();relax_reference=u.snapshot['finger_pos'][i,0,cols].clone()
                        post_return_events.append(dict(event='post_return_relax_start',trial=trial,step=step,env=i,source_motor_target=values.tolist(),physical_reference=relax_reference.tolist(),scope='Latched proprio reference, ramp motor targets at1rad/s; no state writes or completion credit'))
                    elapsed=(int(u.cycle_step[i])-a.post_return_relax_after_steps+1)*u.step_dt
                    values=relax_motor_target(relax_original,relax_reference,elapsed,u.finger_lower[side][i,cols],u.finger_upper[side][i,cols])
                    u.post_return_hold_target=(i,side,cols,values)
                if a.post_return_escape_after_steps is not None and trial==a.path_trial and bool(active[a.path_env]) and int(stage[a.path_env])==2 and int(u.cycle_step[a.path_env])>=a.post_return_escape_after_steps:
                    i,side,cols,values=u.post_return_hold_target
                    if cols.tolist()!=[5,6,7,8]:raise RuntimeError('Unexpected left index mapping')
                    if escape_original is None:
                        escape_original=values.clone()
                        post_return_events.append(dict(event='post_return_escape_start',trial=trial,step=step,env=i,source_motor_target=values.tolist(),scope='First segment only, motor joint6 -.4rad at1rad/s; no state writes or completion credit'))
                    elapsed=(int(u.cycle_step[i])-a.post_return_escape_after_steps+1)*u.step_dt
                    values=escape_motor_target(escape_original,elapsed,u.finger_lower[side][i,cols],u.finger_upper[side][i,cols])
                    u.post_return_hold_target=(i,side,cols,values)
                if a.post_return_left_path is not None and trial==a.path_trial and bool(active[a.path_env]) and int(stage[a.path_env])==2:
                    from rubik_lab.diagnostics.left_return import LeftReturnPath
                    if left_return is None:
                        left_return=LeftReturnPath(a.post_return_left_path)
                        if (left_return.source['trial'],left_return.source['env'])!=(trial,a.path_env):raise ValueError('Left return cohort mismatch')
                    if step>left_return.source['source_step']:
                        if not left_return.started:
                            left_return.verify_history(history)
                            post_return_events.append(dict(event='left_return_start',trial=trial,step=step,env=a.path_env,source=str(a.post_return_left_path),duration_seconds=left_return.duration,scope='Exact prefix verified, privileged pre-audited path; no state/clock/reference writes'))
                        left_return_elapsed=(step-left_return.source['source_step'])*u.step_dt
                        i,side,cols,_=u.post_return_hold_target
                        values=torch.as_tensor(left_return.sample(left_return_elapsed),device=u.device,dtype=u.finger_tgt[side].dtype)
                        u.post_return_hold_target=(i,side,cols,values)
                        if left_return_elapsed>=left_return.duration and not left_return_finished_event:
                            left_return_finished_event=True
                            post_return_events.append(dict(event='left_return_targets_finished',trial=trial,step=step,env=i))
                if a.live_left_withdrawal and trial==a.path_trial and bool(active[a.path_env]) and int(stage[a.path_env])==2 and preload_brake_step is not None:
                    from rubik_lab.diagnostics.left_return import LiveLeftWithdrawal
                    if live_left is None:live_left=LiveLeftWithdrawal(whole_hand_probe)
                    left_return,left_return_elapsed=live_left.prepare(u,trial,step,preload_brake_step,float((relax_reference-relax_original).abs().max()),post_return_events)
                if a.fourth_recontact and trial==a.path_trial and bool(active[a.path_env]) and int(stage[a.path_env])==3:
                    from rubik_lab.diagnostics.left_recontact import LiveFourthRecontact
                    if fourth_recontact is None:fourth_recontact=LiveFourthRecontact(whole_hand_probe,a.fourth_recontact_press_m,a.fourth_recontact_center_m,a.contact_turn,a.fourth_recontact_surface_depth_m,a.contact_turn_path,a.stroke_lift_path,a.stroke_recontact_path,a.live_stroke_cycle,a.contact_turn_depth_reserve_m,a.second_stroke,a.second_stroke_feedback,a.second_stroke_degrees,a.second_stroke_feedback_gain,a.recycle_completed_stroke,a.completed_lift_feedback,a.completed_recontact_outside_m,a.completed_recontact_feedback,a.contact_entry_delay_frames)
                    fourth_recontact.prepare(u,trial,step,post_return_events)
                    if a.coordinated_contact_path:
                        from rubik_lab.diagnostics.coordinated_contact import CoordinatedContactPath
                        if coordinated_contact is None:coordinated_contact=CoordinatedContactPath(a.coordinated_contact_path,bind_actual=a.coordinated_contact_bind_actual)
                        coordinated_contact.prepare(u,whole_hand_probe,fourth_recontact.release_cycle,trial,step,post_return_events)
                if wrench_recorder:u.diagnostic_wrench_active=bool(whole_hand_probe and whole_hand_probe.started and trial==whole_hand_probe.trial)
                obs,_,terminated,truncated,_=u.step(command)
            s=u.snapshot
            core_speed_log.append(s['core_angular_speed'].cpu().numpy())
            if whole_hand_probe is not None:
                motor_target_log.append(s['finger_tgt'].cpu().numpy())
                wrist_target_log.append(s['ee_target_pose'].cpu().numpy())
            for ids,h,cols,expected in output_checks:
                if u.whole_hand_probe_target is not None:
                    keep=ids!=whole_hand_probe.env;ids,h,cols,expected=ids[keep],h,cols[keep],expected[keep]
                    if not len(ids):continue
                error=float((s['finger_tgt'][ids,h].gather(1,cols)-expected).abs().max())
                output_target_error=max(output_target_error,error)
                if error>1e-6:raise RuntimeError(f'Output EMA prediction mismatch: {error}')
            for ids,h,cols,requested in template_checks:
                if u.whole_hand_probe_target is not None:
                    keep=ids!=whole_hand_probe.env;ids,h,cols,requested=ids[keep],h,cols[keep],requested[keep]
                    if not len(ids):continue
                error=float((s['finger_tgt'][ids,h].gather(1,cols)-requested).abs().max())
                template_target_error=max(template_target_error,error)
                if error>1e-6:raise RuntimeError(f'Filtered template disagrees with applied target: {error}')
            invalid=s['cycle_failed'].bool()|s['dropped'].bool()|s['broken'].bool()|terminated.bool()
            full=s['primitive_done'].bool()&~invalid
            if whole_hand_probe is not None and whole_hand_probe.started and trial==whole_hand_probe.trial:
                if not getattr(whole_hand_probe,'permits_completion',False):
                    full[whole_hand_probe.env]=False  # Only an audited, finished return may expose original full physical monitor.
            if u.post_return_hold_target is not None:
                i,side,cols,values=u.post_return_hold_target
                if not torch.equal(s['finger_tgt'][i,0,cols],values):raise RuntimeError('Post-return motor hold target not delivered')
                if left_return is not None and left_return.started and left_return_elapsed>=left_return.duration:
                    total=float(s['link_contact_force'][i,:,0,6:11].sum())
                    left_return_release_frames=left_return_release_frames+1 if total<=.5 and bool(s['cycle_ready'][i]) and not bool(invalid[i]) else 0
                    full[i]=full[i] and left_return_release_frames>=10
                else:full[i]=False
            if u.fourth_recontact_target is not None:
                i,side,cols,values=u.fourth_recontact_target
                if not torch.equal(s['finger_tgt'][i,0,cols],values):raise RuntimeError('Fourth recontact target not delivered')
                full[i]=False  # Contact component never credits a completed quarter turn.
            if u.coordinated_wrist_target is not None:
                i,side,position,quat=u.coordinated_wrist_target
                if not torch.equal(s['ee_target_pose'][i,0,:3],position) or not torch.equal(s['ee_target_pose'][i,0,3:7],quat):raise RuntimeError('Coordinated wrist target not delivered')
            timeout=(s['cycle_step']*u.step_dt>=primitive_seconds-1e-6)&~full
            append_tail=torch.zeros_like(active)
            if a.post_return_move and trial==a.path_trial:
                append_tail[a.path_env]=True
            good_second=active&full&(stage==1)&append_tail&~truncated.bool()
            append_fourth=append_tail&bool(a.after_left_return_move)
            good_third=active&full&(stage==2)&append_fourth&~truncated.bool()
            final_stage=torch.where(append_fourth,3,torch.where(append_tail,2,1))
            ended=active&(invalid|timeout|truncated.bool()|(full&(stage==final_stage)))
            good_first=active&full&(stage==0)&~ended
            first_times[good_first]=s['cycle_step'][good_first]*u.step_dt
            traces.append(torch.stack([torch.full_like(stage,trial),torch.full_like(stage,step),group,stage,active,torch.rad2deg(s['twist_error']),s['twist_velocity'],s['finger_ready_error'],s['release_force'],s['cycle_ready'],s['primitive_done'],invalid,s['peak_cube_pos'],s['peak_cube_rot'],s['peak_wrist_pos'],s['peak_wrist_rot']],-1).cpu().numpy())
            sensory.append(torch.cat([s['finger_pos'].flatten(1),s['finger_velocity'].flatten(1),s['tactile_ft'].flatten(1)],-1).cpu().numpy())
            contact.append(s['link_contact_force'].cpu().numpy())
            if wrench_recorder and 'probe_wrench' in s:wrench_recorder.append(trial,step,s['probe_wrench'])
            poses.append(torch.cat([s['cube_pos'],s['cube_quat'],s['ee_pose'].flatten(1)],-1).cpu().numpy())
            actions_log.append(command.cpu().numpy())
            if withdrawal is not None:withdrawal_log.append(withdrawal.stage.cpu().numpy().copy())
            if a.terminal_command_hold:terminal_hold_log.append(hold_latched.cpu().numpy().copy())
            if u.residual_recontact is not None:
                c=u.residual_recontact
                recontact_trace.append(torch.cat([c.phase[:,None],c.cursor[:,None],c.used[:,None],c.command],-1).cpu().numpy())
            for i in ended.nonzero().flatten().tolist():
                reason='motion_or_drop' if bool(invalid[i]) else ('success' if bool(full[i]&(stage[i]==final_stage[i])) else 'timeout')
                records.append({'trial':trial,'env':i,'pair':list(pairs[int(group[i])]),'sequence':list(pairs[int(group[i])])+([a.post_return_move] if bool(append_tail[i]) else [])+([a.after_left_return_move] if bool(append_fourth[i]) else []),'reason':reason,'completed_stages':int(stage[i])+int(reason=='success'),'first_seconds':float(first_times[i]) if torch.isfinite(first_times[i]) else None,'final_error_deg':float(torch.rad2deg(s['twist_error'][i]))})
            active &= ~ended
            for g,pair in enumerate(pairs):
                ids=(good_first&(group==g)).nonzero().flatten()
                if len(ids):
                    if a.audit_handoff:
                        from rubik_lab.continuous_prefix import advance_preserving_state
                        handoff_audits.append({'trial':trial,'step':step,'pair':list(pair),**advance_preserving_state(u,ids,pair[1])})
                    else:u.advance_cycle(ids,pair[1])
                    stage[ids]=1;wait[ids]=a.handoff_wait_steps
                    for actor in actors:actor.reset(ids)
            if good_second.any():
                ids=good_second.nonzero().flatten()
                from rubik_lab.continuous_prefix import advance_preserving_state
                audit=advance_preserving_state(u,ids,a.post_return_move)
                post_return_events.append(dict(event='post_return_handoff',trial=trial,step=step,envs=ids.tolist(),move=a.post_return_move,controller='frozen_blind_actor',**audit))
                stage[ids]=2;wait[ids]=a.handoff_wait_steps
                hold_latched[ids]=False;release_latched[ids]=False;hold_stable[ids]=0
                # Completed diagnostic overlays must not continue commanding the next primitive.
                if u.residual_recontact is not None:u.residual_recontact.used[ids]=False
                for actor in actors:actor.reset(ids)
            if good_third.any():
                ids=good_third.nonzero().flatten()
                from rubik_lab.continuous_prefix import advance_preserving_state
                audit=advance_preserving_state(u,ids,a.after_left_return_move)
                u.post_return_hold_target=None;u.whole_hand_probe_target=None
                hold_latched[ids]=False;release_latched[ids]=False;hold_stable[ids]=0
                if u.residual_recontact is not None:u.residual_recontact.used[ids]=False
                stage[ids]=3;wait[ids]=a.handoff_wait_steps
                for actor in actors:actor.reset(ids)
                post_return_events.append(dict(event='after_left_return_handoff',trial=trial,step=step,envs=ids.tolist(),move=a.after_left_return_move,controller=('privileged_recontact_component' if a.fourth_recontact else 'frozen_blind_actor'),diagnostic_targets_cleared=True,**audit))
            if good_first.any() or good_second.any() or good_third.any():obs=u._get_observations()
            if not active.any():break
        if active.any():raise RuntimeError('Unfinished trial quota')
        if u.residual_recontact is not None:recontact_events.extend(u.residual_recontact.events)
        print(f'[continuation] round{trial+1}/{a.rounds}',flush=True)
    summary={}
    for pair in pairs:
        rows=[r for r in records if r['pair']==list(pair)]
        summary['->'.join(pair)]={'n':len(rows),'first_complete':sum(r['completed_stages']>=1 for r in rows),'two_complete':sum(r['completed_stages']>=2 for r in rows),'three_complete':sum(r['completed_stages']>=3 for r in rows),'four_complete':sum(r['completed_stages']>=4 for r in rows)}
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out.with_suffix('.npz'),**(wrench_recorder.arrays() if wrench_recorder else {}),**({'whole_hand_return_targets':np.asarray(getattr(whole_hand_probe,'return_targets',[]),dtype=float).reshape(-1,33),'whole_hand_feedback':np.asarray(whole_hand_probe.feedback,dtype=float).reshape(-1,23),'whole_hand_targets':np.asarray(whole_hand_probe.targets,dtype=float).reshape(-1,33),'motor_targets':np.asarray(motor_target_log),'wrist_targets':np.asarray(wrist_target_log),'actuator_lower':torch.stack([u.finger_lower[s] for s in ('left','right')],1).cpu().numpy(),'actuator_upper':torch.stack([u.finger_upper[s] for s in ('left','right')],1).cpu().numpy()} if whole_hand_probe else {}),core_angular_speed=np.asarray(core_speed_log),ready_targets=np.asarray(ready_target_log),**({'recontact':np.asarray(recontact_trace)} if recontact_trace else {}),**{k:np.concatenate(v) for k,v in u.release_states.items()},**({'release_substeps':np.concatenate(u.release_substeps)} if u.release_substeps else {}),**({'terminal_hold':np.asarray(terminal_hold_log)} if terminal_hold_log else {}),**({'withdrawal_stage':np.asarray(withdrawal_log)} if withdrawal_log else {}),**({'tracking':np.concatenate(tracking),'tracking_index':np.concatenate(tracking_index)} if tracking else {}),trace=np.asarray(traces),sensory=np.asarray(sensory),contact=np.asarray(contact),poses=np.asarray(poses),commands=np.asarray(actions_log),env_origins=u.scene.env_origins.cpu().numpy())
    result={'coordinated_contact_bind_actual':a.coordinated_contact_bind_actual,'coordinated_contact_path':str(a.coordinated_contact_path) if a.coordinated_contact_path else None,'contact_entry_delay_frames':a.contact_entry_delay_frames,'completed_recontact_feedback':a.completed_recontact_feedback,'completed_recontact_outside_m':a.completed_recontact_outside_m,'completed_lift_feedback':a.completed_lift_feedback,'recycle_completed_stroke':a.recycle_completed_stroke,'second_stroke_feedback_gain':a.second_stroke_feedback_gain,'second_stroke_degrees':a.second_stroke_degrees,'second_stroke_feedback':a.second_stroke_feedback,'second_stroke':a.second_stroke,'contact_turn_depth_reserve_m':a.contact_turn_depth_reserve_m,'live_stroke_cycle':a.live_stroke_cycle,'stroke_recontact_path':str(a.stroke_recontact_path) if a.stroke_recontact_path else None,'stroke_lift_path':str(a.stroke_lift_path) if a.stroke_lift_path else None,'contact_turn_path':str(a.contact_turn_path) if a.contact_turn_path else None,'fourth_recontact_surface_depth_m':a.fourth_recontact_surface_depth_m,'contact_turn':a.contact_turn,'fourth_recontact_center_m':a.fourth_recontact_center_m,'fourth_recontact_press_m':a.fourth_recontact_press_m,'fourth_recontact':a.fourth_recontact,'live_left_withdrawal':a.live_left_withdrawal,'diagnostic_preload_escape_path':str(a.diagnostic_preload_escape_path) if a.diagnostic_preload_escape_path else None,'diagnostic_preload_at_error_deg':a.diagnostic_preload_at_error_deg,'after_left_return_move':a.after_left_return_move,'post_return_left_path':str(a.post_return_left_path) if a.post_return_left_path else None,'post_return_relax_after_steps':a.post_return_relax_after_steps,'post_return_escape_after_steps':a.post_return_escape_after_steps,'post_return_hold_after_steps':a.post_return_hold_after_steps,'post_return_move':a.post_return_move,'post_return_events':post_return_events,'post_return_controller':('Privileged fourth recontact component; no quarter-turn credit' if a.fourth_recontact else 'Frozen original sensory actor; no second-stage replay or whole-hand overrides; actor recurrent state reset at handoff, physical state/sensorhistory/references retained'),'probe_wrench_start':a.probe_wrench_start,'probe_wrench_schema':wrench_recorder.schema() if wrench_recorder else None,'actuator_record_scope':'When wholehandprobe enabled: motor_targets perframe/env/hand/joint, wrist_targets envrelative xyz+wxyz, actuator_lower/upper env/hand/joint. Diagnostic output only; not actor inputs.','whole_hand_probe':dict(source=str(getattr(whole_hand_probe,'input_prefix',a.whole_hand_path)),live_planning=a.live_whole_hand,ready_return=a.live_ready_return,permits_physical_completion=getattr(whole_hand_probe,'permits_completion',False),contact_lateral_max_m=a.live_contact_lateral_max_m,index_feedback=a.live_index_feedback,feedback_columns='trial,step,env,4physical_reference,4measured_q,4requested_correction,4applied_correction,4limited_flags',feedback_rule='Hold-only after path.duration+.25s, indexDOFs5:9, gain1, .1rad cap,1rad/s command rate, originalactuatorlimits; nointegral orstatewrites',contact_transport=a.live_contact_transport,transport_optimizer_reserve_m=a.live_transport_reserve_m,released_start=a.live_released_start,hand_pairs_from=str(a.live_hand_pairs_from) if a.live_hand_pairs_from else None,contact_face=a.live_contact_face,support_contact=a.live_support_contact,support_optimizer_reserve_m=a.live_support_reserve_m,stop_waypoint=getattr(whole_hand_probe,'requested_end',a.path_end),events=whole_hand_probe.events,target_columns='trial,step,env,elapsed,22finger targets,3wrist position envrelative,4wrist quaternion wxyz',scope=('Privileged correction plus independentlyauditedreturn; original fullphysicalmonitor scorescompletion, not blindacceptance' if a.live_ready_return else 'Privileged prefix-only probe; originalclock/reference/physicsmonitors retained; no actioncompletion credit for intervenedenv')) if whole_hand_probe else None,'recontact_press_m':a.recontact_press,'recontact_dense_clearance':a.recontact_dense_clearance,'residual_recontact':a.residual_recontact,'recontact_events':recontact_events,'ready_targets_axes':'round,env,hand,joint; exact clamped ready reference, evaluation only','core_angular_speed_scope':'Per control-frame pre-reset snapshot, evaluation only; does not resolve all physics substeps','recontact_columns':['phase','cursor','supplemental_finger_participated','left_command0','left_command1','left_command2','left_command3'],'scope':'Development diagnostic; tactile/proprio actors, privileged full-ready handoff; cartesian-replay also uses current true cube pose for IK. Not acceptance. No physical resets, alignment, or refreshed pose references within a live sequence.','forces_each_iteration':a.forces_each_iteration,'friction_law':a.friction_law,'physics_hz':a.physics_hz,'control_hz':40,'tactile_latency_s':legacy_latency,'seed':a.seed,'rounds':a.rounds,'num_envs':a.num_envs,'actor_hidden_reset_during_wait':True,'initial_wait_steps':4,'handoff_wait_steps':a.handoff_wait_steps,'progress_gate':a.progress_gate,'replay_pad_normal':a.replay_pad_normal,'record_tracking':a.record_tracking,'tracking_iters':a.tracking_iters,'handoff_audits':handoff_audits,'release_substep_columns':['trial','physics_step','env','joint_angle_rad','joint_speed_rad_s','top_total_normal_force_N','core_angular_speed_rad_s'],'terminal_state_scope':'One pre-integration state per liveUU return entry; relative rootpose, rootvelocity, alljoint position/velocity and originaltargets. No solver warmstart export; diagnostic replay only.','terminal_state_joint_names':{name:asset.joint_names for name,asset in u.scene.articulations.items()} if a.capture_release_states else None,'capture_release_states':a.capture_release_states,'record_release_substeps':a.record_release_substeps,'release_angle_tolerance_deg':float(np.rad2deg(release_angle_tolerance)),'release_at_task_tolerance':a.release_at_task_tolerance,'hold_then_withdraw':a.hold_then_withdraw,'terminal_command_hold':a.terminal_command_hold,'reversal_withdrawal':a.reversal_withdrawal,'withdrawal_concurrent_return':a.withdrawal_concurrent_return,'saturation_scope':'Replay conversion before optional withdrawal override','replay_time_scale':a.replay_time_scale,'compensate_output_ema':a.compensate_output_ema,'output_target_prediction_max_error_rad':output_target_error,'output_target_clipped_joint_frames':output_saturated,'output_target_joint_frames':output_values,'template_target':a.template_target,'template_applied_target_max_error_rad':template_target_error,'tracking_columns':{'wrist_pos_cube':[0,3],'wrist_rotation_cube':[3,12],'actual_q':[12,34],'seed_target_q':[34,56],'target_pad_cube':[56,59],'solved_q':[59,81]},'tracking_index_columns':['trial','step','env','side_right0_left1','template_step'],'second_controller':a.second_controller,'support_delta_rad':a.support_delta,'timeout_per_primitive_s':primitive_seconds,'models':paths,'second_U_model':a.second_checkpoint_u,'cache':cfg.grasp_cache_path,'trace_columns':['trial','step','pair','stage','active','angle_error_deg','layer_speed','finger_ready_error','release_force','instant_ready','full_ready','invalid','peak_cube_pos_m','peak_cube_rot_rad','peak_wrist_pos_m','peak_wrist_rot_rad'],'sensory_columns':{'q':[0,44],'qd':[44,88],'delivered_ft':[88,148]},'pose_columns':'cube xyz/quaternion then left/right wrist xyz/quaternion','contact_axes':'step,env,cube_body,hand,link','summary':summary,'episodes':records}
    out.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary),flush=True)
except BaseException as error:
    # Preserve all completed frames even when a target or implementation guard fails.
    out.parent.mkdir(parents=True,exist_ok=True)
    if not out.with_suffix('.json').exists():
        np.savez_compressed(out.with_suffix('.npz'),**(wrench_recorder.arrays() if wrench_recorder else {}),trace=np.asarray(traces),sensory=np.asarray(sensory),contact=np.asarray(contact),poses=np.asarray(poses),commands=np.asarray(actions_log),ready_targets=np.asarray(ready_target_log),core_angular_speed=np.asarray(core_speed_log),env_origins=u.scene.env_origins.cpu().numpy(),**({'whole_hand_return_targets':np.asarray(getattr(whole_hand_probe,'return_targets',[]),dtype=float).reshape(-1,33),'whole_hand_feedback':np.asarray(whole_hand_probe.feedback,dtype=float).reshape(-1,23),'whole_hand_targets':np.asarray(whole_hand_probe.targets,dtype=float).reshape(-1,33),'motor_targets':np.asarray(motor_target_log),'wrist_targets':np.asarray(wrist_target_log),'actuator_lower':torch.stack([u.finger_lower[s] for s in ('left','right')],1).cpu().numpy(),'actuator_upper':torch.stack([u.finger_upper[s] for s in ('left','right')],1).cpu().numpy()} if whole_hand_probe else {}))
        out.with_suffix('.json').write_text(json.dumps(dict(probe_wrench_schema=wrench_recorder.schema() if wrench_recorder else None,status='failed_partial_raw_saved_not_accepted',error=repr(error),completed_frames=len(traces),whole_hand_probe=dict(source=str(a.whole_hand_path),events=whole_hand_probe.events) if whole_hand_probe else None,scope='Partial completed controlframes only; requested probe targets do not prove they were applied. No success claim.'),indent=2)+'\n')
    import traceback;traceback.print_exc();raise
finally:
    u.close();app.close()
