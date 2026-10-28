"""Matched direction-sign experiment for the existing privileged flick teacher.
This is a development diagnostic, never sensor-only skill acceptance.
"""
import argparse
import copy
import json
import types
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--progress',choices=['legacy','directed'],default='directed')
p.add_argument('--retry-plan-frame',choices=['cache','actual'],default=None)
p.add_argument('--plan-frame',choices=['cache','actual'],default='cache')
p.add_argument('--num_envs',type=int,default=32)
p.add_argument('--rounds',type=int,default=2)
p.add_argument('--seed',type=int,default=1201)
p.add_argument('--cube-armature',type=float,default=None,help='Isolated physics diagnostic only; never an acceptance relaxation')
p.add_argument('--physics-hz',type=int,default=None)
p.add_argument('--external-forces-every-iteration',action='store_true')
p.add_argument('--retry-after-ready',action='store_true')
p.add_argument('--wrist-tilt-rate-deg-s',type=float,default=10.0)
p.add_argument('--wrist-tilt-gain',type=float,default=0.0)
p.add_argument('--support-tilt-actual-pose',action='store_true')
p.add_argument('--support-tilt-ring',action='store_true')
p.add_argument('--support-tilt-holding',action='store_true',help='Include thumb with middle support compensation')
p.add_argument('--support-tilt-phase',choices=['persistent','recovery'],default='persistent')
p.add_argument('--support-tilt-gain',type=float,default=0.0)
p.add_argument('--finger-control',choices=['active-only','teacher-all','teacher-avoid'],default='active-only')
p.add_argument('--stroke-limit-deg',type=float,default=None,help='Diagnostic release after this directed segment angle; total completion gate unchanged')
p.add_argument('--record-substeps',action='store_true',help='Measure first1.2s everyphysicsstep; no controller change')
p.add_argument('--record-wrenches',action='store_true')
p.add_argument('--contact-offset-mm',type=float,default=None)
p.add_argument('--contact-height-mm',type=float,default=None)
p.add_argument('--middle-support-press',type=float,default=0.0)
p.add_argument('--track-normal',action='store_true')
p.add_argument('--radial-withdrawal',action='store_true')
p.add_argument('--proximal-first-return',action='store_true')
p.add_argument('--release-rate',type=float,default=None,help='Diagnostic active joint withdrawal rate in rad/s, initialized from measured joint positions')
p.add_argument('--support-persistent',action='store_true',help='Keep diagnostic support offsets after the first push until episode end')
p.add_argument('--up-right-thumb-relief',type=float,default=0.0)
p.add_argument('--out',required=True)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args()
if a.proximal_first_return and not a.radial_withdrawal:raise ValueError('Proximal-first return requires radial withdrawal')
if a.radial_withdrawal and a.release_rate is not None:raise ValueError('Select radial withdrawal or joint-only rate limit, not both')
if a.num_envs%2:raise ValueError('Even environment count required for matched U/Up groups')
out=Path(a.out)
if out.with_suffix('.json').exists() or out.with_suffix('.npz').exists():raise FileExistsError(out)
app=AppLauncher(a).app
import numpy as np
import torch
from rubik_lab.tasks.bimanual_twist.cycle_cfg import CycleUEnvCfg
from rubik_lab.tasks.bimanual_twist.cycle_env import PrimitiveCycleEnv
from rubik_lab.primitive_config import restore_checkpoint_settings
from rubik_lab.moves import MOVE_INDEX,ALL_MOVES
from rubik_lab.constants import FINGER_JOINTS
from rubik_lab.teacher import ScriptedTeacher,F_PUSH
cfg=CycleUEnvCfg()
restore_checkpoint_settings(cfg,'logs/rsl_rl/rubik_primitive_cycle/U_blind_full/model.pt')
cfg.scene.num_envs=a.num_envs;cfg.seed=a.seed
cfg.grasp_cache_path='cache/g70s_cycle_demo1_v3.npz'
cfg.moves=('U',"U'")
cfg.cycle_terminate_on_success=False
cfg.gravity_curriculum=False
if a.cube_armature is not None:
    if a.cube_armature<0:raise ValueError("Negative armature")

if a.external_forces_every_iteration:
    cfg.sim.physx.enable_external_forces_every_iteration=True
if a.physics_hz is not None:
    control_dt=cfg.sim.dt*cfg.decimation
    substeps=round(control_dt*a.physics_hz)
    if a.physics_hz<=0 or substeps<1 or abs(substeps/a.physics_hz-control_dt)>1e-9:
        raise ValueError("Physics rate must preserve the original control period exactly")
    cfg.sim.dt=1/a.physics_hz;cfg.decimation=substeps;cfg.sim.render_interval=substeps
class MechanicalEnv(PrimitiveCycleEnv):
    def _apply_action(self):
        if a.record_substeps and getattr(self,'diagnostic_recording',False):
            tick=self.diagnostic_physics_tick
            if 0<tick*self.physics_dt<=1.2:
                from rubik_lab.contact_wrench import point_contact_wrench
                origin=self.cube.data.body_pos_w[:,self.core_id,None,:]
                ww=[]
                for body in ('core','top'):
                    view=self.cube_contacts.sensors[body].contact_physx_view
                    mag,points,normals,_,counts,starts=view.get_contact_data(dt=self.physics_dt)
                    normal=point_contact_wrench(mag.reshape(-1,1)*normals,points,counts,starts,origin).sum(-2)
                    force,points,counts,starts=view.get_friction_data(dt=self.physics_dt)
                    friction=point_contact_wrench(force,points,counts,starts,origin).sum(-2)
                    ww.append(torch.stack([normal,friction],1))
                self.diagnostic_substeps.append((self.diagnostic_trial,tick,self.diagnostic_active.cpu().numpy().copy(),self.cube.data.body_state_w.cpu().numpy().copy(),torch.stack(ww,1).cpu().numpy()))
            self.diagnostic_physics_tick+=1
        super()._apply_action()

    def _expand_flick_action(self,pol):
        full=super()._expand_flick_action(pol)
        if a.finger_control!="active-only" and hasattr(self,"teacher_finger_command"):
            for h in range(2):
                sl=slice(h*self.per_hand+6,(h+1)*self.per_hand)
                mask=self.teacher_pushing[:,None]
                if a.finger_control=="teacher-avoid":
                    avoid=torch.tensor([name.split("_")[0] not in ("thumb","middle") for name in FINGER_JOINTS],device=self.device)
                    mask=mask&avoid[None,:]
                full[:,sl]=torch.where(mask,self.teacher_finger_command[:,sl],full[:,sl])
            rows=torch.arange(self.num_envs,device=self.device)
            cols=(self.flick_hand[self.move_id]*self.per_hand+6)[:,None]+self.flick_cols[self.move_id]
            full[rows[:,None],cols]=pol
        if a.middle_support_press and hasattr(self,"teacher_pushing"):
            for h in range(2):
                full[:,h*self.per_hand+6:(h+1)*self.per_hand]+=getattr(self,"teacher_support_active",self.teacher_pushing)[:,None]*self.finger_flex["middle"][None,:]*a.middle_support_press/self.cfg.residual_finger_range
        if a.up_right_thumb_relief and hasattr(self,"teacher_pushing"):
            mask=(self.move_id==MOVE_INDEX["U'"])&getattr(self,"teacher_support_active",self.teacher_pushing)
            full[:,self.per_hand+6:2*self.per_hand]-=mask[:,None]*self.finger_flex["thumb"][None,:]*a.up_right_thumb_relief/self.cfg.residual_finger_range
        if getattr(self,"support_override",None) is not None:
            mid=torch.tensor([i for i,name in enumerate(FINGER_JOINTS) if (name.startswith("middle_") or (a.support_tilt_holding and name.startswith("thumb_")) or (a.support_tilt_ring and name.startswith("ring_")))],device=self.device)
            for h in range(2):
                idx=h*self.per_hand+6+mid
                full[:,idx]=self.support_override[:,idx]
        if getattr(self,"wrist_override",None) is not None:
            for h in range(2):full[:,h*self.per_hand:h*self.per_hand+6]=self.wrist_override[:,h]
        return full.clamp(-1,1)
    def _snapshot(self,dropped):
        super()._snapshot(dropped)
        self.snapshot['all_link_contacts']=self.cube_contacts.forces().clone()
        if a.record_wrenches:
            from rubik_lab.contact_wrench import point_contact_wrench
            origin=(self.snapshot['cube_pos']+self.scene.env_origins)[:,None,:]
            w=[]
            for body in ('core','top'):
                view=self.cube_contacts.sensors[body].contact_physx_view
                mag,points,normals,_,counts,starts=view.get_contact_data(dt=self.physics_dt)
                normal=point_contact_wrench(mag.reshape(-1,1)*normals,points,counts,starts,origin)
                forces,points,counts,starts=view.get_friction_data(dt=self.physics_dt)
                friction=point_contact_wrench(forces,points,counts,starts,origin)
                w.append(torch.stack([normal,friction],1))
            self.snapshot['cube_pair_wrenches']=torch.stack(w,1)
        self.snapshot['finger_velocity']=torch.stack([self.hands[side].data.joint_vel[:,self.finger_ids[side]] for side in ('left','right')],1).clone()
u=MechanicalEnv(cfg);u.record=True
if a.cube_armature is not None:
    u.cube.write_joint_armature_to_sim(torch.full((u.num_envs,1),a.cube_armature,device=u.device),joint_ids=[u.twist_id])
u.diagnostic_substeps=[]
if a.support_tilt_gain:
    from rubik_lab.diagnostics.support_control import SupportTiltController
    support_fingers=["middle"]+(["thumb"] if a.support_tilt_holding else [])+(["ring"] if a.support_tilt_ring else [])
    support_controllers=[SupportTiltController(u,a.support_tilt_gain,finger=finger,actual_pose=a.support_tilt_actual_pose) for finger in support_fingers]
if a.wrist_tilt_gain:
    from rubik_lab.diagnostics.support_control import WristTiltController
    if cfg.ee_action_mode not in ('residual','flick'):raise ValueError('Wrist diagnostic requires residual pose control')
    wrist_controller=WristTiltController(u,a.wrist_tilt_gain,rate=a.wrist_tilt_rate_deg_s*3.141592653589793/180)
class TeacherView:
    def __init__(self,env):
        self.env=env;self.cfg=copy.copy(env.cfg);self.cfg.action_space=2*env.per_hand
    def __getattr__(self,key):return getattr(self.env,key)
teacher=ScriptedTeacher(TeacherView(u))
teacher.c["flick_plan_mode"]=a.plan_frame
teacher.c["flick_retry_plan_mode"]=a.retry_plan_frame
teacher.c["flick_track_normal"]=a.track_normal
teacher.c["flick_stroke_limit_deg"]=a.stroke_limit_deg
teacher.c["flick_contact_offset"]=None if a.contact_offset_mm is None else a.contact_offset_mm/1000
teacher.c["flick_contact_height"]=None if a.contact_height_mm is None else a.contact_height_mm/1000
if a.progress=='legacy':
    teacher._flick_progress=types.MethodType(lambda self:getattr(self.env,"signed_err",self.env.move_target[self.env.move_id])-self.env.move_target[self.env.move_id],teacher)
forced=torch.tensor([MOVE_INDEX['U']]*(a.num_envs//2)+[MOVE_INDEX["U'"]]*(a.num_envs//2),device=u.device)
records=[];trace=[];contacts=[];commands=[];states=[];planner_state=[];wrenches=[];withdrawal_stages=[]
try:
    for trial in range(a.rounds):
        if hasattr(u,"teacher_pushing"):u.teacher_pushing.zero_()
        if hasattr(u,"teacher_support_active"):u.teacher_support_active.zero_()
        u.support_override=None
        u.wrist_override=None
        u.diagnostic_recording=False
        u.forced_move_id=forced.clone();u.reset(seed=a.seed+trial)
        u.diagnostic_trial=trial;u.diagnostic_physics_tick=0;u.diagnostic_recording=True
        teacher.reset(torch.arange(a.num_envs,device=u.device))
        if a.wrist_tilt_gain:
            wrist_controller.reset();u.wrist_mask.fill_(1.0)
        active=torch.ones(a.num_envs,dtype=torch.bool,device=u.device)
        previous_push=torch.zeros_like(active); withdrawing=torch.zeros_like(active); release_command=None
        if a.radial_withdrawal:
            from rubik_lab.diagnostics.withdrawal import RadialWithdrawal
            withdrawal=RadialWithdrawal(u,proximal_first=a.proximal_first_return)
        for step in range(u.max_episode_length+1):
            if step % 40 == 0: print(f"[mechanical] round={trial} step={step} active={int(active.sum())}",flush=True)
            if a.retry_after_ready:
                # Previous pre-reset snapshot is current for every still-active trial.
                if step == 0:
                    u.flick_retry_ready=torch.ones_like(active)
                else:
                    prior=u.snapshot
                    hand=u.flick_hand[u.move_id]; cols=u.flick_cols[u.move_id]
                    rows=torch.arange(a.num_envs,device=u.device)
                    velocity=prior['finger_velocity'][rows[:,None],hand[:,None],cols].abs().amax(-1)
                    u.flick_retry_ready=(prior['finger_ready_error']<=cfg.cycle_finger_tol)&(velocity<=cfg.cycle_finger_speed)&(prior['release_force']<=cfg.cycle_release_force)
            u.diagnostic_active=active.clone()
            full=teacher.act()
            u.teacher_finger_command=full.clone()
            u.teacher_pushing=teacher.phase==F_PUSH
            u.teacher_support_active=(getattr(u,"teacher_support_active",torch.zeros_like(u.teacher_pushing))|u.teacher_pushing) if a.support_persistent else u.teacher_pushing.clone()
            rows=torch.arange(a.num_envs,device=u.device)
            hand=u.flick_hand[u.move_id];cols=u.flick_cols[u.move_id]
            actions=full[rows[:,None],(hand*u.per_hand+6)[:,None]+cols]
            # Both arms use the same frozen-cycle ready/release convention.
            ready=u._idle_offsets()[cols]/cfg.residual_finger_range
            actions=torch.where((teacher.phase==F_PUSH)[:,None],actions,ready)
            if a.release_rate is not None:
                push=teacher.phase==F_PUSH
                begin=previous_push&~push
                actual_q=torch.stack([u.hands[side].data.joint_pos[:,u.finger_ids[side]] for side in ("left","right")],1)
                grasp=torch.stack([u.finger_grasp[side] for side in ("left","right")],1)
                actual_action=(actual_q[rows[:,None],hand[:,None],cols]-grasp[rows[:,None],hand[:,None],cols])/cfg.residual_finger_range
                if release_command is None:release_command=ready.clone()
                release_command=torch.where(begin[:,None],actual_action,release_command)
                withdrawing=(withdrawing|begin)&~push
                delta=a.release_rate*u.step_dt/cfg.residual_finger_range
                release_command=release_command+(ready-release_command).clamp(-delta,delta)
                actions=torch.where(withdrawing[:,None],release_command,actions)
                previous_push=push.clone()
            if a.radial_withdrawal:
                actions=withdrawal.apply(actions,ready,teacher.phase==F_PUSH,teacher.planner)
                withdrawal_stages.append(withdrawal.stage.cpu().numpy().copy())
            if a.support_tilt_gain:
                u.support_override=None
                support_mask=u.teacher_support_active & (~u.teacher_pushing if a.support_tilt_phase=="recovery" else torch.ones_like(u.teacher_pushing))
                corrected=u._expand_flick_action(actions)
                for controller in support_controllers:corrected=controller.apply(corrected,support_mask,teacher.planner)
                u.support_override=corrected
            if a.wrist_tilt_gain:
                u.wrist_override=wrist_controller.apply(u.teacher_support_active)
            _,_,terminated,truncated,_=u.step(actions)
            s=u.snapshot
            if a.record_wrenches:wrenches.append(s['cube_pair_wrenches'].cpu().numpy())
            contacts.append(s['all_link_contacts'].cpu().numpy())
            states.append(torch.cat([s['cube_pos'],s['cube_quat'],s['ee_pose'].flatten(1),s['finger_pos'].flatten(1),s['finger_velocity'].flatten(1),s['finger_tgt'].flatten(1),s['tactile_ft'].flatten(1)],-1).cpu().numpy())
            planner_state.append(torch.stack([teacher.f_prog,teacher.f_kmax.float(),teacher.f_phi0,teacher.f_press,teacher.f_lead,teacher.f_tries.float()],-1).cpu().numpy())
            commands.append(u._expand_flick_action(actions).cpu().numpy())
            error=torch.rad2deg(s['twist_error'])
            done=s['primitive_done'].bool()
            failed=s['cycle_failed'].bool()|s['dropped'].bool()|terminated.bool()
            ended=active&(done|failed|truncated.bool()|((step+1)*u.step_dt>=6))
            trace.append(torch.stack([torch.full_like(error,trial),torch.full_like(error,step),active.float(),teacher.phase.float(),error,s['twist_velocity'],s['finger_ready_error'],s['release_force'],done.float(),failed.float()],-1).cpu().numpy())
            for i in ended.nonzero().flatten().tolist():
                reason='motion_or_drop' if bool(failed[i]) else ('success' if bool(done[i]) else 'timeout')
                records.append({'trial':trial,'env':i,'move':ALL_MOVES[int(forced[i])],'reason':reason,'error_deg':float(error[i]),'finger_ready_error':float(s['finger_ready_error'][i]),'phase':int(teacher.phase[i]),'seconds':(step+1)*u.step_dt,'motion_peaks':{'cube_position_m':float(s['peak_cube_pos'][i]),'cube_rotation_rad':float(s['peak_cube_rot'][i]),'wrist_position_m':float(s['peak_wrist_pos'][i]),'wrist_rotation_rad':float(s['peak_wrist_rot'][i])}})
            active &= ~ended
            if not active.any():break
        if active.any():raise RuntimeError('Incomplete fixed quota')
        print(f'[mechanical] round {trial+1}/{a.rounds}',flush=True)
    summary={}
    for move in ('U',"U'"):
        subset=[r for r in records if r['move']==move]
        summary[move]={'n':len(subset),'success':sum(r['reason']=='success' for r in subset),'motion_or_drop':sum(r['reason']=='motion_or_drop' for r in subset)}
    result={'scope':'Privileged mechanical diagnostic with active-index commands and explicitly selected support controls; frozen physical contract unchanged. Not blind-policy acceptance.','contact_height_mm':a.contact_height_mm,'middle_support_press_rad':a.middle_support_press,'support_persistent':a.support_persistent,'release_rate_rad_s':a.release_rate,'track_normal':a.track_normal,'support_tilt_gain':a.support_tilt_gain,'radial_withdrawal':a.radial_withdrawal,'proximal_first_return':a.proximal_first_return,'contact_offset_mm':a.contact_offset_mm,'up_right_thumb_relief_rad':a.up_right_thumb_relief,'stroke_limit_deg':a.stroke_limit_deg,'finger_control':a.finger_control,'progress':a.progress,'plan_frame':a.plan_frame,'seed':a.seed,'num_envs':a.num_envs,'rounds':a.rounds,'shared_adaptation':'Active index commands plus explicitly selected diagnostic support/withdrawal controls; wrist commands zero unless explicit bounded tilt diagnostic. Ready targets and physical acceptance gates are unchanged.','summary':summary,'episodes':records,'trace_columns':['trial','step','active','phase','angle_error_deg','layer_speed','finger_ready_error','release_normal_force','full_ready','failed']}
    result['arguments']=vars(a)
    result['configuration_sources']={'checkpoint_settings':'logs/rsl_rl/rubik_primitive_cycle/U_blind_full/model.pt','grasp_cache':cfg.grasp_cache_path,'physics_dt':u.physics_dt,'control_dt':u.step_dt,'external_forces_every_iteration':cfg.sim.physx.enable_external_forces_every_iteration,'solver_type':cfg.sim.physx.solver_type}
    result['states_columns']={'cube_pos':[0,3],'cube_quat_wxyz':[3,7],'wrist_pose_world_left_right':[7,21],'finger_pos':[21,65],'finger_velocity':[65,109],'finger_target':[109,153],'tactile_ft':[153,213]}
    result['planner_state_columns']=['commanded_waypoint','last_feasible_waypoint','initial_directed_phase_rad','press_m','lead_waypoints','retry_count']
    result['withdrawal_stage_labels']=(['teacher_or_idle','radial_retreat','proximal_return','remaining_ready'] if a.proximal_first_return else ['teacher_or_idle','radial_retreat','rate_limited_ready']) if a.radial_withdrawal else None
    result['wrench_layout']='step,env,cube_body(core/top),component(normal/friction),hand_link54,forceXYZ+momentXYZ; world axes, moments about cube core, final physics substep only' if a.record_wrenches else None
    if a.record_substeps:
        result['cube_runtime_mass_properties']={'joint_armature':u.cube.root_physx_view.get_dof_armatures().cpu().tolist(),'mass_kg':u.cube.root_physx_view.get_masses().cpu().tolist(),'inertia':u.cube.root_physx_view.get_inertias().cpu().tolist(),'com_local_pose':u.cube.root_physx_view.get_coms().cpu().tolist(),'note':'PhysX COM transform quaternion uses xyzw; inertia is3x3 flattened.'}
    result['substep_schema']={'timing':'Before next physics step, previous-step contact and body state; tick0 omitted, first1.2s only','body_names':u.cube.body_names,'body_state':'world link-origin position3/link quaternion_wxyz4/COM linear_velocity3/COM angular_velocity3','wrench':'frame,env,core_top,normal_friction,forceXYZ_momentXYZ_about_current_core; summed hand links'} if a.record_substeps else None
    result['states_timing']='Pre-reset physics snapshot; planner state is the command state before this physics step.'
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out.with_suffix('.npz'),trace=np.asarray(trace),contacts=np.asarray(contacts),commands=np.asarray(commands),states=np.asarray(states),planner_state=np.asarray(planner_state),env_origins=u.scene.env_origins.cpu().numpy(),wrenches=np.asarray(wrenches),withdrawal_stages=np.asarray(withdrawal_stages),substep_index=np.asarray([(x[0],x[1]) for x in u.diagnostic_substeps]),substep_active=np.asarray([x[2] for x in u.diagnostic_substeps]),substep_body_state=np.asarray([x[3] for x in u.diagnostic_substeps]),substep_wrenches=np.asarray([x[4] for x in u.diagnostic_substeps]))
    out.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    u.close();app.close()
