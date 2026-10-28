"""Actual-state left fingertip recontact diagnostic; no reset or action credit."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares, minimize
from .left_return import LeftReturnPath



def cube_face_centerward_tangent(target,pad_normal,angle_rad):
    """Privileged geometric face selection, not a tactile surface classifier."""
    target=np.asarray(target,dtype=float);normal=np.asarray(pad_normal,dtype=float)
    if target.shape!=(3,) or normal.shape!=(3,) or not np.isfinite(np.r_[target,normal,angle_rad]).all():raise ValueError('Invalid contact geometry')
    length=np.linalg.norm(normal)
    if length<1e-9:raise ValueError('Invalid pad normal')
    turn=Rotation.from_rotvec([0,0,float(angle_rad)])
    inward=turn.apply(np.array([[1.,0,0],[-1.,0,0],[0,1.,0],[0,-1.,0]]))
    scores=inward@(normal/length);order=np.argsort(scores)
    if scores[order[-1]]-scores[order[-2]]<.1:raise ValueError('Ambiguous cube contact face')
    face=inward[order[-1]];tangent=np.array([-face[1],face[0],0.])
    if tangent@(-target)<0:tangent=-tangent
    return tangent,face

def solve_recontact(raw,trial,env,press_m=.001,tangent_shift_m=0.,constrained_normal=False,surface_normal=False,surface_depth_m=0.):
    import torch
    from rubik_lab.hand_kinematics import HandFK
    if press_m not in (.001,.0015):raise ValueError('Only preregistered1mm or1.5mm press supported')
    if tangent_shift_m not in (0.,.004,.005):raise ValueError('Only preregistered0mm,4mm or5mm tangent shift supported')
    if surface_depth_m not in (0.,.0004,.0008) or (surface_depth_m and not surface_normal):raise ValueError('Invalid registered surface depth')
    t=np.asarray(raw['trace'])[:,env];c=np.asarray(raw['contact'])[:,env]
    i=len(t)-1
    if i<9 or t[i,0]!=trial or t[i,3]!=2 or not t[i,4] or not t[i,10]:raise ValueError('Recontact requires completed third action')
    tail=t[-10:]
    if not np.array_equal(tail[:,1],np.arange(t[i,1]-9,t[i,1]+1)) or not ((tail[:,0]==trial)&(tail[:,3]==2)&(tail[:,4]>0)&(tail[:,9]>0)&(tail[:,11]==0)).all() or not (c[-10:,:,0,6:11].sum(axis=(1,2))<=.5).all():raise ValueError('Third readiness/release history invalid')
    for k in ('trace','sensory','poses','motor_targets','contact'):
        if not np.isfinite(np.asarray(raw[k])[:,env]).all():raise ValueError('Nonfinite recontact source')
    hits=np.flatnonzero((t[:,0]==trial)&(t[:,3]==2)&(t[:,4]>0)&(t[:,11]==0)&(c[:,1,0,10]>.05))
    if not len(hits):raise ValueError('No observed prior fingertip contact')
    first=int(hits[0]);fk=HandFK('left','cpu',pts_per_link=8,seed=0)
    for table in (fk.T0,fk.axis,fk.link_pts):
        for k in table:table[k]=table[k].double()
    tensor=lambda x:torch.as_tensor(x,dtype=torch.float64)
    def state(j):
        pose=np.asarray(raw['poses'])[j,env];r=Rotation.from_quat(pose[3:7][[1,2,3,0]])
        return np.asarray(raw['sensory'])[j,env,:22].astype(float),r.inv().apply(pose[7:10]-raw['env_origins'][env]-pose[:3]),r.inv().as_matrix()@Rotation.from_quat(pose[10:14][[1,2,3,0]]).as_matrix()
    def pad(q,p,r):
        a,b=fk.pads(fk.forward(tensor(p)[None],tensor(r)[None],tensor(q)[None]))['index'];return a[0].numpy(),b[0].numpy()
    q,p,r=state(i);target,normal=pad(*state(first))
    delta=(float(t[first,5])-float(t[i,5])+45)%90-45
    rotation=Rotation.from_rotvec([0,0,np.deg2rad(delta)])
    normal=rotation.apply(normal);target=rotation.apply(target)+press_m*normal
    if tangent_shift_m:
        tangent,face=cube_face_centerward_tangent(target,normal,np.deg2rad(-float(t[i,5])))
        target+=tangent_shift_m*tangent
    if surface_normal:
        _,normal=cube_face_centerward_tangent(target,normal,np.deg2rad(-float(t[i,5])))
        target+=surface_depth_m*normal
    bias=np.asarray(raw['motor_targets'])[i,env,0].astype(float)-q
    lo=np.asarray(raw['actuator_lower'])[env,0,5:9]-bias[5:9];hi=np.asarray(raw['actuator_upper'])[env,0,5:9]-bias[5:9]
    def residual(x):
        v=q.copy();v[5:9]=x;point,n=pad(v,p,r);return np.r_[(point-target)*1000,(n-normal)*5]
    fit=least_squares(residual,np.clip(q[5:9],lo,hi),bounds=(lo,hi),max_nfev=200,xtol=1e-10,ftol=1e-10,gtol=1e-10)
    if constrained_normal:
        def pose_values(x):
            v=q.copy();v[5:9]=x
            point,n=pad(v,p,r)
            return point,n
        def objective(x):return float(np.sum(((pose_values(x)[0]-target)*1000)**2))
        fit=minimize(objective,fit.x,method='SLSQP',bounds=list(zip(lo,hi)),
                     constraints=[dict(type='ineq',fun=lambda x:float(pose_values(x)[1]@normal)-.950001)],
                     options=dict(maxiter=200,ftol=1e-12))
    solved=q.copy();solved[5:9]=fit.x;point,n=pad(solved,p,r)
    error=float(np.linalg.norm(point-target));dot=float(n@normal)
    if not fit.success or error>.001 or dot<.95:raise ValueError(f'Recontact local pose fit failed: solver_success={fit.success}, position_error_m={error:.9g}, normal_dot={dot:.9g}')
    duration=float(np.max(np.abs(solved-q)))
    if not 0<duration<6:raise ValueError('Recontact original deadline')
    return dict(source_frame=i,source_step=int(t[i,1]),template_frame=first,template_step=int(t[first,1]),q=q,solution=solved,wrist_pos=p,wrist_rot=r,bias=bias,duration=duration,position_error_m=error,normal_dot=dot,target_pad_cube_m=target.tolist(),top_geometry_rotation_deg=delta)


def write_recontact_candidate(raw,prefix,trial,env,press_m=.001,tangent_shift_m=0.,constrained_normal=False,surface_normal=False,surface_depth_m=0.):
    result=solve_recontact(raw,trial,env,press_m,tangent_shift_m,constrained_normal,surface_normal,surface_depth_m);prefix=Path(prefix);source=Path(str(prefix)+'_source');paths=Path(str(prefix)+'_paths')
    if any(p.with_suffix(s).exists() for p in (prefix,source,paths) for s in ('.json','.npz')):raise FileExistsError(prefix)
    prefix.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(source.with_suffix('.npz'),**raw)
    source.with_suffix('.json').write_text(json.dumps(dict(scope='Actual completed history before fourth recontact; no state writes')))
    row=dict(trial=trial,env=env,source_frame=result['source_frame'],source_step=result['source_step'],move="U'",side='left',error_deg=float(raw['trace'][-1,env,5]))
    paths.with_suffix('.json').write_text(json.dumps(dict(source=str(source),results=[row],contact_face='B',contact_press_m=press_m)))
    np.savez_compressed(paths.with_suffix('.npz'),U_initial_q=result['q'][None],U_wrist_pos=result['wrist_pos'][None],U_wrist_rot=result['wrist_rot'][None])
    report=dict(source=str(paths),phase='recontact_geometry',sequential=True,whole_hand=True,variable_joint_indices=list(range(22)),support_contact=True,contact_transport=False,bias_rad=result['bias'].tolist(),duration_seconds=result['duration'],remaining_original_seconds=6.,template_contact_step=result['template_step'],target_pad_cube_m=result['target_pad_cube_m'],position_error_m=result['position_error_m'],normal_dot=result['normal_dot'],top_geometry_rotation_deg=result['top_geometry_rotation_deg'],press_m=press_m,tangent_shift_m=tangent_shift_m,constrained_normal=constrained_normal,orientation_reference=('cube_face' if surface_normal else 'historical_pad'),surface_depth_m=surface_depth_m,scope='Privileged recontact geometry candidate, not action completion',results=[dict(trial=trial,env=env,waypoint=0,measured_feasible=False,segment_pass=False)])
    prefix.with_suffix('.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(prefix.with_suffix('.npz'),initial=np.r_[result['q'],np.zeros(6)][None],solution=np.r_[result['solution'],np.zeros(6)][None])
    return report


class LeftRecontactPath(LeftReturnPath):
    def __init__(self,prefix):
        from .live_whole_hand import LiveWholeHandProbe
        self.prefix=Path(prefix);r=json.loads(self.prefix.with_suffix('.json').read_text())
        audit=json.loads(Path(str(prefix)+'_audit.json').read_text())
        if r['phase']!='recontact_geometry' or audit['active_side']!='left' or audit['source']!=str(prefix) or not audit['cube_clearance']:raise ValueError('Invalid recontact audit scope')
        LiveWholeHandProbe.validate_audit(audit,support_contact=True,expected_end=0)
        if not 0<=audit['results'][0]['cube_clearance']['active_pad_top_signed_overlap_mm']<=1.:raise ValueError('Recontact pad geometry outside0..1mm')
        source=json.loads(Path(r['source']).with_suffix('.json').read_text())
        if len(source['results'])!=1 or source['results'][0]['side']!='left':raise ValueError('Invalid recontact source')
        self.source=source['results'][0];self.raw_prefix=Path(source['source']);self.started=False
        with np.load(self.prefix.with_suffix('.npz')) as z:initial=z['initial'][0];solution=z['solution']
        if solution.shape!=(1,28):raise ValueError('Expected one recontact segment')
        fixed=[j for j in range(28) if j not in range(5,9)]
        if not np.isfinite(solution).all() or not np.array_equal(solution[0,fixed],initial[fixed]):raise ValueError('Recontact changes wrist/support')
        self.motor=np.vstack([initial[:22],solution[0,:22]])+np.asarray(r['bias_rad'])
        self.duration=float(np.max(np.abs(self.motor[1]-self.motor[0])));self.times=np.array([0.,self.duration])
        if not 0<self.duration<6:raise ValueError('Recontact deadline')
        with np.load(self.raw_prefix.with_suffix('.npz')) as z:
            i=self.source['source_frame'];env=self.source['env']
            if i!=len(z['trace'])-1 or z['trace'][i,env,1]!=self.source['source_step'] or z['trace'][i,env,3]!=2 or not z['trace'][i,env,10]:raise ValueError('Recontact source is not latest completed third state')
            if not np.array_equal(initial[:22],z['sensory'][i,env,:22]) or not np.allclose(self.motor[0],z['motor_targets'][i,env,0],rtol=0,atol=1e-12):raise ValueError('Recontact source mismatch')
            if np.any(self.motor<z['actuator_lower'][env,0]-1e-12) or np.any(self.motor>z['actuator_upper'][env,0]+1e-12):raise ValueError('Recontact motor bounds')
            self.original={k:z[k][:i+1,env].copy() for k in ('trace','sensory','poses','motor_targets','wrist_targets','contact','core_angular_speed')}


class LiveFourthRecontact:
    def __init__(self,owner,press_m=.001,center_m=0.,turn_after_contact=False,surface_depth_m=0.,turn_candidate=None,lift_candidate=None,recontact_candidate=None,live_stroke_cycle=False,turn_depth_reserve_m=0.,second_stroke=False,second_stroke_feedback=False,second_stroke_degrees=5.,second_stroke_feedback_gain=1.,recycle_completed_stroke=False,completed_lift_feedback=False,completed_recontact_outside_m=.0003,completed_recontact_feedback=False,entry_delay_frames=10):self.owner=owner;self.press_m=press_m;self.center_m=center_m;self.path=None;self.finished=False;self.finished_step=None;self.turn_after_contact=turn_after_contact;self.surface_depth_m=surface_depth_m;self.turn_path=None;self.turn_finished=False;self.turn_candidate=turn_candidate;self.lift_candidate=lift_candidate;self.lift_path=None;self.turn_finished_step=None;self.lift_finished_step=None;self.recontact_candidate=recontact_candidate;self.internal_recontact_path=None;self.live_stroke_cycle=live_stroke_cycle;self.turn_depth_reserve_m=turn_depth_reserve_m;self.second_stroke=second_stroke;self.internal_recontact_finished_step=None;self.second_turn_path=None;self.second_turn_finished=False;self.second_stroke_feedback=second_stroke_feedback;self.second_feedback_started=False;self.second_turn_reference=None;self.second_stroke_degrees=second_stroke_degrees;self.second_stroke_feedback_gain=second_stroke_feedback_gain;self.recycle_completed_stroke=recycle_completed_stroke;self.release_cycle=None;self.completed_lift_feedback=completed_lift_feedback;self.completed_recontact_outside_m=completed_recontact_outside_m;self.completed_recontact_feedback=completed_recontact_feedback;self.entry_delay_frames=entry_delay_frames
    def _plan_internal(self,e,raw,trial,finished,prior=None,drive_boundary=None,label=None,recontact_outside_m=.0003):
        import torch
        from .stroke_cycle import write_internal_candidate
        env=self.owner.env
        if abs(float(e.step_dt)-.025)>1e-12 or int(e.cycle_step[env])!=int(raw['trace'][-1,env,1])-self.path.source['source_step']:raise ValueError('Internal original clock mismatch')
        raw=dict(raw)
        raw.update(env_origins=e.scene.env_origins.cpu().numpy(),actuator_lower=torch.stack([e.finger_lower[s] for s in ('left','right')],1).cpu().numpy(),actuator_upper=torch.stack([e.finger_upper[s] for s in ('left','right')],1).cpu().numpy())
        label=label or ('stroke_recontact' if prior is not None else 'stroke_lift')
        prefix=Path(str(self.owner.out)+'_'+label)
        write_internal_candidate(raw,prefix,trial,env,finished,prior,drive_boundary,recontact_outside_m)
        self.owner._child(['hand-collision','--solutions',str(prefix),'--out',str(prefix)+'_audit','--usd-convex','--cooked-hulls','outputs/diagnostics/hand_physx_cooked_assets','--cube-clearance','--segment-samples','19'],label+'_audit')
        return prefix

    def prepare(self,e,trial,step,events):
        import torch
        env=self.owner.env
        if self.path is None:
            if int(e.cycle_step[env])!=0:raise ValueError('Recontact must begin at unchanged fourth-action handoff')
            raw={k:np.asarray(self.owner.history[k]) for k in ('trace','sensory','poses','motor_targets','wrist_targets','contact','core_angular_speed','ready_targets')}
            raw.update(env_origins=e.scene.env_origins.cpu().numpy(),actuator_lower=torch.stack([e.finger_lower[s] for s in ('left','right')],1).cpu().numpy(),actuator_upper=torch.stack([e.finger_upper[s] for s in ('left','right')],1).cpu().numpy())
            prefix=Path(str(self.owner.out)+'_fourth_recontact')
            write_recontact_candidate(raw,prefix,trial,env,self.press_m,self.center_m,constrained_normal=self.center_m>0,surface_normal=self.surface_depth_m>0,surface_depth_m=self.surface_depth_m)
            self.owner._child(['hand-collision','--solutions',str(prefix),'--out',str(prefix)+'_audit','--usd-convex','--cooked-hulls','outputs/diagnostics/hand_physx_cooked_assets','--cube-clearance','--segment-samples','19'],'fourth_recontact_audit')
            self.path=LeftRecontactPath(prefix);self.path.verify_history(self.owner.history)
            events.append(dict(event='fourth_recontact_start',trial=trial,step=step,env=env,source=str(prefix),duration_seconds=self.path.duration,scope='Actual-state audited contact component, not quarter-turn execution or blind policy; no state/reference/clock writes'))
        elapsed=(step-self.path.source['source_step'])*float(e.step_dt)
        cols=e.flick_cols[e.move_id[env]].clone()
        if cols.tolist()!=[5,6,7,8]:raise RuntimeError('Unexpected fourth recontact joint mapping')
        values=torch.as_tensor(self.path.sample(elapsed),device=e.device,dtype=e.finger_tgt['left'].dtype)
        e.fourth_recontact_target=(env,'left',cols,values)
        if elapsed>=self.path.duration and not self.finished:
            events.append(dict(event='fourth_recontact_targets_finished',trial=trial,step=step,env=env));self.finished=True;self.finished_step=step
        if self.turn_after_contact and self.finished_step is not None:
            raw={k:np.asarray(self.owner.history[k]) for k in ('trace','sensory','poses','motor_targets','wrist_targets','contact','core_angular_speed','ready_targets')}
            if self.turn_path is None and first_turn_source(raw,trial,env,self.finished_step,entry_delay_frames=self.entry_delay_frames) is not None:
                if first_turn_source(raw,trial,env,self.finished_step,entry_delay_frames=self.entry_delay_frames)!=len(raw['trace'])-1:raise ValueError('Missed first eligible turn entry')
                if abs(float(e.step_dt)-.025)>1e-12 or int(e.cycle_step[env])!=int(raw['trace'][-1,env,1])-self.path.source['source_step']:raise ValueError('Turn original clock mismatch')
                raw.update(env_origins=e.scene.env_origins.cpu().numpy(),actuator_lower=torch.stack([e.finger_lower[s] for s in ('left','right')],1).cpu().numpy(),actuator_upper=torch.stack([e.finger_upper[s] for s in ('left','right')],1).cpu().numpy())
                prefix=Path(self.turn_candidate) if self.turn_candidate else Path(str(self.owner.out)+'_contact_turn')
                if self.turn_candidate is None:
                    write_contact_turn(raw,prefix,trial,env,self.finished_step,self.turn_depth_reserve_m,entry_delay_frames=self.entry_delay_frames)
                    self.owner._child(['hand-collision','--solutions',str(prefix),'--out',str(prefix)+'_audit','--usd-convex','--cooked-hulls','outputs/diagnostics/hand_physx_cooked_assets','--cube-clearance','--segment-samples','19'],'contact_turn_audit')
                self.turn_path=ContactTurnPath(prefix)
                if self.turn_path.source['trial']!=trial or self.turn_path.source['env']!=env or self.turn_path.contact_target_finished_step!=self.finished_step:raise ValueError('Turn candidate source or contact timing mismatch')
                if np.any(self.turn_path.motor<raw['actuator_lower'][env,0]-1e-12) or np.any(self.turn_path.motor>raw['actuator_upper'][env,0]+1e-12):raise ValueError('Turn candidate current actuator bounds')
                self.turn_path.verify_history(self.owner.history)
                events.append(dict(event='contact_turn_start',trial=trial,step=step,env=env,source=str(prefix),source_step=self.turn_path.source['source_step'],duration_seconds=self.turn_path.duration,target_degrees=self.turn_path.target_degrees,scope='Privileged contact-turn diagnostic; original clock, target and physical state retained; no quarter-turn credit'))
            if self.turn_path is not None:
                turn_elapsed=(step-self.turn_path.source['source_step'])*float(e.step_dt)
                values=torch.as_tensor(self.turn_path.sample(turn_elapsed),device=e.device,dtype=e.finger_tgt['left'].dtype)
                e.fourth_recontact_target=(env,'left',cols,values)
                if turn_elapsed>=self.turn_path.duration and not self.turn_finished:
                    events.append(dict(event='contact_turn_targets_finished',trial=trial,step=step,env=env));self.turn_finished=True;self.turn_finished_step=step
                if (self.lift_candidate is not None or self.live_stroke_cycle) and self.turn_finished_step is not None:
                    from .stroke_cycle import StrokeLiftPath,first_stroke_lift_source
                    eligible=first_stroke_lift_source(raw,trial,env,self.turn_finished_step)
                    if self.lift_path is None and eligible is not None:
                        if eligible!=len(raw['trace'])-1:raise ValueError('Missed first internal lift window')
                        if self.live_stroke_cycle:self.lift_candidate=self._plan_internal(e,raw,trial,self.turn_finished_step)
                        self.lift_path=StrokeLiftPath(self.lift_candidate,self.turn_finished_step)
                        if self.lift_path.source['env']!=env or self.lift_path.source['trial']!=trial:raise ValueError('Lift source identity')
                        self.lift_path.verify_history(self.owner.history)
                        events.append(dict(event='stroke_lift_start',trial=trial,step=step,env=env,source=str(self.lift_candidate),turn_targets_finished_step=self.turn_finished_step,duration_seconds=self.lift_path.duration))
                    if self.lift_path is not None:
                        values=torch.as_tensor(self.lift_path.sample((step-self.lift_path.source['source_step'])*float(e.step_dt)),device=e.device,dtype=e.finger_tgt['left'].dtype)
                        e.fourth_recontact_target=(env,'left',cols,values)
                        if self.lift_finished_step is None and (step-self.lift_path.source['source_step'])*float(e.step_dt)>=self.lift_path.duration:
                            self.lift_finished_step=step
                            events.append(dict(event='stroke_lift_targets_finished',trial=trial,step=step,env=env))
                        if (self.recontact_candidate is not None or self.live_stroke_cycle) and self.lift_finished_step is not None:
                            from .stroke_cycle import StrokeRecontactPath,first_stroke_recontact_source
                            eligible=first_stroke_recontact_source(raw,trial,env,self.lift_finished_step)
                            if self.internal_recontact_path is None and eligible is not None:
                                if eligible!=len(raw['trace'])-1:raise ValueError('Missed internal recontact window')
                                if self.live_stroke_cycle:self.recontact_candidate=self._plan_internal(e,raw,trial,self.lift_finished_step,self.turn_path.source['source_step'])
                                path=StrokeRecontactPath(self.recontact_candidate,self.lift_finished_step)
                                if path.source['env']!=env or path.source['trial']!=trial:raise ValueError('Internal recontact identity')
                                path.verify_history(self.owner.history);self.internal_recontact_path=path
                                events.append(dict(event='stroke_recontact_start',trial=trial,step=step,env=env,source=str(self.recontact_candidate),lift_targets_finished_step=self.lift_finished_step,duration_seconds=path.duration))
                            if self.internal_recontact_path is not None:
                                path=self.internal_recontact_path
                                values=torch.as_tensor(path.sample((step-path.source['source_step'])*float(e.step_dt)),device=e.device,dtype=e.finger_tgt['left'].dtype)
                                e.fourth_recontact_target=(env,'left',cols,values)
                                elapsed=(step-path.source['source_step'])*float(e.step_dt)
                                if self.internal_recontact_finished_step is None and elapsed>=path.duration:
                                    self.internal_recontact_finished_step=step
                                    events.append(dict(event='stroke_recontact_targets_finished',trial=trial,step=step,env=env))
                                if self.second_stroke and self.internal_recontact_finished_step is not None:
                                    finished=self.internal_recontact_finished_step
                                    eligible=first_turn_source(raw,trial,env,finished,entry_delay_frames=self.entry_delay_frames)
                                    if self.second_turn_path is None and eligible is not None:
                                        if eligible!=len(raw['trace'])-1:raise ValueError('Missed second turn window')
                                        if abs(float(e.step_dt)-.025)>1e-12 or int(e.cycle_step[env])!=int(raw['trace'][-1,env,1])-self.path.source['source_step']:raise ValueError('Second turn original clock mismatch')
                                        raw.update(env_origins=e.scene.env_origins.cpu().numpy(),actuator_lower=torch.stack([e.finger_lower[s] for s in ('left','right')],1).cpu().numpy(),actuator_upper=torch.stack([e.finger_upper[s] for s in ('left','right')],1).cpu().numpy())
                                        prefix=Path(str(self.owner.out)+'_second_turn')
                                        write_contact_turn(raw,prefix,trial,env,finished,self.turn_depth_reserve_m,self.second_stroke_degrees,self.entry_delay_frames)
                                        self.owner._child(['hand-collision','--solutions',str(prefix),'--out',str(prefix)+'_audit','--usd-convex','--cooked-hulls','outputs/diagnostics/hand_physx_cooked_assets','--cube-clearance','--segment-samples','19'],'second_turn_audit')
                                        self.second_turn_path=ContactTurnPath(prefix);self.second_turn_path.verify_history(self.owner.history)
                                        with np.load(prefix.with_suffix('.npz')) as candidate:self.second_turn_reference=candidate['solution'][-1,5:9].copy()
                                        events.append(dict(event='second_contact_turn_start',trial=trial,step=step,env=env,source=str(prefix),source_step=self.second_turn_path.source['source_step'],duration_seconds=self.second_turn_path.duration,target_degrees=self.second_turn_path.target_degrees,scope='Second privileged stroke; original goal/clock/state retained; no atomic completion credit'))
                                    if self.second_turn_path is not None:
                                        path=self.second_turn_path;elapsed=(step-path.source['source_step'])*float(e.step_dt)
                                        command=path.sample(elapsed)
                                        if self.second_stroke_feedback and elapsed>=path.duration+.25:
                                            from .whole_hand_path import bounded_index_feedback
                                            command,_,_,_=bounded_index_feedback(self.second_turn_reference,raw['sensory'][-1,env,5:9],path.motor[-1,5:9],raw['motor_targets'][-1,env,0,5:9],e.finger_lower['left'][env,5:9].cpu().numpy(),e.finger_upper['left'][env,5:9].cpu().numpy(),float(e.step_dt),gain=self.second_stroke_feedback_gain)
                                            if not self.second_feedback_started:
                                                self.second_feedback_started=True;events.append(dict(event='second_stroke_feedback_start',trial=trial,step=step,env=env,gain=self.second_stroke_feedback_gain,scope='Previous completed proprioception only; declaredgain, cap.1rad,1rad/s, originalmotorbounds; privileged pathreference remains fixed'))
                                        values=torch.as_tensor(command,device=e.device,dtype=e.finger_tgt['left'].dtype)
                                        e.fourth_recontact_target=(env,'left',cols,values)
                                        if elapsed>=path.duration and not self.second_turn_finished:
                                            self.second_turn_finished=True;events.append(dict(event='second_contact_turn_targets_finished',trial=trial,step=step,env=env))

        if self.recycle_completed_stroke and self.second_feedback_started:
            from .stroke_cycle import ReleaseRecontactCycle,StrokeLiftPath,StrokeRecontactPath
            if self.release_cycle is None:
                finished=next(x['step'] for x in events if x['event']=='second_contact_turn_targets_finished')
                feedback=next(x['step'] for x in events if x['event']=='second_stroke_feedback_start')
                self.release_cycle=ReleaseRecontactCycle(self.second_turn_path.source['source_step'],finished,feedback,self.second_stroke_degrees,self.completed_lift_feedback,self.completed_recontact_feedback)
            def plan(phase,finished,prior,boundary):
                prefix=self._plan_internal(e,raw,trial,finished,prior,boundary,'completed_cycle_'+phase,self.completed_recontact_outside_m if phase=='recontact' else .0003)
                return (StrokeLiftPath if phase=='lift' else StrokeRecontactPath)(prefix,finished)
            if self.completed_lift_feedback or self.completed_recontact_feedback:
                raw.update(actuator_lower=torch.stack([e.finger_lower[s] for s in ('left','right')],1).cpu().numpy(),actuator_upper=torch.stack([e.finger_upper[s] for s in ('left','right')],1).cpu().numpy())
            command=self.release_cycle.command(raw,trial,env,step,float(e.step_dt),plan,events)
            if command is not None:
                e.fourth_recontact_target=(env,'left',cols,torch.as_tensor(command,device=e.device,dtype=e.finger_tgt['left'].dtype))


def summarize_recontact(report,raw):
    """Independent component scoring; never grants action completion."""
    events=[e for e in report.get('post_return_events',[]) if e['event']=='fourth_recontact_start']
    if len(events)!=1:raise ValueError('Missing unique recontact start')
    event=events[0];env=event['env'];trial=event['trial'];prefix=Path(event['source'])
    plan=json.loads(prefix.with_suffix('.json').read_text())
    source=json.loads(Path(plan['source']).with_suffix('.json').read_text());row=source['results'][0]
    t=np.asarray(raw['trace'])[:,env];i=int(row['source_frame'])
    if event['step']!=row['source_step']+1 or t[i,1]!=row['source_step']:raise ValueError('Noncausal recontact source')
    with np.load(Path(source['source']).with_suffix('.npz')) as old:
        for key in ('trace','sensory','poses','motor_targets','wrist_targets','contact','core_angular_speed'):
            a=np.asarray(raw[key])[:i+1,env];b=old[key][:,env]
            if not np.isfinite(a).all() or not np.array_equal(a,b):raise ValueError('Recontact actual source mismatch: '+key)
    idx=np.flatnonzero((t[:,0]==trial)&(t[:,3]==3)&(t[:,4]>0))
    if not len(idx) or not np.array_equal(t[idx,1],np.arange(event['step'],event['step']+len(idx))):raise ValueError('Missing contact component frames')
    with np.load(prefix.with_suffix('.npz')) as path:
        motors=np.vstack([path['initial'][0,:22],path['solution'][0,:22]])+np.asarray(plan['bias_rad'])
    duration=float(np.max(np.abs(motors[1]-motors[0])))
    elapsed=(t[idx,1]-row['source_step'])/report['control_hz']
    expected=motors[0,5:9]+np.minimum(elapsed/duration,1)[:,None]*(motors[1,5:9]-motors[0,5:9])
    actual=np.asarray(raw['motor_targets'])[idx,env,0,5:9]
    error=float(np.max(np.abs(actual-expected)))
    rate=float(np.max(np.abs(np.diff(np.vstack([motors[0,5:9],actual]),axis=0)))*report['control_hz'])
    if not np.isfinite(actual).all() or error>1e-6 or rate>1.00001:raise ValueError('Recontact motor interpolation/rate')
    if np.any(actual<raw['actuator_lower'][env,0,5:9]-1e-7) or np.any(actual>raw['actuator_upper'][env,0,5:9]+1e-7):raise ValueError('Recontact motor bounds')
    c=np.asarray(raw['contact'])[idx,env];pose=np.asarray(raw['poses'])[idx,env];sens=np.asarray(raw['sensory'])[idx,env]
    if not all(np.isfinite(v).all() for v in (c,pose,sens,t[idx])):raise ValueError('Nonfinite contact component measurement')
    pad=c[:,1,0,10];rigid=c[:,:,0,6:10].sum(axis=(1,2))
    support=np.stack([c[:,:,h,:][:,:,[5,10,15,20,26]].max(axis=(1,2))>.05 for h in range(2)],axis=1).all(axis=1)
    reference=np.asarray(raw['poses'])[i,env]
    displacement=np.linalg.norm(pose[:,:3]-reference[:3],axis=1)
    rotation=(Rotation.from_quat(pose[:,3:7][:,[1,2,3,0]])*Rotation.from_quat(reference[3:7][[1,2,3,0]]).inv()).magnitude()
    disturbance=np.abs(t[idx,5]-t[idx[0],5])
    good=(pad>.05)&(pad<=5)&(rigid<=.5)&support&(t[idx,11]==0)&(displacement<=.002)&(rotation<=np.deg2rad(2))&(disturbance<=5)
    end_ok=len(idx)>=10 and bool(good[-10:].all())
    complete=abs(float(elapsed[-1])-6)<1e-5
    safety=bool((t[idx,11]==0).all() and support.all() and (t[idx,12]<=.015).all() and (t[idx,13]<=np.deg2rad(15)).all() and (t[idx,14]<=.010).all() and (t[idx,15]<=np.deg2rad(10)).all())
    hits=np.flatnonzero(pad>.05)
    return dict(scope='Privileged recontact component only; not blind control or quarter-turn success. Actual source equality is not an independent control comparison',full_action_pass=False,component_pass=bool(complete and safety and end_ok),frames=len(idx),original_deadline_reached=complete,source_step=row['source_step'],first_pad_contact_step=int(t[idx[hits[0]],1]) if len(hits) else None,pad_contact_frames=int(len(hits)),final_pad_force_N=float(pad[-1]),peak_pad_force_N=float(pad.max()),peak_rigid_index_force_N=float(rigid.max()),final_rigid_index_force_N=float(rigid[-1]),last10_contact_gate=good[-10:].tolist(),max_motor_error_rad=error,max_motor_rate_rad_s=rate,global_motion_support_pass=safety,final_cube_displacement_m=float(displacement[-1]),final_cube_rotation_deg=float(np.rad2deg(rotation[-1])),final_layer_disturbance_deg=float(disturbance[-1]),delivered_index_force_peak_N=float(np.linalg.norm(sens[:,94:97],axis=1).max()))


def first_turn_source(raw,trial,env,target_finished_step,stable_frames=10,entry_delay_frames=10):
    """First completed quiet-contact window; privileged diagnostic gate only."""
    if stable_frames!=10:raise ValueError('Turn entry requires original10frame window')
    if entry_delay_frames not in (1,10):raise ValueError('Unregistered contact entry delay')
    t=np.asarray(raw['trace'])[:,env];c=np.asarray(raw['contact'])[:,env];s=np.asarray(raw['sensory'])[:,env];core=np.asarray(raw['core_angular_speed'])[:,env]
    if not len(t):return None
    finite=np.isfinite(t).all(1)&np.isfinite(c).all((1,2,3))&np.isfinite(s).all(1)&np.isfinite(core)
    good=finite&(t[:,0]==trial)&(t[:,3]==3)&(t[:,4]>0)&(t[:,1]>=target_finished_step+entry_delay_frames)&(t[:,11]==0)
    good &= (c[:,1,0,10]>.05)&(c[:,1,0,10]<=5)&(c[:,:,0,6:10].sum((1,2))<=.5)
    good &= (np.abs(t[:,6])<.25)&(core<1)&(np.abs(s[:,49:53]).max(1)<=1)
    good &= (t[:,12]<=.015)&(t[:,13]<=np.deg2rad(15))&(t[:,14]<=.010)&(t[:,15]<=np.deg2rad(10))
    for hand in (0,1):good &= c[:,:,hand,:][:,:,[5,10,15,20,26]].max((1,2))>.05
    count=0
    for i,ok in enumerate(good):
        count=(count+1 if i and t[i,1]==t[i-1,1]+1 else 1) if ok else 0
        if count==stable_frames:return i
    return None


def solve_contact_turn(raw,trial,env,target_finished_step,normal_depth_reserve_m=0.,target_degrees=5.,entry_delay_frames=10):
    """Generate the declared-angle arc from the first eligible actual history."""
    if normal_depth_reserve_m not in (0.,.0001):raise ValueError('Unregistered turn depth reserve')
    if target_degrees not in (5.,10.):raise ValueError('Unregistered generated turn extent')
    import torch
    from rubik_lab.hand_kinematics import HandFK
    i=first_turn_source(raw,trial,env,target_finished_step,entry_delay_frames=entry_delay_frames)
    if i is None or i!=len(raw['trace'])-1:raise ValueError('Turn source must be latest first eligible contact window')
    for key in ('trace','sensory','poses','motor_targets','wrist_targets','contact','core_angular_speed'):
        if not np.isfinite(np.asarray(raw[key])[:,env]).all():raise ValueError('Nonfinite turn source')
    z=raw;e=env;pose=z['poses'][i,e]
    t=np.asarray(raw['trace'])[:,env]
    active=np.flatnonzero((t[:,0]==trial)&(t[:,3]==3)&(t[:,4]>0))
    if not np.array_equal(t[active,1],np.arange(t[active[0],1],t[i,1]+1)):raise ValueError('Missing fourth action history')
    handoff=int(t[active[0],1])-1
    remaining=6-(float(t[i,1])-handoff)/40
    rot=Rotation.from_quat(pose[3:7][[1,2,3,0]]);wp=rot.inv().apply(pose[7:10]-z['env_origins'][e]-pose[:3]);wr=rot.inv().as_matrix()@Rotation.from_quat(pose[10:14][[1,2,3,0]]).as_matrix();q=z['sensory'][i,e,:22].astype(float)
    fk=HandFK('left','cpu',pts_per_link=8,seed=0)
    for table in (fk.T0,fk.axis,fk.link_pts):
     for k in table:table[k]=table[k].double()
    tensor=lambda x:torch.as_tensor(x,dtype=torch.float64)
    def pad(v):
     a,b=fk.pads(fk.forward(tensor(wp)[None],tensor(wr)[None],tensor(v)[None]))['index'];return a[0].numpy(),b[0].numpy()
    
    initial_point,initial_normal=pad(q)
    cube_normal=cube_face_centerward_tangent(initial_point,initial_normal,np.deg2rad(-t[i,5]))[1] if normal_depth_reserve_m else np.zeros(3)
    bias=z['motor_targets'][i,e,0].astype(float)-q;lo=z['actuator_lower'][e,0,5:9]-bias[5:9];hi=z['actuator_upper'][e,0,5:9]-bias[5:9]
    def values(x):
     v=q.copy();v[5:9]=x;return pad(v)
    seed=q[5:9].copy();solutions=[];rows=[];duration=0.
    degrees=[.25,.5,.75,1.,2.,3.,4.,5.]+([6.,7.,8.,9.,10.] if target_degrees==10. else [])
    for waypoint,degree in enumerate(degrees):
     turn=Rotation.from_rotvec([0,0,np.deg2rad(degree)]);target=turn.apply(initial_point+normal_depth_reserve_m*min(degree,1.)*cube_normal);normal=turn.apply(initial_normal)
     def residual(x):
      p,n=values(x);return np.r_[(p-target)*1000,(n-normal)*5]
     fit=least_squares(residual,np.clip(seed,lo,hi),bounds=(lo,hi),max_nfev=200,xtol=1e-10,ftol=1e-10,gtol=1e-10)
     fit=minimize(lambda x:float(np.sum(((values(x)[0]-target)*1000)**2)),fit.x,method='SLSQP',bounds=list(zip(lo,hi)),constraints=[dict(type='ineq',fun=lambda x:float(values(x)[1]@normal)-.950001)],options=dict(maxiter=200,ftol=1e-12))
     point,n=values(fit.x);error=float(np.linalg.norm(point-target));dot=float(n@normal)
     if not fit.success or error>.001 or dot<.95:raise ValueError((degree,error,dot,fit.message))
     duration+=float(np.max(np.abs(fit.x-seed)));seed=fit.x.copy();v=np.r_[q,np.zeros(6)];v[5:9]=seed;solutions.append(v);rows.append(dict(trial=trial,env=env,waypoint=waypoint,position_error_m=error,normal_dot=dot))
    if not 0<duration<remaining:raise ValueError('Turn path exceeds original remaining deadline')
    motor=np.vstack([q,np.asarray(solutions)[:,:22]])+bias
    if np.any(motor<raw['actuator_lower'][env,0]-1e-12) or np.any(motor>raw['actuator_upper'][env,0]+1e-12):raise ValueError('Turn motor bounds')
    return dict(source_frame=i,source_step=int(t[i,1]),handoff_step=handoff,target_finished_step=target_finished_step,remaining_seconds=remaining,q=q,wrist_pos=wp,wrist_rot=wr,bias=bias,solution=np.asarray(solutions),degrees=degrees,results=rows,duration=duration)


def write_contact_turn(raw,prefix,trial,env,target_finished_step,normal_depth_reserve_m=0.,target_degrees=5.,entry_delay_frames=10):
    result=solve_contact_turn(raw,trial,env,target_finished_step,normal_depth_reserve_m,target_degrees,entry_delay_frames)
    prefix=Path(prefix);source=Path(str(prefix)+'_source');paths=Path(str(prefix)+'_paths')
    if any(p.with_suffix(ext).exists() for p in (prefix,source,paths) for ext in ('.json','.npz')):raise FileExistsError(prefix)
    prefix.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(source.with_suffix('.npz'),**raw)
    source.with_suffix('.json').write_text(json.dumps(dict(scope='Actual causal pre-turn history; no state or reference writes')))
    row=dict(trial=trial,env=env,source_frame=result['source_frame'],source_step=result['source_step'],move="U'",side='left',error_deg=float(raw['trace'][-1,env,5]))
    paths.with_suffix('.json').write_text(json.dumps(dict(source=str(source),results=[row])))
    np.savez_compressed(paths.with_suffix('.npz'),U_initial_q=result['q'][None],U_wrist_pos=result['wrist_pos'][None],U_wrist_rot=result['wrist_rot'][None])
    report=dict(entry_delay_frames=entry_delay_frames,normal_depth_reserve_m=normal_depth_reserve_m,normal_depth_ramp_degrees=1.,source=str(paths),phase='turn_geometry',sequential=True,whole_hand=True,variable_joint_indices=list(range(22)),support_contact=True,contact_transport=False,layer_angle_deltas_deg=result['degrees'],bias_rad=result['bias'].tolist(),duration_seconds=result['duration'],remaining_original_seconds=result['remaining_seconds'],handoff_step=result['handoff_step'],contact_target_finished_step=target_finished_step,scope='Privileged declared-angle contact arc; intended synchronized layer geometry only, no quarter-turn completion credit',results=result['results'])
    prefix.with_suffix('.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(prefix.with_suffix('.npz'),initial=np.r_[result['q'],np.zeros(6)][None],solution=result['solution'])
    return report


def contact_turn_degrees(report):
    base=[.25,.5,.75,1.,2.,3.,4.,5.]
    degrees=report.get('layer_angle_deltas_deg')
    if degrees!=base and degrees!=base+[6.,7.,8.,9.,10.] and degrees!=base+[10.,15.,20.,25.,30.]:raise ValueError('Unregistered contact-turn angle sequence')
    return degrees


class ContactTurnPath(LeftReturnPath):
    """Audited actual-history arc; uses motor interpolation only."""
    def __init__(self,prefix):
        from .live_whole_hand import LiveWholeHandProbe
        self.prefix=Path(prefix);self.started=False
        r=json.loads(self.prefix.with_suffix('.json').read_text())
        audit=json.loads(Path(str(prefix)+'_audit.json').read_text())
        degrees=contact_turn_degrees(r);count=len(degrees);self.target_degrees=degrees[-1];self.contact_target_finished_step=r['contact_target_finished_step']
        if r['phase']!='turn_geometry' or r['layer_angle_deltas_deg']!=degrees or audit['source']!=str(prefix) or audit['active_side']!='left' or not audit['cube_clearance']:raise ValueError('Invalid contact-turn audit scope')
        LiveWholeHandProbe.validate_audit(audit,support_contact=True,expected_end=count-1)
        for row in audit['results']:
            for item in [row]+row['segment_samples']:
                depth=item['cube_clearance']['active_pad_top_signed_overlap_mm']
                if not np.isfinite(depth) or not 0<=depth<=1:raise ValueError('Contact-turn pad depth outside0..1mm')
        source=json.loads(Path(r['source']).with_suffix('.json').read_text())
        if len(source['results'])!=1 or source['results'][0]['side']!='left' or source['results'][0]['move']!="U'":raise ValueError('Invalid turn source')
        self.source=source['results'][0];self.raw_prefix=Path(source['source'])
        with np.load(self.prefix.with_suffix('.npz')) as z:
            if z['initial'].shape!=(1,28) or z['solution'].shape!=(count,28):raise ValueError('Invalid turn path shape')
            physical=np.vstack([z['initial'],z['solution']])
        fixed=[j for j in range(28) if j not in range(5,9)]
        if not np.isfinite(physical).all() or not np.array_equal(physical[:,fixed],np.repeat(physical[:1,fixed],count+1,axis=0)):raise ValueError('Turn path changes support/wrist')
        self.motor=physical[:,:22]+np.asarray(r['bias_rad'])
        self.times=np.r_[0.,np.cumsum(np.max(np.abs(np.diff(self.motor[:,5:9],axis=0)),axis=1))]
        self.duration=float(self.times[-1])
        if np.any(np.diff(self.times)<=0):raise ValueError('Empty turn segment')
        with np.load(self.raw_prefix.with_suffix('.npz')) as z:
            i=self.source['source_frame'];env=self.source['env'];trial=self.source['trial']
            if i!=len(z['trace'])-1 or z['trace'][i,env,1]!=self.source['source_step'] or first_turn_source(z,trial,env,r['contact_target_finished_step'],entry_delay_frames=r.get('entry_delay_frames',10))!=i:raise ValueError('Turn source is not first completed eligible history')
            if not np.array_equal(physical[0,:22],z['sensory'][i,env,:22]) or not np.allclose(self.motor[0],z['motor_targets'][i,env,0],rtol=0,atol=1e-12):raise ValueError('Turn actuator source mismatch')
            if np.any(self.motor<z['actuator_lower'][env,0]-1e-12) or np.any(self.motor>z['actuator_upper'][env,0]+1e-12):raise ValueError('Turn actuator bounds')
            t=z['trace'][:,env];idx=np.flatnonzero((t[:,0]==trial)&(t[:,3]==3)&(t[:,4]>0))
            if not np.array_equal(t[idx,1],np.arange(t[idx[0],1],t[i,1]+1)):raise ValueError('Incomplete turn clock history')
            remaining=6-(t[i,1]-t[idx[0],1]+1)/40
            if not 0<self.duration<remaining or abs(remaining-r['remaining_original_seconds'])>1e-12:raise ValueError('Turn original deadline mismatch')
            self.original={k:z[k][:i+1,env].copy() for k in ('trace','sensory','poses','motor_targets','wrist_targets','contact','core_angular_speed')}


def summarize_contact_turn(report,raw,control=None,event_name='contact_turn_start'):
    """Measured declared-angle diagnostic outcome, never full-action acceptance."""
    if event_name not in ('contact_turn_start','second_contact_turn_start'):raise ValueError('Unknown turn event')
    events=[e for e in report.get('post_return_events',[]) if e['event']==event_name]
    if not events:return dict(attempted=False,full_action_pass=False,component_pass=False,reason='No audited causal turn started')
    if len(events)!=1:raise ValueError('Ambiguous contact turn')
    event=events[0];env=event['env'];trial=event['trial'];prefix=Path(event['source'])
    path=ContactTurnPath(prefix);r=json.loads(prefix.with_suffix('.json').read_text());target=contact_turn_degrees(r)[-1]
    i=path.source['source_frame'];t=np.asarray(raw['trace'])[:,env]
    if event['step']!=path.source['source_step']+1:raise ValueError('Noncausal turn start')
    for key,expected in path.original.items():
        actual=np.asarray(raw[key])[:i+1,env]
        if not np.array_equal(actual,expected):raise ValueError('Turn actual source mismatch: '+key)
        if control is not None and not np.array_equal(actual,np.asarray(control[key])[:i+1,env]):raise ValueError('Turn independent control prefix mismatch: '+key)
    idx=np.flatnonzero((t[:,0]==trial)&(t[:,3]==3)&(t[:,4]>0)&(t[:,1]>=event['step']))
    if not len(idx) or not np.array_equal(t[idx,1],np.arange(event['step'],event['step']+len(idx))):raise ValueError('Missing turn frames')
    with np.load(prefix.with_suffix('.npz')) as z:
        motor=np.vstack([z['initial'][:,:22],z['solution'][:,:22]])+np.asarray(r['bias_rad'])
        physical_reference=z['solution'][-1,5:9].copy()
    times=np.r_[0.,np.cumsum(np.max(np.abs(np.diff(motor[:,5:9],axis=0)),axis=1))]
    elapsed=(t[idx,1]-path.source['source_step'])/report['control_hz']
    expected=np.stack([np.interp(elapsed,times,motor[:,j]) for j in range(5,9)],axis=1)
    feedback=bool(report.get('second_stroke_feedback') and event_name=='second_contact_turn_start')
    gain=float(report.get('second_stroke_feedback_gain',1.))
    if gain not in (1.,3.):raise ValueError('Unregistered feedback gain')
    if feedback:
        lower=np.asarray(raw['actuator_lower'])[env,0,5:9];upper=np.asarray(raw['actuator_upper'])[env,0,5:9]
        for j,frame in enumerate(idx):
            if elapsed[j]>=times[-1]+.25:
                measured=np.asarray(raw['sensory'])[frame-1,env,5:9]
                previous=np.asarray(raw['motor_targets'])[frame-1,env,0,5:9]
                desired=motor[-1,5:9]+np.clip(gain*(physical_reference-measured),-.1,.1)
                expected[j]=np.minimum(np.maximum(desired,np.maximum(lower,previous-1/report['control_hz'])),np.minimum(upper,previous+1/report['control_hz']))
    actual=np.asarray(raw['motor_targets'])[idx,env,0,5:9]
    error=float(np.max(np.abs(actual-expected)));rate=float(np.max(np.abs(np.diff(np.vstack([motor[0,5:9],actual]),axis=0)))*report['control_hz'])
    if not np.isfinite(actual).all() or error>1e-6 or rate>1.00001:raise ValueError('Turn motor delivery/rate')
    c=np.asarray(raw['contact'])[idx,env];s=np.asarray(raw['sensory'])[idx,env];core=np.asarray(raw['core_angular_speed'])[idx,env]
    if not all(np.isfinite(x).all() for x in (t[idx],c,s,core)):raise ValueError('Nonfinite turn measurements')
    pad=c[:,1,0,10];rigid=c[:,:,0,6:10].sum((1,2))
    support=np.stack([c[:,:,h,:][:,:,[5,10,15,20,26]].max((1,2))>.05 for h in (0,1)],axis=1).all(1)
    progress=t[i,5]-t[idx,5]
    good=(np.abs(progress-target)<=1)&(np.abs(t[idx,6])<.25)&(core<1)&(np.abs(s[:,49:53]).max(1)<=1)&(pad>.05)&(pad<=5)&(rigid<=.5)&support&(t[idx,11]==0)
    safety=bool(support.all() and (t[idx,11]==0).all() and (t[idx,12]<=.015).all() and (t[idx,13]<=np.deg2rad(15)).all() and (t[idx,14]<=.010).all() and (t[idx,15]<=np.deg2rad(10)).all())
    original_elapsed=float((t[idx[-1],1]-r['handoff_step'])/report['control_hz'])
    complete=abs(original_elapsed-6)<1e-5
    return dict(attempted=True,proprioceptive_hold_feedback=feedback,target_degrees=target,full_action_pass=False,component_pass=bool(complete and safety and len(idx)>=10 and good[-10:].all()),source_step=path.source['source_step'],frames=len(idx),original_elapsed_seconds=original_elapsed,original_deadline_reached=complete,prefix_exact_match=control is not None,max_motor_error_rad=error,max_motor_rate_rad_s=rate,final_progress_deg=float(progress[-1]),progress_range_deg=[float(progress.min()),float(progress.max())],final_full_target_error_deg=float(t[idx[-1],5]),final_pad_force_N=float(pad[-1]),peak_rigid_index_force_N=float(rigid.max()),contact_below_threshold_frames=int((pad<=.05).sum()),last10_turn_gate=good[-10:].tolist(),global_motion_support_pass=safety,scope='One privileged contact-to-turn component; original fullquartertarget remains unchanged. Not full action, blind policy, independent quota or real-hardware acceptance.')
