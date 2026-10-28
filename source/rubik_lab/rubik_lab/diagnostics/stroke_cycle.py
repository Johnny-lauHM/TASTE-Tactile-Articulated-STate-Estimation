"""Privileged internal-stroke release diagnostic; never an atomic completion gate."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .left_return import LeftReturnPath

HISTORY=('trace','sensory','poses','motor_targets','wrist_targets','contact','core_angular_speed')

def first_stroke_lift_source(raw,trial,env,target_finished_step,drive_boundary=None):
    t=np.asarray(raw['trace'])[:,env];c=np.asarray(raw['contact'])[:,env];s=np.asarray(raw['sensory'])[:,env];core=np.asarray(raw['core_angular_speed'])[:,env]
    good=np.isfinite(t).all(1)&np.isfinite(c).all((1,2,3))&np.isfinite(s).all(1)&np.isfinite(core)
    good &= (t[:,0]==trial)&(t[:,3]==3)&(t[:,4]>0)&(t[:,1]>target_finished_step)&(t[:,11]==0)
    good &= (np.abs(t[:,6])<.25)&(np.abs(core)<1)&(np.abs(s[:,49:53]).max(1)<=1)
    good &= (c[:,1,0,10]>.05)&(c[:,1,0,10]<=5)&(c[:,:,0,6:10].sum((1,2))<=.5)
    good &= (t[:,12]<=.015)&(t[:,13]<=np.deg2rad(15))&(t[:,14]<=.010)&(t[:,15]<=np.deg2rad(10))
    for h in (0,1):good &= c[:,:,h,:][:,:,[5,10,15,20,26]].max((1,2))>.05
    if drive_boundary is not None:
        # Privileged progress gate for diagnostics only; original target is untouched.
        if set(drive_boundary)!={'source_step','feedback_start_step','target_degrees'}:
            raise ValueError('Invalid completed-drive boundary fields')
        source_step=drive_boundary['source_step'];start=drive_boundary['feedback_start_step'];degrees=drive_boundary['target_degrees']
        if not all(np.isfinite(x) for x in (source_step,start,degrees)) or source_step!=int(source_step) or start!=int(start) or not source_step<target_finished_step<start or degrees not in (5.,10.):
            raise ValueError('Invalid completed-drive boundary timing/target')
        source=np.flatnonzero((t[:,0]==trial)&(t[:,3]==3)&(t[:,4]>0)&(t[:,1]==source_step))
        if len(source)!=1 or not np.isfinite(t[source[0],5]):
            raise ValueError('Missing unique completed-drive source')
        progress=t[source[0],5]-t[:,5]
        good &= (t[:,1]>=start)&(np.abs(progress-degrees)<=1.)
    count=0
    for i,ok in enumerate(good):
        count=(count+1 if i and t[i,1]==t[i-1,1]+1 else 1) if ok else 0
        if count==4:return i
    return None

def first_stroke_recontact_source(raw,trial,env,target_finished_step):
    t=np.asarray(raw['trace'])[:,env];c=np.asarray(raw['contact'])[:,env];s=np.asarray(raw['sensory'])[:,env];core=np.asarray(raw['core_angular_speed'])[:,env]
    good=np.isfinite(t).all(1)&np.isfinite(c).all((1,2,3))&np.isfinite(s).all(1)&np.isfinite(core)
    good &= (t[:,0]==trial)&(t[:,3]==3)&(t[:,4]>0)&(t[:,1]>target_finished_step)&(t[:,11]==0)
    good &= (np.abs(t[:,6])<.25)&(np.abs(core)<1)&(np.abs(s[:,49:53]).max(1)<=1)
    good &= (c[:,:,0,10].sum(1)<=.005)&(c[:,:,0,6:11].sum((1,2))<=.5)
    good &= (t[:,12]<=.015)&(t[:,13]<=np.deg2rad(15))&(t[:,14]<=.010)&(t[:,15]<=np.deg2rad(10))
    for h in (0,1):good &= c[:,:,h,:][:,:,[5,10,15,20,26]].max((1,2))>.05
    count=0
    for i,ok in enumerate(good):
        count=(count+1 if i and t[i,1]==t[i-1,1]+1 else 1) if ok else 0
        if count==4:return i
    return None

class _InternalStrokePath(LeftReturnPath):
    def __init__(self,prefix,target_finished_step,recontact=False):
        from .live_whole_hand import LiveWholeHandProbe
        self.prefix=Path(prefix);r=json.loads(self.prefix.with_suffix('.json').read_text())
        a=json.loads(Path(str(self.prefix)+'_audit.json').read_text())
        phase='recontact_geometry' if recontact else 'withdrawal_intermediate_stroke'
        count=2 if recontact else 1
        if recontact and (not r.get('internal_stroke_recontact') or r['lift_targets_finished_step']!=target_finished_step):raise ValueError('Internal recontact timing/scope')
        if r['phase']!=phase or not r['sequential'] or not r['support_contact'] or r['contact_transport']:raise ValueError('Internal lift scope')
        if a['source']!=str(prefix) or a['active_side']!='left' or not a['cube_clearance']:raise ValueError('Internal lift audit scope')
        LiveWholeHandProbe.validate_audit(a,support_contact=True,expected_end=count-1)
        if a['results'][0]['cube_clearance']['active_pad_top_signed_overlap_mm']>=0:raise ValueError('Outside endpoint has no pad gap')
        if recontact:
            if not 0<=a['results'][-1]['cube_clearance']['active_pad_top_signed_overlap_mm']<=1:raise ValueError('Recontact endpoint pad depth')
            for row in a['results']:
                for sample in [row,*row['segment_samples']]:
                    if sample['cube_clearance']['active_pad_top_signed_overlap_mm']>1:raise ValueError('Recontact interior pad depth')
        source=json.loads(Path(r['source']).with_suffix('.json').read_text())
        if len(source['results'])!=1:raise ValueError('Unique lift source required')
        self.source=source['results'][0];self.raw_prefix=Path(source['source']);self.started=False
        if self.source['side']!='left' or self.source['move']!="U'":raise ValueError('Unsupported lift side/move')
        with np.load(self.prefix.with_suffix('.npz')) as z:initial=z['initial'][0];solution=z['solution']
        fixed=[j for j in range(28) if j not in range(5,9)]
        if initial.shape!=(28,) or solution.shape!=(count,28) or not np.isfinite(initial).all() or not np.isfinite(solution).all() or not np.array_equal(solution[:,fixed],np.broadcast_to(initial[fixed],(count,len(fixed)))):raise ValueError('Lift changes support/wrist or nonfinite path')
        self.physical_reference=solution[-1,5:9].copy()
        bias=np.asarray(r['bias_rad']);self.motor=np.vstack([initial[:22],solution[:,:22]])+bias
        self.times=np.r_[0.,np.cumsum(np.max(np.abs(np.diff(self.motor[:,5:9],axis=0)),axis=1))];self.duration=float(self.times[-1])
        if np.any(np.diff(self.times)<=0):raise ValueError('Zero internal segment')
        with np.load(self.raw_prefix.with_suffix('.npz')) as z:
            i=self.source['source_frame'];env=self.source['env'];trial=self.source['trial'];t=z['trace'][:,env]
            if i!=len(t)-1 or t[i,1]!=self.source['source_step'] or (first_stroke_recontact_source(z,trial,env,target_finished_step) if recontact else first_stroke_lift_source(z,trial,env,target_finished_step,r.get('drive_boundary')))!=i:raise ValueError('Lift requires first eligible latest actual history')
            active=np.flatnonzero((t[:,0]==trial)&(t[:,3]==3)&(t[:,4]>0))
            if not len(active) or not np.array_equal(t[active,1],np.arange(t[active[0],1],t[i,1]+1)):raise ValueError('Original lift clock has gaps')
            self.handoff_step=int(t[active[0],1])-1
            remaining=6-(t[i,1]-self.handoff_step)*.025
            if not 0<self.duration<remaining or abs(remaining-r['remaining_original_seconds'])>1e-9:raise ValueError('Lift original deadline')
            if not np.array_equal(initial[:22],z['sensory'][i,env,:22]) or not np.allclose(self.motor[0],z['motor_targets'][i,env,0],rtol=0,atol=1e-12):raise ValueError('Lift source motor mismatch')
            if not np.isfinite(self.motor).all() or np.any(self.motor<z['actuator_lower'][env,0]-1e-12) or np.any(self.motor>z['actuator_upper'][env,0]+1e-12):raise ValueError('Lift motor bounds')
            self.original={k:z[k][:i+1,env].copy() for k in HISTORY}
            if not all(np.isfinite(v).all() for v in self.original.values()):raise ValueError('Nonfinite lift history')

class StrokeLiftPath(_InternalStrokePath):
    pass

class StrokeRecontactPath(_InternalStrokePath):
    def __init__(self,prefix,target_finished_step):super().__init__(prefix,target_finished_step,recontact=True)

def _summarize_internal(report,raw,recontact=False,event_name=None):
    event_name=event_name or ('stroke_recontact_start' if recontact else 'stroke_lift_start')
    starts=[x for x in report.get('post_return_events',[]) if x['event']==event_name]
    if not starts:return dict(attempted=False,component_pass=False,full_action_pass=False)
    if len(starts)!=1:raise ValueError('Unique lift event required')
    event=starts[0];path=(StrokeRecontactPath if recontact else StrokeLiftPath)(event['source'],event['lift_targets_finished_step'] if recontact else event['turn_targets_finished_step']);env=event['env'];trial=event['trial'];i=path.source['source_frame']
    if env!=path.source['env'] or trial!=path.source['trial'] or event['step']!=path.source['source_step']+1 or report['control_hz']!=40:raise ValueError('Lift event causality')
    path.verify_history({k:np.asarray(raw[k])[:i+1] for k in HISTORY})
    t=np.asarray(raw['trace'])[:,env];idx=np.flatnonzero((t[:,0]==trial)&(t[:,3]==3)&(t[:,4]>0)&(t[:,1]>=event['step']))
    if not len(idx) or not np.array_equal(t[idx,1],np.arange(event['step'],event['step']+len(idx))):raise ValueError('Missing lift frames')
    elapsed=(t[idx,1]-path.source['source_step'])/40
    times=path.times if recontact else np.array([0.,path.duration])
    expected=np.stack([np.interp(elapsed,times,path.motor[:,j]) for j in range(5,9)],axis=1)
    feedback_key='completed_recontact_feedback' if recontact else 'completed_lift_feedback'
    if event_name.startswith('cycle_') and report.get(feedback_key):
        from .whole_hand_path import bounded_index_feedback
        for j,k in enumerate(idx):
            if elapsed[j]>=path.duration+.25:
                expected[j]=bounded_index_feedback(path.physical_reference,np.asarray(raw['sensory'])[k-1,env,5:9],path.motor[-1,5:9],np.asarray(raw['motor_targets'])[k-1,env,0,5:9],raw['actuator_lower'][env,0,5:9],raw['actuator_upper'][env,0,5:9],.025,gain=3.)[0]
    actual=np.asarray(raw['motor_targets'])[idx,env,0,5:9]
    error=float(np.max(np.abs(actual-expected)));rate=float(np.max(np.abs(np.diff(np.vstack([path.motor[0,5:9],actual]),axis=0)))*40)
    if not np.isfinite(actual).all() or error>1e-6 or rate>1.00001:raise ValueError('Lift motor delivery')
    c=np.asarray(raw['contact'])[idx,env];s=np.asarray(raw['sensory'])[idx,env];p=np.asarray(raw['poses'])[idx,env];core=np.asarray(raw['core_angular_speed'])[idx,env]
    if not all(np.isfinite(x).all() for x in (t[idx],c,s,p,core)):raise ValueError('Nonfinite lift measurements')
    release=c[:,:,0,6:11].sum((1,2));pad=c[:,:,0,10].sum(1)
    support=np.ones(len(idx),dtype=bool)
    for h in (0,1):support &= c[:,:,h,:][:,:,[5,10,15,20,26]].max((1,2))>.05
    ref=np.asarray(raw['poses'])[i,env];dist=np.linalg.norm(p[:,:3]-ref[:3],axis=1)
    rot=(Rotation.from_quat(p[:,3:7][:,[1,2,3,0]])*Rotation.from_quat(ref[3:7][[1,2,3,0]]).inv()).magnitude()
    drift=np.abs(t[idx,5]-t[i,5])
    safety=support&(t[idx,11]==0)&(t[idx,12]<=.015)&(t[idx,13]<=np.deg2rad(15))&(t[idx,14]<=.010)&(t[idx,15]<=np.deg2rad(10))
    force_gate=(c[:,1,0,10]>.05)&(c[:,1,0,10]<=5)&(c[:,:,0,6:10].sum((1,2))<=.5) if recontact else (release<=.5)&(pad<=.005)
    good=safety&force_gate&(dist<=.002)&(rot<=np.deg2rad(2))&(drift<=1)&(np.abs(t[idx,6])<.25)&(np.abs(core)<1)&(np.abs(s[:,49:53]).max(1)<=1)
    complete=abs((t[idx[-1],1]-path.handoff_step)/40-6)<1e-9
    passed=complete and len(idx)>=10 and good[-10:].all() and safety.all() and (dist<=.002).all() and (rot<=np.deg2rad(2)).all() and (drift<=1).all()
    return dict(scope=('Privileged internal recontact only' if recontact else 'Privileged intermediate lift only')+'; no ready-return, quarter-turn or blind-policy credit',attempted=True,component_pass=bool(passed),full_action_pass=False,source_step=path.source['source_step'],frames=len(idx),original_deadline_reached=bool(complete),final_original_angle_error_deg=float(t[idx[-1],5]),final_index_allbody_force_N=float(release[-1]),final_pad_allbody_force_N=float(pad[-1]),max_layer_drift_deg=float(drift.max()),max_cube_displacement_m=float(dist.max()),max_cube_rotation_deg=float(np.rad2deg(rot.max())),global_motion_support_pass=bool(safety.all()),last10_force_gate=good[-10:].tolist(),max_motor_error_rad=error,max_motor_rate_rad_s=rate)

def summarize_stroke_lift(report,raw):return _summarize_internal(report,raw)

def summarize_stroke_recontact(report,raw):return _summarize_internal(report,raw,recontact=True)

def write_internal_candidate(raw,prefix,trial,env,target_finished_step,prior_contact_step=None,drive_boundary=None,recontact_outside_m=.0003):
    """Generate from the latest eligible actual history, never a restored state.

    The same two preregistered geometries are used for all sources: lift2mm;
    or reset2mm outside then press with FK point .3mm outside the face plane.
    Cooked-pad geometry must be independently audited before execution.
    """
    import torch
    from scipy.optimize import least_squares,minimize
    from rubik_lab.hand_kinematics import HandFK
    from .left_recontact import cube_face_centerward_tangent
    recontact=prior_contact_step is not None
    if recontact_outside_m not in (.0003,.0005) or (not recontact and recontact_outside_m!=.0003):raise ValueError('Unregistered recontact outside offset')
    gate=first_stroke_recontact_source if recontact else first_stroke_lift_source
    if recontact and drive_boundary is not None:raise ValueError("Drive boundary applies only to lift")
    i=gate(raw,trial,env,target_finished_step) if recontact else gate(raw,trial,env,target_finished_step,drive_boundary)
    t=np.asarray(raw['trace'])[:,env];s=np.asarray(raw['sensory'])[:,env]
    if i is None or i!=len(t)-1:raise ValueError('Internal candidate requires latest firsteligible actual source')
    keys=(*HISTORY,'ready_targets','env_origins','actuator_lower','actuator_upper')
    if not all(np.isfinite(np.asarray(raw[k])).all() for k in keys):raise ValueError('Nonfinite internal planning input')
    fk=HandFK('left','cpu',pts_per_link=8,seed=0)
    for table in (fk.T0,fk.axis,fk.link_pts):
        for k in table:table[k]=table[k].double()
    ts=lambda a:torch.as_tensor(a,dtype=torch.float64)[None]
    def state(j):
        q=s[j,:22].astype(float);pose=np.asarray(raw['poses'])[j,env]
        rot=Rotation.from_quat(pose[3:7][[1,2,3,0]])
        wp=rot.inv().apply(pose[7:10]-raw['env_origins'][env]-pose[:3])
        wr=rot.inv().as_matrix()@Rotation.from_quat(pose[10:14][[1,2,3,0]]).as_matrix()
        p,n=fk.pads(fk.forward(ts(wp),ts(wr),ts(q)))['index']
        return q,wp,wr,p[0].numpy(),n[0].numpy()
    q,wp,wr,p0,n0=state(i);bias=np.asarray(raw['motor_targets'])[i,env,0]-q
    if recontact:
        previous=np.flatnonzero((t[:,0]==trial)&(t[:,3]==3)&(t[:,1]==prior_contact_step))
        if len(previous)!=1 or previous[0]>=i:raise ValueError('Invalid previous actual contact source')
        j=previous[0]
        if np.asarray(raw['contact'])[j,env,1,0,10]<=.05:raise ValueError('Previous source lacks top contact')
        _,_,_,prior,prior_normal=state(j)
        _,normal=cube_face_centerward_tangent(prior,prior_normal,np.deg2rad(-t[i,5]))
        projection=prior+(-.035-float(prior@normal))*normal
        targets=[projection-.002*normal,projection-recontact_outside_m*normal];desired_normal=normal
    else:
        _,normal=cube_face_centerward_tangent(p0,n0,np.deg2rad(-t[i,5]))
        targets=[p0-.002*normal];desired_normal=n0
    lo=np.asarray(raw['actuator_lower'])[env,0,5:9]-bias[5:9]
    hi=np.asarray(raw['actuator_upper'])[env,0,5:9]-bias[5:9]
    def pad(x):
        v=q.copy();v[5:9]=x;p,n=fk.pads(fk.forward(ts(wp),ts(wr),ts(v)))['index']
        return p[0].numpy(),n[0].numpy()
    seed=q[5:9].copy();rows=[];solutions=[]
    for k,target in enumerate(targets):
        fit=least_squares(lambda x:np.r_[(pad(x)[0]-target)*1000,(pad(x)[1]-desired_normal)*5],np.clip(seed,lo,hi),bounds=(lo,hi),max_nfev=200)
        fit=minimize(lambda x:float(np.sum(((pad(x)[0]-target)*1000)**2)),fit.x,method='SLSQP',bounds=list(zip(lo,hi)),constraints=[dict(type='ineq',fun=lambda x:float(pad(x)[1]@desired_normal)-.950001)],options=dict(maxiter=200,ftol=1e-12))
        p,n=pad(fit.x);err=float(np.linalg.norm(p-target));dot=float(n@desired_normal)
        if not np.isfinite(fit.x).all() or err>.001 or dot<.95:raise ValueError('Internal path pose fit rejected: '+str((k,err,dot)))
        solution=np.r_[q,np.zeros(6)];solution[5:9]=fit.x;solutions.append(solution);seed=fit.x
        rows.append(dict(trial=trial,env=env,waypoint=k,position_error_m=err,normal_dot=dot,solver_success=bool(fit.success)))
    initial=np.r_[q,np.zeros(6)];solutions=np.asarray(solutions)
    motors=np.vstack([initial[:22],solutions[:,:22]])+bias
    if np.any(motors<np.asarray(raw['actuator_lower'])[env,0]-1e-12) or np.any(motors>np.asarray(raw['actuator_upper'])[env,0]+1e-12):raise ValueError('Internal planner motor bounds')
    duration=float(np.max(np.abs(np.diff(motors[:,5:9],axis=0)),axis=1).sum())
    active=np.flatnonzero((t[:,0]==trial)&(t[:,3]==3)&(t[:,4]>0))
    if not len(active) or not np.array_equal(t[active,1],np.arange(t[active[0],1],t[i,1]+1)):raise ValueError('Internal planning original clock gaps')
    remaining=6-(t[i,1]-(t[active[0],1]-1))*.025
    if not 0<duration<remaining:raise ValueError('Internal path misses original deadline')
    prefix=Path(prefix);source=Path(str(prefix)+'_source');paths=Path(str(prefix)+'_paths')
    if any(p.with_suffix(ext).exists() for p in (prefix,source,paths) for ext in ('.json','.npz')):raise FileExistsError(prefix)
    prefix.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(source.with_suffix('.npz'),**{k:np.asarray(raw[k]) for k in keys})
    source.with_suffix('.json').write_text(json.dumps(dict(scope='Actual completed internal source; no physical replay or target/clock reset')))
    paths.with_suffix('.json').write_text(json.dumps(dict(source=str(source),results=[dict(trial=trial,env=env,source_frame=i,source_step=int(t[i,1]),side='left',move="U'",error_deg=float(t[i,5]))])))
    np.savez_compressed(paths.with_suffix('.npz'),U_initial_q=q[None],U_wrist_pos=wp[None],U_wrist_rot=wr[None])
    report=dict(source=str(paths),phase='recontact_geometry' if recontact else 'withdrawal_intermediate_stroke',sequential=True,whole_hand=True,variable_joint_indices=list(range(22)),support_contact=True,contact_transport=False,bias_rad=bias.tolist(),duration_seconds=duration,remaining_original_seconds=remaining,internal_stroke_recontact=recontact,lift_targets_finished_step=target_finished_step if recontact else None,prior_contact_source_step=prior_contact_step,scope='Privileged actual-source internal path; independent geometry audit and dynamic scoring required; no atomic completion credit',results=rows)
    if recontact:report['recontact_outside_m']=recontact_outside_m
    if drive_boundary is not None:report['drive_boundary']=dict(drive_boundary)
    prefix.with_suffix('.json').write_text(json.dumps(report,indent=2));np.savez_compressed(prefix.with_suffix('.npz'),initial=initial[None],solution=solutions)
    return report


class ReleaseRecontactCycle:
    """Reusable privileged release/recontact phases after a completed drive.

    The caller supplies audited actual-source planning and continues its drive
    command while this machine returns None. No environment state is written.
    """
    def __init__(self,drive_source_step,drive_finished_step,feedback_start_step,target_degrees,lift_feedback=False,recontact_feedback=False):
        self.boundary=dict(source_step=drive_source_step,feedback_start_step=feedback_start_step,target_degrees=target_degrees)
        self.drive_finished_step=drive_finished_step
        self.phase='await_drive';self.path=None;self.finished_step=None
        self.lift_feedback=lift_feedback;self.recontact_feedback=recontact_feedback;self.feedback_started=False

    def command(self,raw,trial,env,step,dt,plan,events):
        if abs(dt-.025)>1e-12 or step!=int(raw['trace'][-1,env,1])+1:
            raise ValueError('Cycle requires previous completed frame at original 40Hz')
        if self.phase=='await_drive':
            i=first_stroke_lift_source(raw,trial,env,self.drive_finished_step,self.boundary)
            if i is None:return None
            if i!=len(raw['trace'])-1:raise ValueError('Missed completed-drive lift boundary')
            self.path=plan('lift',self.drive_finished_step,None,self.boundary)
            self._start(raw,trial,env,step,events,'lift')
        elif self.phase=='lift' and self.finished_step is not None:
            i=first_stroke_recontact_source(raw,trial,env,self.finished_step)
            if i is not None:
                if i!=len(raw['trace'])-1:raise ValueError('Missed released recontact boundary')
                self.path=plan('recontact',self.finished_step,self.boundary['source_step'],None)
                self._start(raw,trial,env,step,events,'recontact')
        elapsed=(step-self.path.source['source_step'])*dt
        command=self.path.sample(elapsed)
        if ((self.lift_feedback and self.phase=='lift') or (self.recontact_feedback and self.phase=='recontact')) and elapsed>=self.path.duration+.25:
            from .whole_hand_path import bounded_index_feedback
            command,_,_,_=bounded_index_feedback(self.path.physical_reference,raw['sensory'][-1,env,5:9],self.path.motor[-1,5:9],raw['motor_targets'][-1,env,0,5:9],raw['actuator_lower'][env,0,5:9],raw['actuator_upper'][env,0,5:9],dt,gain=3.)
            if not self.feedback_started:
                self.feedback_started=True
                events.append(dict(event='cycle_'+self.phase+'_feedback_start',trial=trial,env=env,step=step,gain=3.,scope='Previous proprioception only; fixed physical reference, cap.1rad,1rad/s, original bounds'))
        if self.finished_step is None and elapsed>=self.path.duration:
            self.finished_step=step
            events.append(dict(event='cycle_'+self.phase+'_targets_finished',trial=trial,env=env,step=step))
        return command

    def _start(self,raw,trial,env,step,events,phase):
        source=self.path.source
        if source['trial']!=trial or source['env']!=env or source['source_step']!=step-1:
            raise ValueError('Cycle path actual-source identity')
        self.path.verify_history(raw)
        previous_finished=self.finished_step
        self.phase=phase;self.finished_step=None;self.feedback_started=False
        events.append(dict(event='cycle_'+phase+'_start',trial=trial,env=env,step=step,
                           source=str(self.path.prefix),source_step=source['source_step'],
                           duration_seconds=self.path.duration,drive_boundary=dict(self.boundary),
                           turn_targets_finished_step=self.drive_finished_step,lift_targets_finished_step=previous_finished,
                           scope='Privileged internal phase; original target and clock retained; no atomic credit'))
