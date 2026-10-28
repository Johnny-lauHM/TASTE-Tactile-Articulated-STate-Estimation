"""Offline-audited left withdrawal replay; privileged diagnostic, never state reset."""
import json
from pathlib import Path
import numpy as np

class LeftReturnPath:
    def __init__(self,prefix):
        self.prefix=Path(prefix)
        r=json.loads(self.prefix.with_suffix('.json').read_text())
        audit=json.loads(Path(str(prefix)+'_audit.json').read_text())
        source=json.loads(Path(r['source']).with_suffix('.json').read_text())
        if len(source['results'])!=1 or source['results'][0]['side']!='left':raise ValueError('Expected one left source')
        self.source=source['results'][0];self.raw_prefix=Path(source['source'])
        if audit['source']!=str(prefix) or audit['active_side']!='left' or audit['segment_samples']<19 or not audit['cube_clearance'] or not audit['support_contact_audited']:raise ValueError('Incomplete independent path audit')
        with np.load(self.prefix.with_suffix('.npz')) as z:
            initial=z['initial'][0];solution=z['solution'].copy()
        if len(audit['results'])!=len(solution) or [v['waypoint'] for v in audit['results']]!=list(range(len(solution))):raise ValueError('Incomplete audit waypoints')
        for row in audit['results']:
            for item in [dict(row,collisions=row['candidate_collisions'])]+row['segment_samples']:
                if not item['cube_clearance']['pass_at_0p1mm'] or not item['support_contact']['passed']:raise ValueError('Rejected cube/support geometry')
                for c in item['collisions']:
                    if c['rigid_pair'] and (c['sat_signed_overlap_m'] is None or not np.isfinite(c['sat_signed_overlap_m']) or c['sat_signed_overlap_m']>.0001):raise ValueError('Rejected hand geometry')
        self.physical=np.vstack([initial,solution])
        if self.physical.shape[1]!=28 or not np.isfinite(self.physical).all():raise ValueError('Invalid path')
        fixed=[i for i in range(28) if i not in range(5,9)]
        if not np.all(self.physical[:,fixed]==initial[fixed]):raise ValueError('Path changes wrist/support')
        self.motor=self.physical[:,:22]+np.asarray(r['bias_rad'])
        self.times=np.r_[0.,np.cumsum(np.max(np.abs(np.diff(self.motor[:,5:9],axis=0)),axis=1))]
        if np.any(np.diff(self.times)<=0):raise ValueError('Empty segment')
        self.duration=float(self.times[-1]);self.started=False
        with np.load(self.raw_prefix.with_suffix('.npz')) as z:
            i=self.source['source_frame'];env=self.source['env'];trial=self.source['trial']
            self.original={k:z[k][:i+1,env].copy() for k in ('trace','sensory','poses','motor_targets','wrist_targets','contact','core_angular_speed')}
            eligible=first_withdrawal_source(z,trial,env,self.source['source_step']-3)
            if eligible!=i:raise ValueError('Audited source is not arrived, quiet and released for four frames')
            lo=z['actuator_lower'][env,0];hi=z['actuator_upper'][env,0]
            if np.any(self.motor<lo-1e-7) or np.any(self.motor>hi+1e-7):raise ValueError('Motor bounds')
            if np.max(np.abs(self.motor[-1,5:9]-z['ready_targets'][trial,env,0,5:9]))>1e-7:raise ValueError('Motor endpoint is not original ready')
        if self.duration>=r['remaining_original_seconds']:raise ValueError('Original deadline exceeded')
    def verify_history(self,history):
        env=self.source['env']
        for key,expected in self.original.items():
            actual=np.asarray(history[key])[:,env]
            if actual.shape!=expected.shape or not np.array_equal(actual,expected):raise ValueError('Left return prefix differs: '+key)
        self.started=True
    def sample(self,elapsed):
        if elapsed<0:raise ValueError('Negative path time')
        t=min(float(elapsed),self.duration);j=min(int(np.searchsorted(self.times,t,side='right')-1),len(self.times)-2)
        f=(t-self.times[j])/(self.times[j+1]-self.times[j])
        return self.motor[j,5:9]+f*(self.motor[j+1,5:9]-self.motor[j,5:9])

def _first_left_source(raw,trial,env,min_step,stable_frames=4,release_required=True):
    """Privileged diagnostic gate; never substitutes for the blind completion detector.

    min_step must include the completed motor relaxation ramp. Only completed
    control frames are eligible; the caller retains original primitive time.
    """
    if stable_frames<1:raise ValueError('Invalid stability window')
    t=np.asarray(raw['trace'])[:,env];q=np.asarray(raw['sensory'])[:,env]
    c=np.asarray(raw['contact'])[:,env];core=np.asarray(raw['core_angular_speed'])[:,env]
    total=c[:,:,0,6:11].sum(axis=(1,2));speed=np.max(np.abs(q[:,49:53]),axis=1)
    supporting=np.stack([c[:,:,h,:][:,:,[5,10,15,20,26]].max(axis=(1,2))>.05 for h in range(2)],axis=1).all(axis=1)
    finite=np.isfinite(t).all(axis=1)&np.isfinite(q).all(axis=1)&np.isfinite(c).all(axis=(1,2,3))&np.isfinite(core)
    good=finite&(t[:,0]==trial)&(t[:,1]>=min_step)&(t[:,3]==2)&(t[:,4]>0)&(t[:,11]==0)&(np.abs(t[:,5])<5)&(np.abs(t[:,6])<.25)&(core<1)&(speed<=1)&supporting
    if release_required:good &= total<=.5
    count=0;previous=None
    for i,ok in enumerate(good):
        consecutive=previous is not None and t[i,1]==previous+1
        count=(count+1 if consecutive else 1) if ok else 0
        previous=t[i,1]
        if count>=stable_frames:return i
    return None

def first_withdrawal_source(raw,trial,env,min_step,stable_frames=4):
    """Strict full-withdrawal entry; includes all-body release, never relaxed."""
    return _first_left_source(raw,trial,env,min_step,stable_frames,release_required=True)


def next_left_path_kind(raw,trial,env,min_step,residual_attempted=False):
    """Truth-only diagnostic staging, not a blind completion decision.

    Four latest completed frames must be quiet, supported and inside angle
    tolerance. Residual contact may permit one small escape; it never permits
    full withdrawal until the unchanged all-body release gate also passes.
    """
    keys=('trace','sensory','contact','core_angular_speed')
    if len(raw['trace'])<4:return None
    recent={k:np.asarray(raw[k])[-4:] for k in keys}
    if _first_left_source(recent,trial,env,min_step,release_required=True)==3:return 'full'
    if not residual_attempted and _first_left_source(recent,trial,env,min_step,release_required=False)==3:return 'residual'
    return None


class LeftResidualEscapePath(LeftReturnPath):
    """Small audited physical displacement mapped to motor increments, diagnostic only.

    Unlike full withdrawal, this starts under residual contact and cannot credit
    completion. The exact recorded history is mandatory; no state restoration.
    """
    def __init__(self,prefix):
        self.prefix=Path(prefix)
        r=json.loads(self.prefix.with_suffix('.json').read_text())
        audit=json.loads(Path(str(prefix)+'_audit.json').read_text())
        source=json.loads(Path(r['source']).with_suffix('.json').read_text())
        if len(source['results'])!=1 or source['results'][0]['side']!='left':raise ValueError('Expected one left source')
        self.source=source['results'][0];self.raw_prefix=Path(source['source']);self.started=False
        if audit['source']!=str(prefix) or audit['active_side']!='left' or audit['segment_samples']<19 or not audit['cube_clearance'] or not audit['support_contact_audited']:raise ValueError('Incomplete escape audit')
        with np.load(self.prefix.with_suffix('.npz')) as z:
            initial=z['initial'][0];solution=z['solution'].copy()
        if solution.shape!=(1,28) or len(audit['results'])!=1 or audit['results'][0]['waypoint']!=0:raise ValueError('Expected one audited escape segment')
        for row in audit['results']:
            if len(row['segment_samples'])<19:raise ValueError('Missing escape samples')
            for item in [dict(row,collisions=row['candidate_collisions'])]+row['segment_samples']:
                if not item['cube_clearance']['pass_at_0p1mm'] or not item['support_contact']['passed']:raise ValueError('Rejected escape geometry')
                for c in item['collisions']:
                    if c['rigid_pair'] and (c['sat_signed_overlap_m'] is None or not np.isfinite(c['sat_signed_overlap_m']) or c['sat_signed_overlap_m']>.0001):raise ValueError('Rejected escape hand geometry')
        delta=solution[0]-initial
        fixed=[i for i in range(28) if i not in range(5,9)]
        if not np.isfinite(np.r_[initial,delta]).all() or np.any(delta[fixed]!=0) or not 0<np.max(np.abs(delta))<=.020000001:raise ValueError('Escape exceeds audited small-index scope')
        with np.load(self.raw_prefix.with_suffix('.npz')) as z:
            i=self.source['source_frame'];env=self.source['env']
            if z['trace'][i,env,1]!=self.source['source_step'] or not np.array_equal(initial[:22],z['sensory'][i,env,:22]):raise ValueError('Escape source pose mismatch')
            self.original={k:z[k][:i+1,env].copy() for k in ('trace','sensory','poses','motor_targets','wrist_targets','contact','core_angular_speed')}
            original=z['motor_targets'][i,env,0].astype(float)
            self.motor=np.stack([original,original+delta[:22]])
            if np.any(self.motor<z['actuator_lower'][env,0]-1e-7) or np.any(self.motor>z['actuator_upper'][env,0]+1e-7):raise ValueError('Escape motor bounds')
        self.duration=float(np.max(np.abs(delta)));self.times=np.array([0.,self.duration])


def write_actual_left_candidate(raw,prefix,trial,env,kind,control_hz=40.,deadline=6.):
    """Export an actual completed source and one predefined diagnostic candidate.

    No simulation state is changed. The caller must independently audit the
    candidate and verify its history before applying any motor target.
    """
    from scipy.spatial.transform import Rotation
    if kind not in ('residual','full'):raise ValueError('Unknown left path kind')
    prefix=Path(prefix);source=Path(str(prefix)+'_source');paths=Path(str(prefix)+'_paths')
    arrays={k:np.asarray(v) for k,v in raw.items()}
    t=arrays['trace'][:,env];i=len(t)-1
    if not len(t) or int(t[i,0])!=trial or t[i,3]!=2 or not t[i,4]:raise ValueError('Expected current active third action')
    for k in ('trace','sensory','poses','motor_targets','wrist_targets','contact','core_angular_speed'):
        if len(arrays[k])!=len(t) or not np.isfinite(arrays[k][:,env]).all():raise ValueError('Invalid actual source history: '+k)
    ids=np.flatnonzero((t[:,0]==trial)&(t[:,3]==2)&(t[:,4]>0))
    if not np.array_equal(t[ids,1],np.arange(t[ids[0],1],t[i,1]+1)):raise ValueError('Missing source frames')
    remaining=deadline-(t[i,1]-t[ids[0],1]+1)/control_hz
    q=arrays['sensory'][i,env,:22].astype(float);p=arrays['poses'][i,env]
    rot=Rotation.from_quat(p[3:7][[1,2,3,0]])
    wp=rot.inv().apply(p[7:10]-arrays['env_origins'][env]-p[:3])
    wr=rot.inv().as_matrix()@Rotation.from_quat(p[10:14][[1,2,3,0]]).as_matrix()
    initial=np.r_[q,np.zeros(6)];x=initial.copy();bias=arrays['motor_targets'][i,env,0].astype(float)-q
    ready=arrays['ready_targets'][trial,env,0].astype(float)-bias
    if kind=='residual':x[7]+=.02;steps=[x.copy()]
    else:
        x[6]-=.4;steps=[x.copy()]
        for j in (5,7,6,8):x[j]=ready[j];steps.append(x.copy())
        if np.max(np.abs(steps[-1][5:9]-arrays['ready_targets'][trial,env,0,5:9]))>.1:raise ValueError('Unloaded source bias exceeds planning ready margin')
    physical=np.vstack([initial,steps]);motor=physical[:,:22]+bias
    lo=arrays['actuator_lower'][env,0];hi=arrays['actuator_upper'][env,0]
    if np.any(motor<lo-1e-7) or np.any(motor>hi+1e-7):raise ValueError('Actual-source motor bounds')
    duration=float(np.max(np.abs(np.diff(motor[:,5:9],axis=0)),axis=1).sum())
    if not 0<duration<remaining:raise ValueError('Actual-source deadline')
    outputs=[p.with_suffix(s) for p in (prefix,source,paths) for s in ('.json','.npz')]
    if any(p.exists() for p in outputs):raise FileExistsError(prefix)
    prefix.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(source.with_suffix('.npz'),**arrays)
    source.with_suffix('.json').write_text(json.dumps(dict(control_hz=control_hz,scope='Actual completed history; no state writes')))
    row=dict(trial=trial,env=env,source_frame=i,source_step=int(t[i,1]),move="U'",side='left',error_deg=float(t[i,5]))
    paths.with_suffix('.json').write_text(json.dumps(dict(source=str(source),results=[row],contact_face='B',contact_press_m=.001)))
    np.savez_compressed(paths.with_suffix('.npz'),U_initial_q=q[None],U_wrist_pos=wp[None],U_wrist_rot=wr[None])
    result=dict(source=str(paths),phase='withdrawal_actual_'+kind,sequential=True,whole_hand=True,variable_joint_indices=list(range(22)),support_contact=True,contact_transport=False,cooked_hulls='outputs/diagnostics/hand_physx_cooked_assets',bias_rad=bias.tolist(),duration_seconds=duration,remaining_original_seconds=float(remaining),scope='Actual-state geometry candidate only; independent audit and motor execution required',results=[dict(trial=trial,env=env,waypoint=j,measured_feasible=False,segment_pass=False) for j in range(len(steps))])
    prefix.with_suffix('.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(prefix.with_suffix('.npz'),initial=np.repeat(initial[None],len(steps),axis=0),solution=np.array(steps))
    return result


class LiveLeftWithdrawal:
    """Privileged actual-state diagnostic; one residual attempt, then full return."""
    def __init__(self,owner):
        self.owner=owner;self.escape=None;self.full=None;self.rejected=False
        self.residual_attempted=False;self.finished_event=False

    def prepare(self,e,trial,step,preload_start,preload_duration,events):
        import torch
        if preload_start is None:return None,0.
        env=self.owner.env;dt=float(e.step_dt)
        minimum=preload_start+int(np.ceil(preload_duration/dt-1e-9))-1
        if self.escape is not None:minimum=self.escape.source['source_step']+int(np.ceil(self.escape.duration/dt-1e-9))
        history=self.owner.history
        if self.full is None and not self.rejected:
            kind=next_left_path_kind(history,trial,env,minimum,self.residual_attempted)
            if kind:
                prefix=Path(str(self.owner.out)+'_left_'+kind)
                try:
                    arrays={k:np.asarray(history[k]) for k in ('trace','sensory','poses','motor_targets','wrist_targets','contact','core_angular_speed','ready_targets')}
                    arrays.update(env_origins=e.scene.env_origins.cpu().numpy(),actuator_lower=torch.stack([e.finger_lower[s] for s in ('left','right')],1).cpu().numpy(),actuator_upper=torch.stack([e.finger_upper[s] for s in ('left','right')],1).cpu().numpy())
                    write_actual_left_candidate(arrays,prefix,trial,env,kind,control_hz=1/dt)
                    self.owner._child(['hand-collision','--solutions',str(prefix),'--out',str(prefix)+'_audit','--usd-convex','--cooked-hulls','outputs/diagnostics/hand_physx_cooked_assets','--cube-clearance','--segment-samples','19'],'left_'+kind+'_audit')
                    path=(LeftReturnPath if kind=='full' else LeftResidualEscapePath)(prefix)
                    path.verify_history(history)
                    if kind=='full':self.full=path
                    else:self.escape=path;self.residual_attempted=True
                    events.append(dict(event='left_return_start' if kind=='full' else 'preload_residual_escape_start',trial=trial,step=step,env=env,source=str(prefix),duration_seconds=path.duration,scope='Firsteligible actual-state candidate, independent audit and exact livehistory; no state/clock/reference writes'))
                except Exception as exc:
                    self.rejected=True
                    events.append(dict(event='left_live_plan_rejected',trial=trial,step=step,env=env,kind=kind,error=repr(exc)))
        path=self.full or self.escape
        if path is None:return None,0.
        elapsed=(step-path.source['source_step'])*dt
        cols=e.flick_cols[e.move_id[env]].clone()
        if cols.tolist()!=[5,6,7,8]:raise RuntimeError('Unexpected live left mapping')
        values=torch.as_tensor(path.sample(elapsed),device=e.device,dtype=e.finger_tgt['left'].dtype)
        e.post_return_hold_target=(env,'left',cols,values)
        if self.full is not None and elapsed>=path.duration and not self.finished_event:
            events.append(dict(event='left_return_targets_finished',trial=trial,step=step,env=env));self.finished_event=True
        return self.full,elapsed if self.full is not None else 0.
