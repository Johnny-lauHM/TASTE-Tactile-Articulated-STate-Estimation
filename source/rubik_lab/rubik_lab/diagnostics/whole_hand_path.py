"""Recorded whole-hand target trajectory for privileged diagnostic execution.

No environment writes, alignment, clock reset or task-success judgment. The
caller must check actual start fidelity and retain the original task monitors.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation


def tangential_transport(pad,rotation,initial_rotation,anchor,tangent,tolerance=.0005):
    """Diagnostic fingertip transport constraint; not a force/sticking model."""
    if tolerance<=0:raise ValueError('Positive transport tolerance required')
    target=rotation@initial_rotation.T@anchor
    direction=rotation@tangent
    error=((pad-target)*direction).sum()
    return 1-(error/tolerance).square(),error,target


def contact_approach_targets(pad_cube,error_deg,face,press_m=.001,lateral_max_m=.028):
    """Same lift/outside/press path family on a selected U side face.

    Negative lateral offset along z cross normal generates negative-z torque
    under inward normal force. This geometry does not assert actual contact.
    """
    normals={'R':(0.,-1.,0.),'B':(1.,0.,0.)}
    pad=np.asarray(pad_cube,dtype=float)
    if face not in normals or pad.shape!=(3,) or not np.isfinite(pad).all():raise ValueError('Invalid contact path input')
    if not np.isfinite(error_deg) or not -90<=error_deg<0 or not 0<=press_m<.035:raise ValueError('Expected negative U residual and valid press')
    if not .019-1e-12<=lateral_max_m<=.028+1e-12:raise ValueError('Contact corridor must be a subset of original19..28mm range')
    normal=np.array(normals[face]);axis=np.array([0.,0.,1.]);tangent=np.cross(axis,normal)
    turn=Rotation.from_rotvec(axis*np.deg2rad(-error_deg))
    local=turn.inv().apply(pad)
    lateral=-np.clip(-float(local@tangent),.019,lateral_max_m)
    contact=(.035-press_m)*normal+lateral*tangent+(2*.035/3)*axis
    p0=turn.apply(contact);cn=turn.apply(normal)
    a0=pad+.008*axis
    keys=np.stack([pad,a0,a0+.007*cn,p0+.007*cn+.003*axis,p0+.007*cn,p0])
    cumulative=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(keys,axis=0),axis=1))]
    distances=np.linspace(0.,cumulative[-1],13)[1:]
    approach=np.stack([np.interp(distances,cumulative,keys[:,j]) for j in range(3)],axis=1)
    turns=np.stack([Rotation.from_rotvec(axis*np.deg2rad(-error_deg-d)).apply(contact) for d in range(int(np.ceil(abs(error_deg)))+1)])
    return np.concatenate([approach,turns])


def side_patch_error(pad,angle,face,lateral_max_m,press_m=.001):
    """Independent NumPy distance to the declared rectangular side patch."""
    pad=np.asarray(pad,dtype=float)
    if pad.shape!=(3,) or not np.isfinite(pad).all() or not np.isfinite(angle):raise ValueError('Invalid contact geometry')
    if face not in ('R','B') or not .019-1e-12<=lateral_max_m<=.028+1e-12 or not 0<=press_m<.035:raise ValueError('Invalid side patch')
    rotation=Rotation.from_rotvec([0.,0.,angle]);local=rotation.inv().apply(pad)
    normal=np.array([1.,0.,0.]) if face=='B' else np.array([0.,-1.,0.])
    tangent=np.cross([0.,0.,1.],normal)
    lateral=float(local@tangent)
    height=float(local[2])
    projected=(.035-press_m)*normal+np.clip(lateral,-lateral_max_m,-.019)*tangent+np.array([0.,0.,np.clip(height,.035/3+.003,.035-.003)])
    return dict(error_m=float(np.linalg.norm(local-projected)),local_pad_m=local.tolist(),lateral_m=lateral,projected_local_m=projected.tolist())


def bounded_index_feedback(reference,measured,base,previous,lower,upper,dt,gain=1.):
    """Proprioceptive P correction, declared gain1 or3, .1rad cap and1rad/s command rate.

    Limits commands, never physical states or completion requirements. No
    integral state; saturation cannot wind up. Applies only to four index DOFs.
    """
    arrays=[np.asarray(v,dtype=float) for v in (reference,measured,base,previous,lower,upper)]
    if any(v.shape!=(4,) or not np.isfinite(v).all() for v in arrays) or not np.isfinite(dt) or dt<=0:raise ValueError('Invalid index feedback input')
    reference,measured,base,previous,lower,upper=arrays
    if (lower>=upper).any() or (previous<lower).any() or (previous>upper).any():raise ValueError('Invalid feedback command domain')
    if gain not in (1.,3.):raise ValueError('Unregistered feedback gain')
    requested=gain*(reference-measured)
    desired=base+np.clip(requested,-.1,.1)
    command=np.clip(desired,np.maximum(lower,previous-dt),np.minimum(upper,previous+dt))
    applied=command-base
    return command,requested,applied,np.abs(applied-requested)>1e-12


def _rotation(wxyz):
    q=np.asarray(wxyz,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-3:
        raise ValueError('Expected a finite unit wxyz quaternion')
    return Rotation.from_quat(q[[1,2,3,0]])


def actuator_compatible_bounds(hard_lower,hard_upper,command_lower,command_upper,bias):
    """Physical-joint planning range whose biased actuator targets remain legal."""
    arrays=[np.asarray(x,dtype=float) for x in (hard_lower,hard_upper,command_lower,command_upper,bias)]
    if any(x.shape!=(22,) or not np.isfinite(x).all() for x in arrays):
        raise ValueError('Expected five finite22joint vectors')
    hl,hu,cl,cu,b=arrays
    if (hl>=hu).any() or (cl>=cu).any():raise ValueError('Invalid jointlimits')
    lower=np.maximum(hl,cl-b);upper=np.minimum(hu,cu-b)
    if (lower>=upper).any():raise ValueError('Empty physical/actuator joint-domain intersection')
    return lower,upper


@dataclass(frozen=True)
class WholeHandPath:
    variables: np.ndarray  # actual initial state, then contiguous solved waypoints
    wrist_position_cube: np.ndarray
    wrist_rotation_cube: np.ndarray
    cube_position: np.ndarray  # environment-relative, recorded source frame
    cube_quaternion: np.ndarray
    translation_scale: float
    rotation_scale: float
    remaining_seconds: float
    source_error_deg: float
    finger_rate: float=1.
    translation_rate: float=.02
    rotation_rate: float=np.pi/6
    control_dt: float=.025
    source_step: int=0
    primitive_seconds: float=6.

    def __post_init__(self):
        for name,shape in [('variables',None),('wrist_position_cube',(3,)),
                           ('wrist_rotation_cube',(3,3)),('cube_position',(3,)),
                           ('cube_quaternion',(4,))]:
            value=np.array(getattr(self,name),dtype=float,copy=True)
            if not np.isfinite(value).all() or (shape and value.shape!=shape):
                raise ValueError('Invalid '+name)
            value.setflags(write=False);object.__setattr__(self,name,value)
        x=self.variables
        if x.ndim!=2 or x.shape[1]!=28 or len(x)<2:
            raise ValueError('Need actual initial state and at least one 28-variable waypoint')
        R=self.wrist_rotation_cube
        if not np.allclose(R.T@R,np.eye(3),atol=1e-6) or np.linalg.det(R)<.999999:
            raise ValueError('Invalid wrist rotation')
        _rotation(self.cube_quaternion)
        values=[self.translation_scale,self.rotation_scale,self.remaining_seconds,
                self.finger_rate,self.translation_rate,self.rotation_rate,self.control_dt]
        if not np.isfinite(values).all() or min(values)<=0:
            raise ValueError('Scales, rates and remaining time must be finite and positive')
        if not np.isfinite(self.source_error_deg):raise ValueError('Invalid source angle')
        if np.max(np.linalg.norm(x[:,22:25],axis=1))>1+1e-8 or np.max(np.linalg.norm(x[:,25:],axis=1))>1+1e-8:
            raise ValueError('Trajectory exceeds its original normalized wrist budget')
        if not np.allclose(x[0,22:],0,atol=1e-12):
            raise ValueError('Initial waypoint must retain the recorded wrist reference')
        delta=np.diff(x,axis=0)
        durations=np.maximum.reduce([
            np.max(np.abs(delta[:,:22]),axis=1)/self.finger_rate,
            np.linalg.norm(delta[:,22:25],axis=1)*self.translation_scale/self.translation_rate,
            np.linalg.norm(delta[:,25:],axis=1)*self.rotation_scale/self.rotation_rate,
            np.full(len(delta),self.control_dt)])
        times=np.r_[0.,np.cumsum(durations)];times.setflags(write=False)
        object.__setattr__(self,'times',times)

    @property
    def duration(self):return float(self.times[-1])

    def sample(self,elapsed):
        """Targets only; elapsed is time since probe start, never the task clock."""
        if not np.isfinite(elapsed) or elapsed<0:raise ValueError('Invalid elapsed time')
        i=min(int(np.searchsorted(self.times,elapsed,side='right'))-1,len(self.times)-2)
        f=np.clip((elapsed-self.times[i])/(self.times[i+1]-self.times[i]),0,1)
        x=self.variables[i]+f*(self.variables[i+1]-self.variables[i])
        cube_R=_rotation(self.cube_quaternion)
        position=self.cube_position+cube_R.apply(self.wrist_position_cube+x[22:25]*self.translation_scale)
        rotation=cube_R*Rotation.from_rotvec(x[25:]*self.rotation_scale)*Rotation.from_matrix(self.wrist_rotation_cube)
        q=rotation.as_quat()[[3,0,1,2]]
        return dict(fingers=x[:22].copy(),wrist_position=position,wrist_quaternion=q,
                    path_finished=bool(elapsed>=self.duration))

    def start_errors(self,fingers,wrist_position,wrist_quaternion,cube_position,cube_quaternion,angle_error_deg):
        """Read-only mismatch measures; caller must preregister rejection thresholds."""
        start=self.sample(0.)
        fingers=np.asarray(fingers);position=np.asarray(wrist_position);cube_position=np.asarray(cube_position)
        if fingers.shape!=(22,) or position.shape!=(3,) or cube_position.shape!=(3,):raise ValueError('Invalid start shapes')
        if not all(np.isfinite(v).all() for v in (fingers,position,cube_position)) or not np.isfinite(angle_error_deg):raise ValueError('Nonfinite start')
        return dict(finger_max_rad=float(np.max(np.abs(fingers-start['fingers']))),
                    wrist_position_m=float(np.linalg.norm(position-start['wrist_position'])),
                    wrist_rotation_rad=float((_rotation(wrist_quaternion)*_rotation(start['wrist_quaternion']).inv()).magnitude()),
                    cube_position_m=float(np.linalg.norm(cube_position-self.cube_position)),
                    cube_rotation_rad=float((_rotation(cube_quaternion)*_rotation(self.cube_quaternion).inv()).magnitude()),
                    layer_error_difference_deg=abs(float(angle_error_deg)-self.source_error_deg))

    @classmethod
    def from_result(cls,prefix,trial,env,*,stop_waypoint=None,**rates):
        """Load a full geometric path, or an explicitly requested diagnostic prefix."""
        prefix=Path(prefix);report=json.loads(prefix.with_suffix('.json').read_text())
        if not report.get('sequential') or report['variable_joint_indices']!=list(range(22)):
            raise ValueError('Requires sequential whole-hand solutions')
        paths=Path(report['source']);pr=json.loads(paths.with_suffix('.json').read_text())
        source_rows=[r for r in pr['results'] if (r['trial'],r['env'])==(trial,env)]
        if len(source_rows)!=1:raise ValueError('Ambiguous source')
        row=source_rows[0]
        if (row['move'],row['side'])!=('U','right'):raise ValueError('Only recorded right/U diagnostic supported')
        full_end=12+int(np.ceil(abs(row['error_deg'])))
        end=full_end if stop_waypoint is None else stop_waypoint
        if not isinstance(end,int) or not 0<=end<=full_end:raise ValueError('Invalid prefix endpoint')
        selected=[(i,r) for i,r in enumerate(report['results']) if (r['trial'],r['env'])==(trial,env) and r['waypoint']<=end]
        if [r['waypoint'] for _,r in selected]!=list(range(end+1)):raise ValueError('Missing or reordered waypoints')
        if not all(r['measured_feasible'] and r.get('segment_pass',False) for _,r in selected):
            raise ValueError('Rejected geometric waypoint cannot be silently replayed')
        with np.load(prefix.with_suffix('.npz')) as z:
            variables=np.vstack([z['initial'][selected[0][0]],z['solution'][[i for i,_ in selected]]])
        si=next(i for i,r in enumerate(pr['results']) if r is row)
        with np.load(paths.with_suffix('.npz')) as z:
            wp=z['U_wrist_pos'][si].reshape(3);wr=z['U_wrist_rot'][si].reshape(3,3)
            if not np.allclose(variables[0,:22],z['U_initial_q'][si].reshape(22),atol=1e-8,rtol=0):
                raise ValueError('Stored initial joints do not match the actual source')
        raw_prefix=Path(pr['source']);meta=json.loads(raw_prefix.with_suffix('.json').read_text())
        with np.load(raw_prefix.with_suffix('.npz')) as z:
            frame=row['source_frame'];trace=z['trace'];pose=z['poses'][frame,env].copy()
            history=trace[:frame+1,env];previous=np.flatnonzero((history[:,0]==trial)&(history[:,3]==0))
            if not len(previous) or trace[frame,env,3]!=1:raise ValueError('Missing original handoff clock')
            elapsed=(trace[frame,env,1]-history[previous[-1],1])/meta['control_hz']
            peak=trace[frame,env,14:16].copy()
        return cls(variables,wp,wr,pose[:3],pose[3:7],.010-float(peak[0])-1e-6,
                   np.deg2rad(10)-float(peak[1])-1e-6,
                   float(meta['timeout_per_primitive_s']-elapsed),row['error_deg'],source_step=int(row['source_step']),primitive_seconds=float(meta['timeout_per_primitive_s']),**rates)


class WholeHandProbe:
    """One explicit prefix probe inside an existing continuous episode.

    Holds its last target after the prefix; no withdrawal/success claim. A
    completed action cannot be credited to this prefix-only intervention.
    """
    LIMITS=dict(finger_max_rad=.02,wrist_position_m=.0005,
                wrist_rotation_rad=np.deg2rad(.5),cube_position_m=.0005,
                cube_rotation_rad=np.deg2rad(.5),layer_error_difference_deg=1.)

    def __init__(self,path,trial,env,preserve_servo_bias=False,index_feedback=False):
        self.path=path;self.trial=trial;self.env=env
        self.started=False;self.checked=False;self.finished=False
        self.events=[];self.targets=[];self.preserve_servo_bias=preserve_servo_bias
        self.index_feedback=bool(index_feedback);self.feedback=[]

    def _event(self,event):
        self.events.append(event);print(json.dumps(dict(whole_hand_probe_event=event)),flush=True)

    def prepare(self,e,trial,step,active,stage):
        e.whole_hand_probe_target=None
        if trial!=self.trial:return
        i=self.env
        if not self.checked and step==self.path.source_step+1:
            self.checked=True
            if not bool(active[i]) or int(stage[i])!=1:
                self._event(dict(event='start_rejected_inactive_or_stage',trial=trial,step=step,env=i));return
            s=e.snapshot
            array=lambda x:x.detach().cpu().numpy()
            errors=self.path.start_errors(array(s['finger_pos'][i,1]),
                array(s['ee_pose'][i,1,:3]-e.scene.env_origins[i]),array(s['ee_pose'][i,1,3:]),
                array(s['cube_pos'][i]),array(s['cube_quat'][i]),float(s['twist_error'][i])*180/np.pi)
            remaining=self.path.primitive_seconds-float(e.cycle_step[i])*e.step_dt
            clock_ok=abs(remaining-self.path.remaining_seconds)<1e-6
            accepted=clock_ok and all(errors[k]<=v for k,v in self.LIMITS.items())
            accepted=accepted and self.path.duration<remaining
            start=self.path.sample(0.)
            self.finger_bias=np.zeros(22);self.position_bias=np.zeros(3);self.rotation_bias=Rotation.identity()
            if self.preserve_servo_bias:
                self.finger_bias=array(e.finger_tgt['right'][i])-start['fingers']
                self.position_bias=array(e.ee_pos_tgt['right'][i])-start['wrist_position']
                self.rotation_bias=_rotation(start['wrist_quaternion']).inv()*_rotation(array(e.ee_quat_tgt['right'][i]))
            lower=array(e.finger_lower['right'][i]);upper=array(e.finger_upper['right'][i])
            command_path=(self.path.variables[:,:22]+self.finger_bias).astype(array(e.finger_tgt['right'][i]).dtype)
            excess=np.maximum(lower-command_path,command_path-upper)
            command_bounds_ok=bool((excess<=0).all())
            accepted=accepted and command_bounds_ok
            self._event(dict(event='start_accepted' if accepted else 'start_rejected_mismatch_or_budget',trial=trial,step=step,env=i,errors=errors,limits=self.LIMITS,clock_matches=clock_ok,remaining_seconds=remaining,path_duration_seconds=self.path.duration,preserve_servo_bias=self.preserve_servo_bias,command_bounds_ok=command_bounds_ok,max_command_bound_excess_rad=float(excess.max()),worst_command_bound_index=list(map(int,np.unravel_index(np.argmax(excess),excess.shape))),finger_bias_rad=self.finger_bias.tolist(),wrist_position_bias_m=self.position_bias.tolist(),wrist_rotation_bias_rad=self.rotation_bias.as_rotvec().tolist()))
            if not accepted:return
            self.started=True;self.start_step=step
        if not self.started or not bool(active[i]):return
        target=self.path.sample((step-self.start_step+1)*e.step_dt)
        physical_reference=target['fingers'].copy()
        target['fingers']+=self.finger_bias;target['wrist_position']+=self.position_bias
        elapsed=(step-self.start_step+1)*e.step_dt
        if self.index_feedback and elapsed>=self.path.duration+.25:
            array=lambda x:x.detach().cpu().numpy()
            measured=array(e.snapshot['finger_pos'][i,1,5:9])
            previous=array(e.finger_tgt['right'][i,5:9])
            command,requested,applied,limited=bounded_index_feedback(physical_reference[5:9],measured,target['fingers'][5:9],previous,array(e.finger_lower['right'][i,5:9]),array(e.finger_upper['right'][i,5:9]),e.step_dt)
            target['fingers'][5:9]=command
            self.feedback.append([trial,step,i,*physical_reference[5:9],*measured,*requested,*applied,*limited.astype(float)])
        rotation=_rotation(target['wrist_quaternion'])*self.rotation_bias
        target['wrist_quaternion']=rotation.as_quat()[[3,0,1,2]]
        e.whole_hand_probe_target=(i,target)
        self.targets.append([trial,step,i,(step-self.start_step+1)*e.step_dt,*target['fingers'],*target['wrist_position'],*target['wrist_quaternion']])
        if target['path_finished'] and not self.finished:
            self.finished=True
            self._event(dict(event='prefix_targets_finished_holding_endpoint',trial=trial,step=step,env=i))
