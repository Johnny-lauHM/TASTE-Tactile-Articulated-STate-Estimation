"""Audited privileged withdrawal from the actual arrival state.

No physical-state, task-clock, goal-angle or pose-reference writes. The inner
ready box is a planning target; the original full-cycle monitor scores success.
"""
import json,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from rubik_lab.diagnostics.whole_hand_path import WholeHandPath,WholeHandProbe


def inner_ready_endpoint(q,command,ready,lower,upper):
    arrays=[np.asarray(v,dtype=float) for v in (q,command,ready,lower,upper)]
    if any(v.shape!=(22,) or not np.isfinite(v).all() for v in arrays):raise ValueError('Invalid return vectors')
    q,command,ready,lower,upper=arrays
    if (lower>=upper).any() or (command<lower).any() or (command>upper).any():raise ValueError('Invalid return actuator domain')
    bias=command-q;exact=q.copy();exact[5:9]=ready[5:9]-bias[5:9]
    low,high=0.,1.
    for qi,di,ri in zip(q[5:9],exact[5:9]-q[5:9],ready[5:9]):
        if abs(di)<1e-14:
            if abs(qi-ri)>.1:raise ValueError('No inner-ready intersection')
            continue
        t0,t1=sorted(((ri-.1-qi)/di,(ri+.1-qi)/di));low=max(low,t0);high=min(high,t1)
    if low>high:raise ValueError('No inner-ready intersection')
    fraction=float(np.nextafter(low,high));end=q+fraction*(exact-q)
    motor=command.copy();motor[5:9]=end[5:9]+bias[5:9]
    if (motor<lower-1e-12).any() or (motor>upper+1e-12).any():raise ValueError('Return endpoint outside actuator limits')
    if np.max(np.abs(end[5:9]-ready[5:9]))>.1+1e-12:raise ValueError('Inner-ready rule failed')
    return end,bias,fraction


def write_candidate(prefix,arrays,trial,env,step,control_hz,primitive_seconds,side='right',stage=1):
    """Snapshot completed history for independent audit, not simulator reset."""
    from rubik_lab.hand_kinematics import HandFK
    if side not in ('left','right') or stage not in (1,2):raise ValueError('Unsupported return side/stage')
    hand=0 if side=='left' else 1;pose_offset=7+7*hand
    prefix=Path(prefix);source=Path(str(prefix)+'_source');paths=Path(str(prefix)+'_paths')
    for p in (prefix,source,paths):
        for suffix in ('.json','.npz'):
            if p.with_suffix(suffix).exists():raise FileExistsError(p)
    frame=len(arrays['trace'])-1;trace=arrays['trace'][:,env]
    if frame<0 or tuple(trace[frame,:2].astype(int))!=(trial,step-1):raise ValueError('Missing exact return source')
    q=arrays['sensory'][frame,env,22*hand:22*(hand+1)].astype(float);command=arrays['motor_targets'][frame,env,hand].astype(float)
    ready=arrays['ready_targets'][trial,env,hand].astype(float)
    lower=arrays['actuator_lower'][env,hand];upper=arrays['actuator_upper'][env,hand]
    end,bias,fraction=inner_ready_endpoint(q,command,ready,lower,upper)
    fk=HandFK(side,'cpu',pts_per_link=8,seed=0)
    if (end[5:9]<fk.lower.numpy()[5:9]).any() or (end[5:9]>fk.upper.numpy()[5:9]).any():raise ValueError('Return endpoint exceeds hard joint limits')
    if not (arrays['contact'][frame,env,0,hand,[5,15]]>.05).all():raise ValueError('Missing actual thumb/middle support')
    previous=np.flatnonzero((trace[:frame+1,0]==trial)&(trace[:frame+1,3]==stage-1))
    if not len(previous) or trace[frame,3]!=stage:raise ValueError('Missing original primitive clock')
    remaining=primitive_seconds-(trace[frame,1]-trace[previous[-1],1])/control_hz
    duration=max(float(np.max(np.abs(end-q))),1/control_hz)
    if duration>=remaining:raise ValueError('Return exceeds original deadline')
    pose=arrays['poses'][frame,env];cube=Rotation.from_quat(pose[3:7][[1,2,3,0]])
    wp=cube.inv().apply(pose[pose_offset:pose_offset+3]-arrays['env_origins'][env]-pose[:3])
    wr=cube.inv().as_matrix()@Rotation.from_quat(pose[pose_offset+3:pose_offset+7][[1,2,3,0]]).as_matrix()
    row=dict(trial=trial,env=env,source_frame=frame,source_step=step-1,move='U' if side=='right' else "U'",side=side,error_deg=float(trace[frame,5]))
    prefix.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(source.with_suffix('.npz'),**arrays)
    source.with_suffix('.json').write_text(json.dumps(dict(control_hz=control_hz,timeout_per_primitive_s=primitive_seconds,scope='Completed actual history at withdrawal entry; no state/clock/reference reset'))+'\n')
    paths.with_suffix('.json').write_text(json.dumps(dict(source=str(source),results=[row],contact_face='B',contact_press_m=.001))+'\n')
    np.savez_compressed(paths.with_suffix('.npz'),U_initial_q=q[None],U_wrist_pos=wp[None],U_wrist_rot=wr[None],U_actuator_lower=lower[None],U_actuator_upper=upper[None],U_target_bias=bias[None])
    report=dict(source=str(paths),sequential=True,whole_hand=True,variable_joint_indices=list(range(22)),cooked_hulls='outputs/diagnostics/hand_physx_cooked_assets',support_contact=True,contact_transport=False,contact_face='B',phase='withdrawal_inner_ready_candidate',duration_seconds=duration,remaining_original_seconds=float(remaining),return_fraction=fraction,predicted_ready_error_rad=float(np.max(np.abs(end[5:9]-ready[5:9]))),results=[dict(trial=trial,env=env,waypoint=0,measured_feasible=False,segment_pass=False)],scope='Candidate only; originalready reference and.2rad completioncriterion unchanged')
    prefix.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(prefix.with_suffix('.npz'),initial=np.r_[q,np.zeros(6)][None],solution=np.r_[end,np.zeros(6)][None])
    return report


class LiveReadyReturn:
    def __init__(self,parent):
        self.parent=parent;self.probe=None;self.stable=0;self.attempted=False
    @property
    def targets(self):return self.probe.targets if self.probe else []
    @property
    def permits_completion(self):return bool(self.probe and self.probe.started and self.probe.finished)
    def _delegate(self,e,trial,step,active,stage):
        before=len(self.probe.targets);self.probe.prepare(e,trial,step,active,stage)
        for row in self.probe.targets[before:]:
            merged=list(row);merged[3]=(step-self.parent.start_step+1)*e.step_dt
            self.parent.targets.append(merged)
        return self.probe.started
    def prepare(self,e,trial,step,active,stage):
        p=self.parent;i=p.env
        if not p.started or trial!=p.trial:return False
        if self.probe is not None:return self._delegate(e,trial,step,active,stage)
        if self.attempted:return False
        s=e.snapshot
        eligible=bool(active[i]) and int(stage[i])==1 and abs(float(s['twist_error'][i]))<np.deg2rad(5) and abs(float(s['twist_velocity'][i]))<.25 and float(s['core_angular_speed'][i])<1 and bool(e.cycle_motion_ok[i]) and not bool(e.cycle_monitor.failed[i])
        self.stable=self.stable+1 if eligible else 0
        if self.stable<4:return False
        self.attempted=True;clock=e.cycle_step.clone();start=time.monotonic()
        prefix=Path(str(p.out)+'_return');audit=Path(str(p.out)+'_return_audit')
        p._event(dict(event='return_planning_started',trial=trial,step=step,env=i,error_deg=float(s['twist_error'][i])*180/np.pi,scope='First4arrival-validframes; pauseddiagnostic planning, originalclock/reference retained'))
        try:
            report=write_candidate(prefix,p.capture_arrays(e),trial,i,step,1/e.step_dt,p.path.primitive_seconds)
            p._child(['hand-collision','--solutions',str(prefix),'--out',str(audit),'--usd-convex','--cooked-hulls','outputs/diagnostics/hand_physx_cooked_assets','--cube-clearance','--segment-samples','19'],'return_audit',timeout=600)
            first_audit=json.loads(audit.with_suffix('.json').read_text())
            try:
                p.validate_audit(first_audit,True,0)
            except ValueError as rejected:
                if str(rejected)!='Independent rigid-hand overlap rejected':raise
                # One prescribed geometry refinement, never a retry of physical execution.
                solved=Path(str(prefix)+'_constrained');checked=Path(str(solved)+'_audit')
                p._event(dict(event='return_geometry_refinement',trial=trial,step=step,env=i,source=str(prefix),rejected_audit=str(audit),reason=str(rejected)))
                p._child(['ready-return-geometry','--candidate',str(prefix),'--audit',str(audit.with_suffix('.json')),'--out',str(solved)],'return_geometry',timeout=600)
                p._child(['hand-collision','--solutions',str(solved),'--out',str(checked),'--usd-convex','--cooked-hulls','outputs/diagnostics/hand_physx_cooked_assets','--cube-clearance','--segment-samples','19'],'return_constrained_audit',timeout=600)
                p.validate_audit(json.loads(checked.with_suffix('.json').read_text()),True,0)
                prefix,audit=solved,checked
                report=json.loads(prefix.with_suffix('.json').read_text())
            if not bool((clock==e.cycle_step).all()):raise RuntimeError('Original clock changed during return audit')
            report['independent_audit']=str(audit)
            report['results'][0].update(measured_feasible=True,segment_pass=True)
            prefix.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
            self.probe=WholeHandProbe(WholeHandPath.from_result(prefix,trial,i,stop_waypoint=0),trial,i,preserve_servo_bias=True)
            def event_callback(event):
                self.probe.events.append(event)
                name='return_targets_finished_holding_ready' if event['event']=='prefix_targets_finished_holding_endpoint' else 'return_'+event['event']
                p._event(dict(event, event=name))
            self.probe._event=event_callback
            p._event(dict(event='return_plan_audited',trial=trial,step=step,env=i,source=str(prefix),audit=str(audit),planning_wall_seconds=time.monotonic()-start,duration_seconds=report['duration_seconds'],remaining_original_seconds=report['remaining_original_seconds']))
            if self._delegate(e,trial,step,active,stage):return True
            self.probe=None
        except Exception as error:
            self.probe=None
            p._event(dict(event='return_planning_rejected',trial=trial,step=step,env=i,error=repr(error),planning_wall_seconds=time.monotonic()-start))
        return False
