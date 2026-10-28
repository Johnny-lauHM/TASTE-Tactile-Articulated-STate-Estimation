"""Audited privileged full-hand contact arc, without state/reference updates."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

HISTORY=('trace','sensory','poses','motor_targets','wrist_targets','contact','core_angular_speed')

def validate_audit(audit,prefix,count):
    if audit.get('source')!=str(prefix)+'_geometry' or audit.get('active_side')!='left' or not audit.get('cube_clearance') or not audit.get('support_contact_audited') or audit.get('segment_samples')!=19:
        raise ValueError('Coordinated audit scope mismatch')
    rows=audit['results']
    if len(rows)!=count or [r['waypoint'] for r in rows]!=list(range(count)):
        raise ValueError('Coordinated audit missing waypoints')
    for row in rows:
        if len(row['segment_samples'])!=19:raise ValueError('Missing interior geometry')
        for sample in [dict(row,collisions=row['candidate_collisions']),*row['segment_samples']]:
            cube=sample['cube_clearance'];depth=cube['active_pad_top_signed_overlap_mm']
            if not cube['pass_at_0p1mm'] or not np.isfinite(depth) or not 0<=depth<=1 or not sample['support_contact']['passed']:
                raise ValueError('Rejected coordinated contact geometry')
            for collision in sample['collisions']:
                depth=collision['sat_signed_overlap_m']
                if collision['rigid_pair'] and (depth is None or not np.isfinite(depth) or depth>.0001):
                    raise ValueError('Rejected coordinated rigid hand collision')

class CoordinatedContactPath:
    def __init__(self,prefix,bind_actual=False):
        self.bind_actual=bool(bind_actual)
        self.prefix=Path(prefix)
        report=json.loads(self.prefix.with_suffix('.json').read_text())
        geometry=Path(str(prefix)+'_geometry')
        gr=json.loads(geometry.with_suffix('.json').read_text())
        source=json.loads(Path(gr['source']).with_suffix('.json').read_text())
        if len(source['results'])!=1:raise ValueError('Ambiguous coordinated source')
        row=source['results'][0];self.env=int(row['env']);self.trial=int(row['trial']);self.frame=int(row['source_frame'])
        if (row['side'],row['move'])!=('left',"U'") or gr['phase']!='turn_geometry' or gr['variable_joint_indices']!=list(range(22)):
            raise ValueError('Coordinated path identity')
        self.count=len(gr['results'])
        if self.count<1 or not all(r['kinematic_pass'] for r in report['results'][:self.count]):raise ValueError('Rejected solver prefix')
        if gr['layer_angle_deltas_deg']!=[r['degrees'] for r in report['results'][:self.count]]:raise ValueError('Declared angle mismatch')
        self.target_degrees=float(gr['layer_angle_deltas_deg'][-1])
        if source['source']!=report['source']:raise ValueError('Coordinated raw identity mismatch')
        audit=json.loads(Path(str(geometry)+'_audit.json').read_text());validate_audit(audit,self.prefix,self.count)
        with np.load(Path(report['source']).with_suffix('.npz')) as raw:
            self.history={k:raw[k][:self.frame+1,self.env].copy() for k in HISTORY}
            self.source_step=int(self.history['trace'][-1,1]);self.origins=raw['env_origins'][self.env].copy()
            self.lower=raw['actuator_lower'][self.env,0].copy();self.upper=raw['actuator_upper'][self.env,0].copy()
        trace=self.history['trace'];active=np.flatnonzero((trace[:,0]==self.trial)&(trace[:,3]==3)&(trace[:,4]>0))
        if not len(active) or active[-1]!=self.frame or not np.array_equal(trace[active,1],np.arange(trace[active[0],1],self.source_step+1)):raise ValueError('Missing original action history')
        self.handoff_step=int(trace[active[0],1])-1
        if report['source_step']!=self.source_step:raise ValueError('Coordinated source step mismatch')
        with np.load(self.prefix.with_suffix('.npz')) as candidate,np.load(geometry.with_suffix('.npz')) as audited:
            initial=candidate['initial'][0].copy();v=candidate['solution'][:self.count].copy()
            budget=.01-float(trace[-1,14])-1e-6
            normalized=v.copy();normalized[:,22:25]/=budget
            if not np.array_equal(audited['initial'],candidate['initial']) or not np.array_equal(audited['solution'],normalized):raise ValueError('Candidate differs from audited geometry')
        if initial.shape!=(28,) or v.shape!=(self.count,28) or not np.isfinite(v).all() or not np.array_equal(initial[:22],self.history['sensory'][-1,:22]) or np.any(initial[22:]!=0) or np.any(v[:,25:]!=0):raise ValueError('Invalid coordinated variables')
        if budget<=0 or np.any(np.linalg.norm(v[:,22:25],axis=1)>budget):raise ValueError('Original wrist budget exceeded')
        values=np.vstack([initial,v]);bias=self.history['motor_targets'][-1,0]-initial[:22]
        self.motor=values[:,:22]+bias
        if np.any(self.motor<self.lower) or np.any(self.motor>self.upper):raise ValueError('Coordinated motor bounds')
        intervals=np.ceil(np.maximum(np.abs(np.diff(values[:,:22],axis=0)).max(1),.025)*40)
        self.times=np.r_[0.,np.cumsum(intervals)]/40;self.duration=float(self.times[-1])
        if self.duration>=6-(self.source_step-self.handoff_step)/40:raise ValueError('Original deadline exceeded')
        cube=Rotation.from_quat(self.history['poses'][-1,3:7][[1,2,3,0]])
        baseline=self.history['wrist_targets'][-1,0]
        self.positions=baseline[:3]+cube.apply(values[:,22:25]);self.quaternion=baseline[3:7].copy()
        self.started=False;self.finished=False

    def validate_geometry_inputs(self,raw):
        """Cache eligibility, not relaxed history equality or sensor acceptance.

        Delivered FT is not consumed by this privileged geometric planner.
        Preserve it in a new actual-source record; never replace it by template FT.
        """
        for key,expected in self.history.items():
            actual=np.asarray(raw[key])
            if len(actual)!=self.frame+1 or not np.isfinite(actual[:,self.env]).all():raise ValueError('Invalid actual source: '+key)
            value=actual[:,self.env]
            if key=='sensory':value,expected=value[:,:88],expected[:,:88]
            if not np.array_equal(value,expected):raise ValueError('Changed geometric input: '+key)
        if not np.array_equal(raw['env_origins'][self.env],self.origins) or not np.array_equal(raw['actuator_lower'][self.env,0],self.lower) or not np.array_equal(raw['actuator_upper'][self.env,0],self.upper):raise ValueError('Changed geometric actuator/origin input')
        actual=np.asarray(raw['sensory'])[:,self.env,88:];old=self.history['sensory'][:,88:]
        indices=np.argwhere(actual!=old)
        return dict(delivered_ft_difference_count=len(indices),delivered_ft_max_abs_difference=float(np.abs(actual-old).max()),delivered_ft_differences=[dict(frame=int(i),column=int(j+88),actual=float(actual[i,j]),template=float(old[i,j])) for i,j in indices],scope='FT excluded only from privileged geometry cache eligibility; actual FT history retained, not blind-policy or sensor-robustness evidence')

    def bind_to_actual(self,raw,prefix,audit):
        provenance=self.validate_geometry_inputs(raw);prefix=Path(prefix)
        template=self.prefix;geometry=Path(str(template)+'_geometry')
        old_geometry=json.loads(geometry.with_suffix('.json').read_text())
        old_source=Path(old_geometry['source'])
        source=Path(str(prefix)+'_raw');paths=Path(str(prefix)+'_source');new_geometry=Path(str(prefix)+'_geometry')
        outputs=(source.with_suffix('.npz'),paths.with_suffix('.json'),paths.with_suffix('.npz'),prefix.with_suffix('.json'),prefix.with_suffix('.npz'),new_geometry.with_suffix('.json'),new_geometry.with_suffix('.npz'))
        if any(p.exists() for p in outputs):raise FileExistsError(prefix)
        np.savez_compressed(source.with_suffix('.npz'),**raw)
        sr=json.loads(old_source.with_suffix('.json').read_text());sr['source']=str(source)
        paths.with_suffix('.json').write_text(json.dumps(sr,indent=2))
        # All inputs to these arrays were checked exactly; no sensor history copied.
        with np.load(old_source.with_suffix('.npz')) as z:np.savez_compressed(paths.with_suffix('.npz'),**dict(z))
        report=json.loads(template.with_suffix('.json').read_text());report['source']=str(source)
        report['actual_source_binding']=dict(template=str(template),**provenance)
        prefix.with_suffix('.json').write_text(json.dumps(report,indent=2))
        with np.load(template.with_suffix('.npz')) as z:np.savez_compressed(prefix.with_suffix('.npz'),**dict(z))
        old_geometry['source']=str(paths);new_geometry.with_suffix('.json').write_text(json.dumps(old_geometry,indent=2))
        with np.load(geometry.with_suffix('.npz')) as z:np.savez_compressed(new_geometry.with_suffix('.npz'),**dict(z))
        audit(new_geometry)
        bound=CoordinatedContactPath(prefix)
        bound.verify_history(raw)
        return bound,provenance

    def verify_history(self,history):
        for key,expected in self.history.items():
            actual=np.asarray(history[key])
            if len(actual)!=self.frame+1 or not np.array_equal(actual[:,self.env],expected):raise ValueError('Coordinated source history mismatch: '+key)

    def sample(self,elapsed):
        if not np.isfinite(elapsed) or elapsed<0:raise ValueError('Invalid coordinated elapsed time')
        interpolate=lambda a:np.asarray([np.interp(elapsed,self.times,a[:,j]) for j in range(a.shape[1])])
        return dict(fingers=interpolate(self.motor),wrist_position=interpolate(self.positions),wrist_quaternion=self.quaternion.copy())

    def prepare(self,e,owner,release_cycle,trial,step,events):
        import torch
        if trial!=self.trial or owner.env!=self.env:raise ValueError('Coordinated runtime identity')
        if step<=self.source_step:return
        if not self.started:
            from .left_recontact import first_turn_source
            if step!=self.source_step+1 or release_cycle is None or release_cycle.phase!='recontact' or release_cycle.finished_step is None:raise ValueError('Missed coordinated contact boundary')
            raw={k:np.asarray(owner.history[k]) for k in HISTORY}
            if self.bind_actual:
                raw.update(env_origins=e.scene.env_origins.cpu().numpy(),actuator_lower=torch.stack([e.finger_lower[s] for s in ('left','right')],1).cpu().numpy(),actuator_upper=torch.stack([e.finger_upper[s] for s in ('left','right')],1).cpu().numpy())
                def audit(geometry):
                    owner._child(['hand-collision','--solutions',str(geometry),'--out',str(geometry)+'_audit','--usd-convex','--cooked-hulls','outputs/diagnostics/hand_physx_cooked_assets','--cube-clearance','--segment-samples','19'],'coordinated_contact_actual_audit')
                bound,provenance=self.bind_to_actual(raw,Path(str(owner.out)+'_coordinated_contact'),audit)
                self.__dict__.update(bound.__dict__)
                events.append(dict(event='coordinated_contact_actual_source_bound',trial=trial,env=self.env,step=step,source=str(self.prefix),**provenance))
            self.verify_history(owner.history)
            if first_turn_source(raw,trial,self.env,release_cycle.finished_step,entry_delay_frames=1)!=self.frame:raise ValueError('Coordinated stable contact gate')
            if abs(float(e.step_dt)-.025)>1e-12 or int(e.cycle_step[self.env])!=self.source_step-self.handoff_step:raise ValueError('Coordinated original clock mismatch')
            if not np.array_equal(e.finger_lower['left'][self.env].cpu().numpy(),self.lower) or not np.array_equal(e.finger_upper['left'][self.env].cpu().numpy(),self.upper):raise ValueError('Coordinated actuator configuration changed')
            self.started=True;events.append(dict(event='coordinated_contact_start',trial=trial,env=self.env,step=step,source=str(self.prefix),source_step=self.source_step,duration_seconds=self.duration,target_degrees=self.target_degrees,scope='Privileged actual-source diagnostic; original clock/references unchanged; no atomic credit'))
        elapsed=(step-self.source_step)*float(e.step_dt);value=self.sample(elapsed)
        tensor=lambda a:torch.as_tensor(a,device=e.device,dtype=e.finger_tgt['left'].dtype)
        e.fourth_recontact_target=(self.env,'left',torch.arange(22,device=e.device),tensor(value['fingers']))
        e.coordinated_wrist_target=(self.env,'left',tensor(value['wrist_position']),tensor(value['wrist_quaternion']))
        if elapsed>=self.duration and not self.finished:
            self.finished=True;events.append(dict(event='coordinated_contact_targets_finished',trial=trial,env=self.env,step=step))
