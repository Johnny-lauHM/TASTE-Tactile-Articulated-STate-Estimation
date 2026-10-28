"""Privileged planning at a paused, actual continuation state; no state reset.

Planning wall time is logged and is NOT evidence of real-time robot capability.
Only a preregistered prefix is attempted, with original task monitors retained.
"""
import json,os,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from rubik_lab.diagnostics.whole_hand_path import WholeHandPath,WholeHandProbe


class LiveWholeHandProbe(WholeHandProbe):
    def __init__(self,trial,env,out,history,prefix_degrees=15,support_contact=False,support_reserve_m=0.,contact_face=None,hand_pairs_from=None,released_start=False,contact_transport=False,transport_reserve_m=0.,index_feedback=False,contact_lateral_max_m=None,ready_return=False):
        super().__init__(None,trial,env,preserve_servo_bias=True,index_feedback=index_feedback)
        self.out=Path(out);self.history=history;self.prefix_degrees=prefix_degrees
        if not 0<=support_reserve_m<50e-6 or (support_reserve_m and not support_contact):raise ValueError('Invalid support reserve')
        self.support_contact=support_contact;self.support_reserve_m=support_reserve_m
        if contact_face not in (None,'R','B'):raise ValueError('Unsupported contact face')
        self.contact_face=contact_face
        if contact_lateral_max_m is not None and (contact_face is None or not .019-1e-12<=contact_lateral_max_m<=.028+1e-12):raise ValueError('Invalid live contact corridor')
        self.contact_lateral_max_m=contact_lateral_max_m
        self.hand_pairs_from=Path(hand_pairs_from) if hand_pairs_from else None
        self.released_start=bool(released_start)
        if not 0<=transport_reserve_m<.0005 or (transport_reserve_m and not contact_transport):raise ValueError('Invalid transport reserve')
        self.contact_transport=bool(contact_transport);self.transport_reserve_m=transport_reserve_m
        self.attempted=False;self.stable=0;self.requested_end=None
        self.project=Path(__file__).resolve().parents[4]
        self.input_prefix=Path(str(self.out)+'_live_paths')
        self.ready_return=None
        if ready_return:
            from rubik_lab.diagnostics.ready_return import LiveReadyReturn
            self.ready_return=LiveReadyReturn(self)

    @staticmethod
    def validate_audit(report,support_contact=False,expected_end=None,contact_transport=False,contact_lateral_max_m=None):
        if not report.get('results'):raise ValueError('Missing independent audit rows')
        if expected_end is not None:
            if [r['waypoint'] for r in report['results']]!=list(range(expected_end+1)):raise ValueError('Incomplete independent waypoint audit')
            if any(len(r['segment_samples'])!=19 for r in report['results']):raise ValueError('Incomplete independent segment audit')
        if support_contact and not report.get('support_contact_audited'):raise ValueError('Missing independent support audit')
        if contact_transport and not report.get('contact_transport_audited'):raise ValueError('Missing independent transport audit')
        if contact_lateral_max_m is not None:
            if not report.get('contact_patch_audited') or not np.isclose(report.get('contact_lateral_max_m',np.nan),contact_lateral_max_m,rtol=0,atol=1e-12):raise ValueError('Missing or mismatched independent contact patch audit')
        for row in report['results']:
            samples=[dict(cube_clearance=row['cube_clearance'],collisions=row['candidate_collisions'],support_contact=row.get('support_contact'),contact_transport=row.get('contact_transport'),contact_patch=row.get('contact_patch')),*row['segment_samples']]
            for sample_index,sample in enumerate(samples):
                if not sample['cube_clearance']['pass_at_0p1mm']:raise ValueError('Independent exact-cube clearance rejected')
                if support_contact and (not sample.get('support_contact') or not sample['support_contact']['passed']):raise ValueError('Independent support contact rejected')
                if contact_transport:
                    transport=sample.get('contact_transport')
                    if not transport or not transport['passed']:raise ValueError('Independent tangential transport rejected')
                    if row['waypoint']>12 and (not transport.get('active') or not np.isfinite(transport.get('error_m',np.nan)) or abs(transport['error_m'])>.0005):raise ValueError('Missing or invalid active transport audit')
                if contact_lateral_max_m is not None:
                    patch=sample.get('contact_patch')
                    if not patch or not patch['passed']:raise ValueError('Independent contact patch rejected')
                    active=row['waypoint']>=(11 if sample_index==0 else 13)
                    tolerance=.006 if row['waypoint']==11 else .004
                    if active and (not patch.get('active') or not np.isfinite(patch.get('error_m',np.nan)) or patch['error_m']>tolerance):raise ValueError('Missing or invalid contact patch distance')
                for c in sample['collisions']:
                    if c['rigid_pair'] and (c['sat_signed_overlap_m'] is None or not np.isfinite(c['sat_signed_overlap_m']) or c['sat_signed_overlap_m']>.0001):raise ValueError('Independent rigid-hand overlap rejected')

    def _child(self,args,label,timeout=1200):
        env=os.environ.copy();env['OMP_NUM_THREADS']='1';env['OPENBLAS_NUM_THREADS']='1'
        cache=Path(sys.prefix)/'lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages/isaacsim/extscache'
        libs=list(cache.glob('omni.usd.libs-*'))
        if len(libs)!=1:raise RuntimeError('Need unique installed USD runtime for independent audit')
        env['PYTHONPATH']=os.pathsep.join([str(self.project/'source/rubik_lab'),str(libs[0]),env.get('PYTHONPATH','')])
        env['LD_LIBRARY_PATH']=os.pathsep.join([str(libs[0]/'bin'),str(Path(sys.prefix)/'lib'),env.get('LD_LIBRARY_PATH','')])
        log=Path(str(self.out)+'_'+label+'.log')
        with log.open('w') as stream:
            child=subprocess.Popen([sys.executable,'-u','scripts/diagnostics.py',*args],cwd=self.project,env=env,stdout=stream,stderr=subprocess.STDOUT)
            self._event(dict(event='planning_child_started',label=label,pid=child.pid,log=str(log),args=args))
            try:code=child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                child.kill();child.wait();raise RuntimeError(label+' timed out; no prefix execution')
        if code:raise RuntimeError(label+' failed; see '+str(log))

    @property
    def permits_completion(self):return bool(self.ready_return and self.ready_return.permits_completion)

    @property
    def return_targets(self):return self.ready_return.targets if self.ready_return else []

    def capture_arrays(self,e):
        arrays={k:np.asarray(v) for k,v in self.history.items()}
        arrays['env_origins']=e.scene.env_origins.detach().cpu().numpy()
        arrays['actuator_lower']=np.stack([e.finger_lower[s].detach().cpu().numpy() for s in ('left','right')],1)
        arrays['actuator_upper']=np.stack([e.finger_upper[s].detach().cpu().numpy() for s in ('left','right')],1)
        return arrays

    def _capture_input(self,e,step):
        import torch
        from rubik_lab.hand_kinematics import HandFK
        from rubik_lab.flick import FlickPlanner,LAT_MIN,LAT_MAX
        arrays=self.capture_arrays(e)
        frame=len(arrays['trace'])-1;i=self.env
        if frame<0 or int(arrays['trace'][frame,i,1])!=step-1:raise ValueError('Missing exact pre-plan controlframe')
        if int(arrays['trace'][frame,i,0])!=self.trial:raise ValueError('Wrong source trial')
        source=Path(str(self.out)+'_live_source')
        if source.with_suffix('.npz').exists():raise FileExistsError(source)
        source.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(source.with_suffix('.npz'),**arrays)
        source.with_suffix('.json').write_text(json.dumps(dict(control_hz=1/e.step_dt,timeout_per_primitive_s=6.,scope='Actual completed continuationframes before paused planning; no invented state or reset')))
        if self.support_contact:
            forces=arrays['contact'][frame,i,0,1,[5,15]]
            if not np.isfinite(forces).all() or not (forces>.05).all():raise ValueError('Contact model requires measured thumb/middle core support')
        pose=arrays['poses'][frame,i];q=arrays['sensory'][frame,i,22:44]
        cube=Rotation.from_quat(pose[3:7][[1,2,3,0]])
        wp=cube.inv().apply(pose[14:17]-arrays['env_origins'][i]-pose[:3])
        wr=cube.inv().as_matrix()@Rotation.from_quat(pose[17:21][[1,2,3,0]]).as_matrix()
        error=float(arrays['trace'][frame,i,5]);degrees=int(np.ceil(abs(error)))
        self.requested_end=12+min(degrees,self.prefix_degrees)
        fk=HandFK('right','cpu',pts_per_link=8,seed=0)
        for table in (fk.T0,fk.axis,fk.link_pts):
            for key in table:table[key]=table[key].double()
        tensor=lambda x:torch.tensor(x,dtype=torch.float64)
        pad=fk.pads(fk.forward(tensor(wp)[None],tensor(wr)[None],tensor(q)[None]))['index'][0]
        turn=Rotation.from_rotvec([0,0,np.deg2rad(-error)]).as_matrix()
        contact=turn.T@pad[0].numpy();contact[0]=-np.clip(-contact[0],LAT_MIN,LAT_MAX);contact[1]=-(.035-.001);contact[2]=2*.035/3
        approach=FlickPlanner._approach(SimpleNamespace(dev='cpu',n_app=12),pad,tensor(turn@contact)[None],tensor([0,0,1]),tensor(turn@np.array([0,-1,0]))[None])
        targets=np.vstack([approach[0].numpy(),np.stack([Rotation.from_rotvec([0,0,np.deg2rad(-error-d)]).apply(contact) for d in range(degrees+1)])])
        np.savez_compressed(self.input_prefix.with_suffix('.npz'),U_initial_q=q[None],U_wrist_pos=wp[None],U_wrist_rot=wr[None],U_target=targets[None],U_actuator_lower=arrays['actuator_lower'][i,1][None],U_actuator_upper=arrays['actuator_upper'][i,1][None],U_target_bias=(arrays['motor_targets'][frame,i,1].astype(float)-q.astype(float))[None])
        row=dict(trial=self.trial,env=i,source_frame=frame,source_step=step-1,error_deg=error,side='right',move='U',first_failed_waypoint=None)
        self.input_prefix.with_suffix('.json').write_text(json.dumps(dict(source=str(source),surface_contact=True,wrist_assist=True,contact_face='R',contact_press_m=.001,results=[row],scope='Unattempted actual-live source; not a prior failed-pose cohort')))
        return row

    def prepare(self,e,trial,step,active,stage):
        if self.path is not None:
            if self.ready_return is not None and self.ready_return.prepare(e,trial,step,active,stage):return
            return super().prepare(e,trial,step,active,stage)
        e.whole_hand_probe_target=None
        if trial!=self.trial or self.attempted:return
        i=self.env
        if not bool(active[i]) or int(stage[i])!=1:self.stable=0;return
        s=e.snapshot;error=float(s['twist_error'][i]);omega=float(s['twist_velocity'][i])
        eligible=(-np.deg2rad(50)<error<-np.deg2rad(5) and abs(omega)<.25
                  and (self.released_start or float(e.cycle_finger_error[i])<=.2) and float(e.cycle_finger_speed[i])<=1
                  and float(e.cycle_release_force[i])<=.5 and float(e.cycle_step[i])*e.step_dt<5.)
        self.stable=self.stable+1 if eligible else 0
        if self.stable<4:return
        self.attempted=True;clock=e.cycle_step.clone();start=time.monotonic()
        self._event(dict(event='live_planning_started',released_start=self.released_start,source_ready_error_rad=float(e.cycle_finger_error[i]),trial=trial,step=step,env=i,error_deg=np.rad2deg(error),scope='Simulation integration pauses for diagnostic computation; no real-time claim'))
        try:
            self._capture_input(e,step)
            solved=Path(str(self.out)+'_live_ik');audit=Path(str(self.out)+'_live_audit')
            contact_args=['--support-contact','--support-optimizer-reserve-m',str(self.support_reserve_m)] if self.support_contact else []
            if self.contact_transport:contact_args+=['--contact-transport','--transport-optimizer-reserve-m',str(self.transport_reserve_m)]
            if self.contact_lateral_max_m is not None:contact_args+=['--contact-lateral-max-m',str(self.contact_lateral_max_m)]
            if self.contact_face:contact_args+=['--contact-face',self.contact_face]
            if self.hand_pairs_from:contact_args+=['--hand-pairs-from',str(self.hand_pairs_from)]
            self._child(['constrained-ik','--paths',str(self.input_prefix),'--out',str(solved),'--whole-hand','--sequential','--segment-constraints','--cooked-hulls','outputs/diagnostics/hand_physx_cooked_assets','--approach-orientation','--adaptive-segment','--batched-jacobian','--optimizer-clearance-margin-m','0.00005','--prefix-degrees',str(self.prefix_degrees),*contact_args],'live_ik')
            path=WholeHandPath.from_result(solved,self.trial,i,stop_waypoint=self.requested_end)
            self._child(['hand-collision','--solutions',str(solved),'--out',str(audit),'--usd-convex','--cooked-hulls','outputs/diagnostics/hand_physx_cooked_assets','--cube-clearance','--segment-samples','19'],'live_audit',timeout=600)
            report=json.loads(audit.with_suffix('.json').read_text())
            self.validate_audit(report,self.support_contact,self.requested_end,self.contact_transport,self.contact_lateral_max_m)
            if not bool((clock==e.cycle_step).all()):raise RuntimeError('Task clock changed during planning')
            self.path=path
            self._event(dict(event='live_plan_audited',trial=trial,step=step,env=i,planning_wall_seconds=time.monotonic()-start,requested_end=self.requested_end,source=str(solved),audit=str(audit)))
            super().prepare(e,trial,step,active,stage)
        except Exception as error:
            self.checked=True
            self._event(dict(event='live_planning_rejected',trial=trial,step=step,env=i,error=repr(error),planning_wall_seconds=time.monotonic()-start))
