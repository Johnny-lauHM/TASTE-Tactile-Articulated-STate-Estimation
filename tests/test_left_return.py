import unittest
import numpy as np
from rubik_lab.diagnostics.left_return import LeftReturnPath

class LeftReturnTests(unittest.TestCase):
    def path(self):
        p=LeftReturnPath.__new__(LeftReturnPath);p.motor=np.zeros((4,22));p.motor[1:,6]=-.4;p.motor[2:,5]=.3;p.motor[3:,7]=.2
        p.times=np.array([0.,.4,.7,.9]);p.duration=.9;p.source={'env':0};p.started=False
        return p
    def test_sampling_preserves_order_rate_and_endpoint(self):
        p=self.path();previous=p.sample(0)
        for time in np.arange(.025,1.1,.025):
            current=p.sample(time)
            self.assertLessEqual(float(np.max(np.abs(current-previous))),.025000001);previous=current
        np.testing.assert_array_equal(p.sample(5),p.motor[-1,5:9])
        self.assertEqual(p.sample(.2)[0],0);self.assertEqual(p.sample(.5)[2],0)
        with self.assertRaises(ValueError):p.sample(-.1)
    def test_source_guard_rejects_changed_or_missing_history(self):
        p=self.path();p.original={'poses':np.zeros((3,21)),'sensory':np.zeros((3,148))}
        history={k:v[:,None].copy() for k,v in p.original.items()};p.verify_history(history);self.assertTrue(p.started)
        p.started=False;history['poses'][2,0,0]=.00001
        with self.assertRaises(ValueError):p.verify_history(history)
        self.assertFalse(p.started)
        history['poses']=history['poses'][:2]
        with self.assertRaises(ValueError):p.verify_history(history)

class WithdrawalEntryTests(unittest.TestCase):
    def raw(self):
        t=np.zeros((6,1,16));t[:,0,1]=np.arange(6);t[:,0,3]=2;t[:,0,4]=1
        c=np.zeros((6,1,2,2,27));c[:,0,0,:,5]=1
        return dict(trace=t,sensory=np.zeros((6,1,148)),contact=c,core_angular_speed=np.zeros((6,1)))
    def test_angle_core_contact_and_missing_frames_reject_entry(self):
        from rubik_lab.diagnostics.left_return import first_withdrawal_source
        z=self.raw();self.assertEqual(first_withdrawal_source(z,0,0,0),3)
        z['trace'][:,0,5]=-6.3;self.assertIsNone(first_withdrawal_source(z,0,0,0))
        z=self.raw();z['contact'][:,0,0,0,9]=.6;self.assertIsNone(first_withdrawal_source(z,0,0,0))
        z=self.raw();z['trace'][:,0,1]=[0,1,2,4,5,6];self.assertIsNone(first_withdrawal_source(z,0,0,0))
        z=self.raw();z['sensory'][:,0,49]=np.nan;self.assertIsNone(first_withdrawal_source(z,0,0,0))

class ResidualEscapeTests(unittest.TestCase):
    def test_audit_bounds_and_history_protect_motor_only_escape(self):
        import json,tempfile
        from pathlib import Path
        from rubik_lab.diagnostics.left_return import LeftResidualEscapePath
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'escape';source=Path(tmp)/'source';raw=Path(tmp)/'raw'
            t=np.zeros((2,1,16));t[:,0,1]=[0,1]
            z=dict(trace=t,sensory=np.zeros((2,1,148)),poses=np.zeros((2,1,21)),motor_targets=np.full((2,1,2,22),.1),wrist_targets=np.zeros((2,1,2,7)),contact=np.zeros((2,1,2,2,27)),core_angular_speed=np.zeros((2,1)),actuator_lower=np.full((1,2,22),-1.),actuator_upper=np.ones((1,2,22)))
            np.savez(raw.with_suffix('.npz'),**z)
            source.with_suffix('.json').write_text(json.dumps(dict(source=str(raw),results=[dict(side='left',trial=0,env=0,source_frame=1,source_step=1)])))
            p.with_suffix('.json').write_text(json.dumps(dict(source=str(source))))
            x=np.zeros((1,28));y=x.copy();y[0,7]=.02
            np.savez(p.with_suffix('.npz'),initial=x,solution=y)
            sample=dict(cube_clearance={'pass_at_0p1mm':True},support_contact={'passed':True},collisions=[])
            audit=dict(source=str(p),active_side='left',segment_samples=19,cube_clearance=True,support_contact_audited=True,results=[dict(sample,waypoint=0,candidate_collisions=[],segment_samples=[sample]*19)])
            ap=Path(str(p)+'_audit.json');ap.write_text(json.dumps(audit))
            obj=LeftResidualEscapePath(p);np.testing.assert_allclose(obj.sample(1),[.1,.1,.12,.1]);self.assertFalse(obj.started)
            obj.verify_history(z);self.assertTrue(obj.started)
            z['poses'][1,0,0]=1e-6
            with self.assertRaises(ValueError):LeftResidualEscapePath(p).verify_history(z)
            y[0,7]=.03;np.savez(p.with_suffix('.npz'),initial=x,solution=y)
            with self.assertRaises(ValueError):LeftResidualEscapePath(p)
            y[0,7]=.02;np.savez(p.with_suffix('.npz'),initial=x,solution=y)
            audit['results'][0]['segment_samples'][5]['support_contact']['passed']=False;ap.write_text(json.dumps(audit))
            with self.assertRaises(ValueError):LeftResidualEscapePath(p)

class ActualLeftBuilderTests(unittest.TestCase):
    def raw(self):
        z=WithdrawalEntryTests().raw();n=len(z['trace']);z['trace'][:,0,1]=np.arange(20,20+n)
        z['sensory'][:,0,5:9]=[.3,.5,.5,.6]
        z['motor_targets']=np.zeros((n,1,2,22));z['motor_targets'][:,0,0,5:9]=[.3,.5,.5,.6]
        z['poses']=np.zeros((n,1,21));z['poses'][:,:,3]=1;z['poses'][:,:,10]=1;z['poses'][:,:,17]=1
        z['wrist_targets']=np.zeros((n,1,2,7));z['ready_targets']=np.zeros((1,1,2,22));z['env_origins']=np.zeros((1,3))
        z['actuator_lower']=np.full((1,2,22),-2.);z['actuator_upper']=np.full((1,2,22),2.)
        return z
    def test_actual_history_retained_and_original_clock_used(self):
        import tempfile,json
        from pathlib import Path
        from rubik_lab.diagnostics.left_return import write_actual_left_candidate
        z=self.raw();before={k:v.copy() for k,v in z.items()}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'path';r=write_actual_left_candidate(z,p,0,0,'full')
            self.assertAlmostEqual(r['remaining_original_seconds'],5.85)
            with np.load(str(p)+'_source.npz') as saved:
                for k,v in before.items():np.testing.assert_array_equal(saved[k],v);np.testing.assert_array_equal(z[k],v)
            with np.load(str(p)+'.npz') as path:
                np.testing.assert_array_equal(path['solution'][-1,5:9],[0,0,0,0])
                fixed=[j for j in range(28) if j not in range(5,9)]
                np.testing.assert_array_equal(path['solution'][:,fixed],path['initial'][:,fixed])
            with self.assertRaises(FileExistsError):write_actual_left_candidate(z,p,0,0,'full')
    def test_missing_history_and_motor_bounds_reject_before_export(self):
        import tempfile
        from pathlib import Path
        from rubik_lab.diagnostics.left_return import write_actual_left_candidate
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'path';z=self.raw();z['trace'][2,0,1]+=1
            with self.assertRaises(ValueError):write_actual_left_candidate(z,p,0,0,'full')
            self.assertEqual(list(Path(tmp).iterdir()),[])
            z=self.raw();z['actuator_upper'][:]=.1
            with self.assertRaises(ValueError):write_actual_left_candidate(z,p,0,0,'residual')
            self.assertEqual(list(Path(tmp).iterdir()),[])

class LiveLeftGateTests(unittest.TestCase):
    def test_residual_contact_never_credits_full_withdrawal(self):
        from rubik_lab.diagnostics.left_return import next_left_path_kind
        z=WithdrawalEntryTests().raw();z['contact'][:,0,0,0,9]=.6
        self.assertEqual(next_left_path_kind(z,0,0,2),'residual')
        self.assertIsNone(next_left_path_kind(z,0,0,2,residual_attempted=True))
        z['contact'][:,0,0,0,9]=.3
        self.assertEqual(next_left_path_kind(z,0,0,2,residual_attempted=True),'full')
        z['trace'][-1,0,5]=5.01
        self.assertIsNone(next_left_path_kind(z,0,0,2))
    def test_ramp_completion_and_consecutive_recent_frames_required(self):
        from rubik_lab.diagnostics.left_return import next_left_path_kind
        z=WithdrawalEntryTests().raw();self.assertIsNone(next_left_path_kind(z,0,0,3))
        z['trace'][-1,0,1]+=1
        self.assertIsNone(next_left_path_kind(z,0,0,2))
        z=WithdrawalEntryTests().raw();z['sensory'][-1,0,49]=np.nan
        self.assertIsNone(next_left_path_kind(z,0,0,2))

class LiveLeftFailureTests(unittest.TestCase):
    def test_rejected_full_plan_preserves_prior_escape_without_completion_credit(self):
        import torch
        from types import SimpleNamespace
        from rubik_lab.diagnostics.left_return import LiveLeftWithdrawal
        owner=SimpleNamespace(env=0,history={})
        p=LiveLeftWithdrawal(owner);p.rejected=True
        p.escape=SimpleNamespace(source={'source_step':10},duration=.02,sample=lambda t:np.array([.1,.2,.3,.4]))
        e=SimpleNamespace(step_dt=.025,device='cpu',move_id=torch.tensor([0]),flick_cols=torch.tensor([[5,6,7,8]]),finger_tgt={'left':torch.zeros((1,22))},post_return_hold_target=None)
        full,elapsed=p.prepare(e,0,20,5,.1,[])
        self.assertIsNone(full);self.assertEqual(elapsed,0)
        np.testing.assert_allclose(e.post_return_hold_target[3].numpy(),[.1,.2,.3,.4])

class LeftGeometryScopeTests(unittest.TestCase):
    def test_recontact_geometry_does_not_enable_unsupported_contact_models(self):
        import ast
        from pathlib import Path
        p=Path(__file__).resolve().parents[1]/'source/rubik_lab/rubik_lab/diagnostics/hand_collision.py'
        fn=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='validate_active_geometry_scope')
        scope={};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(p),'exec'),scope);check=scope['validate_active_geometry_scope']
        check('left',dict(phase='recontact_geometry',support_contact=True))
        check('left',dict(phase='withdrawal_full'))
        for r in [dict(phase='turn'),dict(phase='recontact_geometry',contact_transport=True),dict(phase='recontact_geometry',contact_lateral_max_m=.02)]:
            with self.assertRaises(ValueError):check('left',r)
