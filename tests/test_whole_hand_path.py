import unittest
import pytest
import numpy as np
from scipy.spatial.transform import Rotation
from rubik_lab.diagnostics.whole_hand_path import WholeHandPath

class WholeHandPathTests(unittest.TestCase):
    def make(self):
        x=np.zeros((3,28));x[1,:22]=.2;x[2,:22]=-.1
        x[1,22:25]=[.2,-.1,.1];x[2,22:25]=[-.2,.3,.1]
        x[1,25:]=[.1,.2,-.1];x[2,25:]=[-.1,.1,.3]
        cube=Rotation.from_euler('xyz',[.2,-.3,.7]);q=cube.as_quat()[[3,0,1,2]]
        wrist=Rotation.from_euler('xyz',[.5,.1,-.2]).as_matrix()
        return WholeHandPath(x,np.array([.08,-.02,.03]),wrist,np.array([.4,.2,.1]),q,.008,.12,5.,-20.)

    def test_fixed_reference_and_noncommuting_rotation(self):
        p=self.make();before=p.sample(0.)
        result=p.sample(p.duration)
        cube=Rotation.from_quat(p.cube_quaternion[[1,2,3,0]])
        expected=p.cube_position+cube.apply(p.wrist_position_cube+p.variables[-1,22:25]*p.translation_scale)
        np.testing.assert_allclose(result['wrist_position'],expected,atol=1e-14)
        expected_R=cube.as_matrix()@Rotation.from_rotvec(p.variables[-1,25:]*p.rotation_scale).as_matrix()@p.wrist_rotation_cube
        actual_R=Rotation.from_quat(result['wrist_quaternion'][[1,2,3,0]]).as_matrix()
        np.testing.assert_allclose(actual_R,expected_R,atol=1e-14)
        np.testing.assert_array_equal(p.sample(0.)['wrist_position'],before['wrist_position'])
        with self.assertRaises(ValueError):p.variables[0,0]=1
        result['fingers'][0]=99
        self.assertNotEqual(p.sample(p.duration)['fingers'][0],99)

    def test_sampled_motion_respects_declared_rates(self):
        p=self.make();times=np.linspace(0,p.duration,2001);prev=p.sample(0.)
        for t0,t in zip(times[:-1],times[1:]):
            cur=p.sample(t);dt=t-t0
            self.assertLessEqual(np.max(np.abs(cur['fingers']-prev['fingers']))/dt,p.finger_rate+1e-10)
            self.assertLessEqual(np.linalg.norm(cur['wrist_position']-prev['wrist_position'])/dt,p.translation_rate+1e-10)
            a=Rotation.from_quat(cur['wrist_quaternion'][[1,2,3,0]]);b=Rotation.from_quat(prev['wrist_quaternion'][[1,2,3,0]])
            self.assertLessEqual((a*b.inv()).magnitude()/dt,p.rotation_rate+1e-10)
            prev=cur
        self.assertTrue(p.sample(p.duration+1)['path_finished'])
        with self.assertRaises(ValueError):p.sample(-.1)

    def test_start_mismatch_is_reported_not_aligned(self):
        p=self.make();s=p.sample(0.)
        errors=p.start_errors(s['fingers']+.02,s['wrist_position']+[.001,0,0],s['wrist_quaternion'],p.cube_position,p.cube_quaternion,-19.)
        self.assertAlmostEqual(errors['finger_max_rad'],.02)
        self.assertAlmostEqual(errors['wrist_position_m'],.001)
        self.assertEqual(errors['layer_error_difference_deg'],1.)
        np.testing.assert_array_equal(p.sample(0.)['wrist_position'],s['wrist_position'])


    def fake_env(self,p):
        import torch
        from types import SimpleNamespace
        s=p.sample(0.);q=torch.zeros(1,2,22,dtype=torch.float64);q[0,1]=torch.tensor(s['fingers'])
        pose=torch.zeros(1,2,7,dtype=torch.float64);pose[0,1,:3]=torch.tensor(s['wrist_position']);pose[0,1,3:]=torch.tensor(s['wrist_quaternion'])
        return SimpleNamespace(snapshot=dict(finger_pos=q,ee_pose=pose,cube_pos=torch.tensor(p.cube_position.copy())[None],cube_quat=torch.tensor(p.cube_quaternion.copy())[None],twist_error=torch.tensor([np.deg2rad(p.source_error_deg)])),scene=SimpleNamespace(env_origins=torch.zeros(1,3,dtype=torch.float64)),cycle_step=torch.tensor([40]),step_dt=.025,finger_tgt={'right':q[:,1].clone()},ee_pos_tgt={'right':pose[:,1,:3].clone()},ee_quat_tgt={'right':pose[:,1,3:].clone()},finger_lower={'right':torch.full((1,22),-10.)},finger_upper={'right':torch.full((1,22),10.)})

    def test_probe_matching_start_only_generates_targets(self):
        import torch
        from rubik_lab.diagnostics.whole_hand_path import WholeHandProbe
        p=self.make();e=self.fake_env(p);probe=WholeHandProbe(p,0,0)
        before=e.snapshot['finger_pos'].clone();clock=e.cycle_step.clone()
        probe.prepare(e,0,1,torch.tensor([True]),torch.tensor([1]))
        self.assertTrue(probe.started);self.assertIsNotNone(e.whole_hand_probe_target)
        self.assertTrue(torch.equal(before,e.snapshot['finger_pos']))
        self.assertTrue(torch.equal(clock,e.cycle_step))
        probe.prepare(e,0,2,torch.tensor([False]),torch.tensor([1]))
        self.assertIsNone(e.whole_hand_probe_target)

    def test_probe_rejects_pose_or_clock_mismatch(self):
        import torch
        from rubik_lab.diagnostics.whole_hand_path import WholeHandProbe
        for mismatch in ('joint','clock'):
            p=self.make();e=self.fake_env(p);probe=WholeHandProbe(p,0,0)
            if mismatch=='joint':e.snapshot['finger_pos'][0,1,0]+=.1
            else:e.cycle_step+=1
            probe.prepare(e,0,1,torch.tensor([True]),torch.tensor([1]))
            self.assertFalse(probe.started);self.assertIsNone(e.whole_hand_probe_target)
            self.assertEqual(len(probe.targets),0)


    def test_servo_bias_retains_command_origin_without_state_write(self):
        import torch
        from rubik_lab.diagnostics.whole_hand_path import WholeHandProbe
        p=self.make();e=self.fake_env(p);e.finger_tgt['right']+=.03
        e.ee_pos_tgt['right']+=torch.tensor([.001,0.,0.])
        before=e.snapshot['finger_pos'].clone();probe=WholeHandProbe(p,0,0,preserve_servo_bias=True)
        probe.prepare(e,0,1,torch.tensor([True]),torch.tensor([1]))
        self.assertTrue(probe.started)
        target=e.whole_hand_probe_target[1];raw=p.sample(.025)
        np.testing.assert_allclose(target['fingers'],raw['fingers']+.03,atol=1e-12)
        np.testing.assert_allclose(target['wrist_position'],raw['wrist_position']+[.001,0,0],atol=1e-9)
        self.assertTrue(torch.equal(before,e.snapshot['finger_pos']))

    def test_future_target_limit_violation_rejects_before_execution(self):
        import torch
        from rubik_lab.diagnostics.whole_hand_path import WholeHandProbe
        p=self.make();e=self.fake_env(p);e.finger_upper['right'].fill_(.15)
        probe=WholeHandProbe(p,0,0,preserve_servo_bias=True)
        probe.prepare(e,0,1,torch.tensor([True]),torch.tensor([1]))
        self.assertFalse(probe.started);self.assertFalse(probe.events[-1]['command_bounds_ok'])
        self.assertIsNone(e.whole_hand_probe_target)


    def test_actuator_bounds_account_for_preload_without_relaxing_hard_limits(self):
        from rubik_lab.diagnostics.whole_hand_path import actuator_compatible_bounds
        hl=np.full(22,-1.);hu=np.ones(22);cl=np.full(22,-.8);cu=np.full(22,.8);bias=np.linspace(-.3,.3,22)
        lo,hi=actuator_compatible_bounds(hl,hu,cl,cu,bias)
        self.assertTrue((lo>=hl).all() and (hi<=hu).all())
        self.assertTrue((lo+bias>=cl-1e-15).all() and (hi+bias<=cu+1e-15).all())
        with self.assertRaises(ValueError):actuator_compatible_bounds(hl,hu,cl,cu,np.full(22,3.))


    def test_live_probe_requires_four_eligible_frames_and_attempts_once(self):
        import torch,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from rubik_lab.diagnostics.live_whole_hand import LiveWholeHandProbe
        e=self.fake_env(self.make());e.snapshot['twist_velocity']=torch.tensor([0.])
        e.cycle_finger_error=torch.tensor([.1]);e.cycle_finger_speed=torch.tensor([.2]);e.cycle_release_force=torch.tensor([.1])
        with tempfile.TemporaryDirectory() as d:
            p=LiveWholeHandProbe(0,0,Path(d)/'probe',{})
            clock=e.cycle_step.clone()
            with patch.object(p,'_capture_input',side_effect=RuntimeError('injected test rejection')) as capture:
                for step in range(1,4):p.prepare(e,0,step,torch.tensor([True]),torch.tensor([1]))
                self.assertFalse(p.attempted);capture.assert_not_called()
                p.prepare(e,0,4,torch.tensor([True]),torch.tensor([1]))
                self.assertTrue(p.attempted);self.assertFalse(p.started)
                p.prepare(e,0,5,torch.tensor([True]),torch.tensor([1]))
                self.assertEqual(capture.call_count,1)
                self.assertTrue(torch.equal(clock,e.cycle_step));self.assertIsNone(e.whole_hand_probe_target)

    def test_live_probe_does_not_select_overturn_as_underturn(self):
        import torch,tempfile
        from pathlib import Path
        from rubik_lab.diagnostics.live_whole_hand import LiveWholeHandProbe
        e=self.fake_env(self.make());e.snapshot['twist_velocity']=torch.tensor([0.]);e.snapshot['twist_error']=torch.tensor([.2])
        e.cycle_finger_error=torch.tensor([0.]);e.cycle_finger_speed=torch.tensor([0.]);e.cycle_release_force=torch.tensor([0.])
        with tempfile.TemporaryDirectory() as d:
            p=LiveWholeHandProbe(0,0,Path(d)/'probe',{})
            for step in range(10):p.prepare(e,0,step,torch.tensor([True]),torch.tensor([1]))
            self.assertFalse(p.attempted);self.assertEqual(p.events,[])

    def test_live_contact_pair_configuration_and_child_rejection(self):
        import torch,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from rubik_lab.diagnostics.live_whole_hand import LiveWholeHandProbe
        e=self.fake_env(self.make());e.snapshot['twist_velocity']=torch.tensor([0.])
        e.cycle_finger_error=torch.tensor([.1]);e.cycle_finger_speed=torch.tensor([.2]);e.cycle_release_force=torch.tensor([.1])
        before=e.snapshot['finger_pos'].clone();clock=e.cycle_step.clone()
        with tempfile.TemporaryDirectory() as d:
            probe=LiveWholeHandProbe(0,0,Path(d)/'probe',{},support_contact=True,support_reserve_m=5e-6,contact_face='B',hand_pairs_from='audited_pairs.json',contact_transport=True,transport_reserve_m=.00005,contact_lateral_max_m=.020)
            with patch.object(probe,'_capture_input',return_value={}),patch.object(probe,'_child',side_effect=RuntimeError('injected child failure')) as child:
                for step in range(1,5):probe.prepare(e,0,step,torch.tensor([True]),torch.tensor([1]))
                args=child.call_args.args[0]
                self.assertEqual(args[args.index('--contact-face')+1],'B')
                self.assertIn('--support-contact',args)
                self.assertIn('--contact-transport',args)
                self.assertEqual(float(args[args.index('--contact-lateral-max-m')+1]),.020)
                self.assertEqual(float(args[args.index('--transport-optimizer-reserve-m')+1]),.00005)
                self.assertEqual(args[args.index('--hand-pairs-from')+1],'audited_pairs.json')
                self.assertFalse(probe.started);self.assertIsNone(e.whole_hand_probe_target)
                self.assertTrue(torch.equal(clock,e.cycle_step));self.assertTrue(torch.equal(before,e.snapshot['finger_pos']))

    def test_released_start_only_changes_entry_position_gate(self):
        import torch,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from rubik_lab.diagnostics.live_whole_hand import LiveWholeHandProbe
        for enabled,force,expected in [(False,.1,False),(True,.1,True),(True,.6,False)]:
            e=self.fake_env(self.make());e.snapshot['twist_velocity']=torch.tensor([0.])
            e.cycle_finger_error=torch.tensor([.21]);e.cycle_finger_speed=torch.tensor([.2]);e.cycle_release_force=torch.tensor([force])
            with tempfile.TemporaryDirectory() as d:
                probe=LiveWholeHandProbe(0,0,Path(d)/'probe',{},released_start=enabled)
                with patch.object(probe,'_capture_input',side_effect=RuntimeError('teststop')):
                    for step in range(1,5):probe.prepare(e,0,step,torch.tensor([True]),torch.tensor([1]))
                self.assertEqual(probe.attempted,expected);self.assertFalse(probe.started)
                self.assertAlmostEqual(float(e.cycle_finger_error[0]),.21,places=6)

class IndexFeedbackTests(unittest.TestCase):
    def test_rate_bounds_and_no_integral_windup(self):
        from rubik_lab.diagnostics.whole_hand_path import bounded_index_feedback
        z=np.zeros(4);lo=np.full(4,-.02);hi=np.full(4,.02)
        command,requested,applied,limited=bounded_index_feedback(np.ones(4),z,z,z,lo,hi,.01)
        np.testing.assert_allclose(command,.01);self.assertTrue(limited.all())
        command,*_=bounded_index_feedback(np.ones(4),z,z,command,lo,hi,.025)
        np.testing.assert_allclose(command,.02)
        command,*_=bounded_index_feedback(z,z,z,command,lo,hi,.025)
        np.testing.assert_allclose(command,0.)
        with self.assertRaises(ValueError):bounded_index_feedback(z,z,z,z,lo,hi,0)
        with self.assertRaises(ValueError):bounded_index_feedback(z,np.full(4,np.nan),z,z,lo,hi,.025)

    def test_hold_only_index_feedback_does_not_write_state(self):
        import torch
        from rubik_lab.diagnostics.whole_hand_path import WholeHandProbe
        helper=WholeHandPathTests();path=helper.make();e=helper.fake_env(path)
        p=WholeHandProbe(path,0,0,index_feedback=True)
        q=e.snapshot['finger_pos'].clone();clock=e.cycle_step.clone()
        p.prepare(e,0,1,torch.tensor([True]),torch.tensor([1]));self.assertEqual(p.feedback,[])
        e.finger_tgt['right'][0]=torch.tensor(p.targets[-1][4:26])
        step=int(np.ceil((path.duration+.25)/e.step_dt))
        p.prepare(e,0,step,torch.tensor([True]),torch.tensor([1]))
        self.assertEqual(len(p.feedback),1);self.assertEqual(len(p.feedback[0]),23)
        nominal=path.sample(step*e.step_dt)['fingers'];actual=e.whole_hand_probe_target[1]['fingers']
        other=[i for i in range(22) if i not in range(5,9)]
        np.testing.assert_allclose(actual[other],nominal[other])
        self.assertTrue(torch.equal(q,e.snapshot['finger_pos']));self.assertTrue(torch.equal(clock,e.cycle_step))

class ContactTargetTests(unittest.TestCase):
    def test_both_faces_keep_u_torque_and_endpoint(self):
        from rubik_lab.diagnostics.whole_hand_path import contact_approach_targets
        from scipy.spatial.transform import Rotation
        for face,normal in [('R',np.array([0.,-1.,0.])),('B',np.array([1.,0.,0.]))]:
            targets=contact_approach_targets([.08,-.06,.01],-6.7,face)
            self.assertEqual(targets.shape,(20,3))
            np.testing.assert_allclose(targets[11],targets[12],atol=1e-12)
            for k,p in enumerate(targets[12:]):
                turn=Rotation.from_rotvec([0,0,np.deg2rad(6.7-k)])
                local=turn.inv().apply(p)
                self.assertAlmostEqual(float(local@normal),.034)
                self.assertLess(float(np.cross(local,-normal)[2]),-.0189)
                self.assertAlmostEqual(local[2],.07/3)

    def test_narrow_contact_corridor_preserves_torque_and_rejects_widening(self):
        from rubik_lab.diagnostics.whole_hand_path import contact_approach_targets
        from scipy.spatial.transform import Rotation
        targets=contact_approach_targets([.08,-.06,.01],-6.7,'B',lateral_max_m=.020)
        for k,p in enumerate(targets[12:]):
            local=Rotation.from_rotvec([0,0,np.deg2rad(6.7-k)]).inv().apply(p)
            self.assertAlmostEqual(local[1],-.020);self.assertAlmostEqual(local[0],.034)
            self.assertLess(np.cross(local,[-1.,0.,0.])[2],0.)
        with self.assertRaises(ValueError):contact_approach_targets([.08,-.06,.01],-6.7,'B',lateral_max_m=.029)

    def test_independent_patch_distance_rejects_corner_lane(self):
        from rubik_lab.diagnostics.whole_hand_path import side_patch_error
        from scipy.spatial.transform import Rotation
        for face,point in [('B',[.034,-.027,.023]),('R',[-.027,-.034,.023])]:
            for angle in [0.,.3,-.2]:
                rotated=Rotation.from_rotvec([0,0,angle]).apply(point)
                wide=side_patch_error(rotated,angle,face,.028)
                narrow=side_patch_error(rotated,angle,face,.020)
                self.assertAlmostEqual(wide['error_m'],0.)
                self.assertAlmostEqual(narrow['error_m'],.007)
                self.assertGreater(narrow['error_m'],.004)

    def test_invalid_direction_rejected(self):
        from rubik_lab.diagnostics.whole_hand_path import contact_approach_targets
        with self.assertRaises(ValueError):contact_approach_targets([.08,-.06,.01],6.7,'B')

class ContactTransportTests(unittest.TestCase):
    def test_stationary_pad_cannot_count_as_seven_degree_turn(self):
        import torch
        from rubik_lab.diagnostics.whole_hand_path import tangential_transport
        from scipy.spatial.transform import Rotation
        t=lambda x:torch.tensor(x,dtype=torch.float64)
        anchor=t([.04,-.027,.02]);rotation=t(Rotation.from_euler('z',-7,degrees=True).as_matrix());initial=t(np.eye(3));tangent=t([0.,1.,0.])
        margin,error,target=tangential_transport(anchor,rotation,initial,anchor,tangent)
        self.assertLess(float(margin),0);self.assertGreater(abs(float(error)),.004)
        margin,error,_=tangential_transport(target,rotation,initial,anchor,tangent)
        self.assertAlmostEqual(float(error),0.);self.assertAlmostEqual(float(margin),1.)

    def test_transport_optimization_reserve_does_not_change_readout(self):
        import torch
        from rubik_lab.diagnostics.whole_hand_path import tangential_transport
        t=lambda x:torch.tensor(x,dtype=torch.float64)
        anchor=t([.04,-.027,.02]);eye=t(np.eye(3));tangent=t([0.,1.,0.]);pad=anchor+t([0.,.00047,0.])
        self.assertGreater(float(tangential_transport(pad,eye,eye,anchor,tangent,.0005)[0]),0)
        self.assertLess(float(tangential_transport(pad,eye,eye,anchor,tangent,.00045)[0]),0)

    def test_tangent_band_does_not_replace_normal_clearance(self):
        import torch
        from rubik_lab.diagnostics.whole_hand_path import tangential_transport
        t=lambda x:torch.tensor(x,dtype=torch.float64)
        anchor=t([.04,-.027,.02]);eye=t(np.eye(3));tangent=t([0.,1.,0.])
        self.assertGreaterEqual(float(tangential_transport(anchor+t([.003,0.,0.]),eye,eye,anchor,tangent)[0]),0)
        self.assertLess(float(tangential_transport(anchor+t([0.,.0006,0.]),eye,eye,anchor,tangent)[0]),0)
        with self.assertRaises(ValueError):tangential_transport(anchor,eye,eye,anchor,tangent,0)

class IndependentAuditTests(unittest.TestCase):
    def report(self):
        return {'support_contact_audited':True,'results':[{'cube_clearance':{'pass_at_0p1mm':True},'support_contact':{'passed':True},'candidate_collisions':[], 'segment_samples':[{'cube_clearance':{'pass_at_0p1mm':True},'support_contact':{'passed':True},'collisions':[]}]}]}

    def test_support_violation_at_interior_rejects(self):
        from rubik_lab.diagnostics.live_whole_hand import LiveWholeHandProbe
        r=self.report();LiveWholeHandProbe.validate_audit(r,True)
        r['results'][0]['segment_samples'][0]['support_contact']['passed']=False
        with self.assertRaises(ValueError):LiveWholeHandProbe.validate_audit(r,True)

    def test_transport_missing_interior_or_false_pass_rejected(self):
        from copy import deepcopy
        from rubik_lab.diagnostics.live_whole_hand import LiveWholeHandProbe
        r=self.report();r['contact_transport_audited']=True
        row=r['results'][0];row['waypoint']=13
        row['contact_transport']={'active':True,'passed':True,'error_m':.0004}
        row['segment_samples'][0]['contact_transport']=dict(row['contact_transport'])
        LiveWholeHandProbe.validate_audit(r,True,contact_transport=True)
        bad=deepcopy(r);del bad['contact_transport_audited']
        with self.assertRaises(ValueError):LiveWholeHandProbe.validate_audit(bad,True,contact_transport=True)
        for invalid in [None,{'passed':False},{'passed':True,'active':False},{'passed':True,'active':True,'error_m':.0006},{'passed':True,'active':True,'error_m':float('nan')}]:
            bad=deepcopy(r);bad['results'][0]['segment_samples'][0]['contact_transport']=invalid
            with self.assertRaises(ValueError):LiveWholeHandProbe.validate_audit(bad,True,contact_transport=True)

    def test_contact_corridor_mismatch_and_interior_failure_reject(self):
        from copy import deepcopy
        from rubik_lab.diagnostics.live_whole_hand import LiveWholeHandProbe
        r=self.report();r.update(contact_patch_audited=True,contact_lateral_max_m=.020)
        row=r['results'][0];row['waypoint']=13
        row['contact_patch']={'active':True,'passed':True,'error_m':.003}
        row['segment_samples'][0]['contact_patch']=dict(row['contact_patch'])
        LiveWholeHandProbe.validate_audit(r,True,contact_lateral_max_m=.020)
        with self.assertRaises(ValueError):LiveWholeHandProbe.validate_audit(r,True,contact_lateral_max_m=.028)
        for patch in [None,{'active':False,'passed':True},{'active':True,'passed':True,'error_m':.005}]:
            bad=deepcopy(r);bad['results'][0]['segment_samples'][0]['contact_patch']=patch
            with self.assertRaises(ValueError):LiveWholeHandProbe.validate_audit(bad,True,contact_lateral_max_m=.020)

    def test_approach_and_contact_patch_limits_remain_distinct(self):
        from rubik_lab.diagnostics.live_whole_hand import LiveWholeHandProbe
        r=self.report();r.update(contact_patch_audited=True,contact_lateral_max_m=.020)
        row=r['results'][0];row['waypoint']=11
        row['contact_patch']={'active':True,'passed':True,'error_m':.005}
        row['segment_samples'][0]['contact_patch']={'active':False,'passed':True}
        LiveWholeHandProbe.validate_audit(r,True,contact_lateral_max_m=.020)
        row['waypoint']=12
        with self.assertRaises(ValueError):LiveWholeHandProbe.validate_audit(r,True,contact_lateral_max_m=.020)
        row['contact_patch']['error_m']=.003
        LiveWholeHandProbe.validate_audit(r,True,contact_lateral_max_m=.020)
        row['waypoint']=13
        with self.assertRaises(ValueError):LiveWholeHandProbe.validate_audit(r,True,contact_lateral_max_m=.020)

    def test_incomplete_audit_rejected(self):
        from rubik_lab.diagnostics.live_whole_hand import LiveWholeHandProbe
        r=self.report();r['results'][0]['waypoint']=0
        with self.assertRaises(ValueError):LiveWholeHandProbe.validate_audit(r,True,expected_end=1)
        with self.assertRaises(ValueError):LiveWholeHandProbe.validate_audit(r,True,expected_end=0)

    def test_missing_support_or_nonfinite_collision_rejects(self):
        from rubik_lab.diagnostics.live_whole_hand import LiveWholeHandProbe
        r=self.report();del r['support_contact_audited']
        with self.assertRaises(ValueError):LiveWholeHandProbe.validate_audit(r,True)
        r=self.report();r['results'][0]['candidate_collisions']=[{'rigid_pair':True,'sat_signed_overlap_m':float('nan')}]
        with self.assertRaises(ValueError):LiveWholeHandProbe.validate_audit(r,False)

if __name__=='__main__':unittest.main()


def test_gain3_feedback_preserves_correction_and_rate_limits():
    from rubik_lab.diagnostics.whole_hand_path import bounded_index_feedback
    base=np.full(4,.5)
    command,requested,applied,limited=bounded_index_feedback(np.array([.02,.2,-.2,0.]),np.zeros(4),base,base,np.zeros(4),np.ones(4),.025,gain=3.)
    assert np.allclose(requested,[.06,.6,-.6,0.])
    assert np.allclose(command,[.525,.525,.475,.5])
    assert np.max(np.abs(applied))<=.025000000001

@pytest.mark.parametrize('gain',[0.,2.,4.,float('nan')])
def test_unregistered_feedback_gain_rejected(gain):
    from rubik_lab.diagnostics.whole_hand_path import bounded_index_feedback
    with pytest.raises(ValueError,match='Unregistered feedback gain'):
        bounded_index_feedback(np.zeros(4),np.zeros(4),np.zeros(4),np.zeros(4),-np.ones(4),np.ones(4),.025,gain=gain)
