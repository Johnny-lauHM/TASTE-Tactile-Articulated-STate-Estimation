import unittest,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from rubik_lab.diagnostics.ready_return import inner_ready_endpoint,LiveReadyReturn
from rubik_lab.diagnostics.whole_hand_path import WholeHandProbe
import test_whole_hand_path as fixtures

class ReadyReturnTests(unittest.TestCase):
    def test_inner_box_preserves_support_and_motor_origin(self):
        q=np.zeros(22);q[5:9]=[.510047,.10912,.052745,.652913]
        bias=np.full(22,.03);bias[5:9]=[.00606,-.00496,.016413,.005481]
        cmd=q+bias;ready=np.zeros(22);ready[5:9]=[.11307,.326318,.42445,.034908]
        saved=[x.copy() for x in (q,cmd,ready)]
        end,actual_bias,fraction=inner_ready_endpoint(q,cmd,ready,np.full(22,-2.),np.full(22,2.))
        self.assertLess(fraction,1.);self.assertGreater(fraction,.8)
        self.assertLessEqual(np.max(np.abs(end[5:9]-ready[5:9])),.1+1e-12)
        other=[i for i in range(22) if i not in range(5,9)]
        np.testing.assert_array_equal(end[other],q[other]);np.testing.assert_allclose(actual_bias,bias)
        np.testing.assert_allclose(end+actual_bias,cmd+(end-q))
        for x,y in zip((q,cmd,ready),saved):np.testing.assert_array_equal(x,y)

    def test_invalid_domain_and_incompatible_intersections_reject(self):
        q=np.zeros(22);q[5]=1;cmd=q.copy();cmd[5]+=.1;cmd[6]+=1
        ready=np.zeros(22);lo=np.full(22,-2.);hi=np.full(22,2.)
        with self.assertRaises(ValueError):inner_ready_endpoint(q,cmd,ready,lo,hi)
        with self.assertRaises(ValueError):inner_ready_endpoint(q,cmd,ready,lo,np.full(22,.5))
        cmd[0]=np.nan
        with self.assertRaises(ValueError):inner_ready_endpoint(q,cmd,ready,lo,hi)

    def parent_env(self,d):
        helper=fixtures.WholeHandPathTests();path=helper.make();e=helper.fake_env(path)
        e.snapshot.update(twist_error=torch.tensor([.02]),twist_velocity=torch.tensor([.1]),core_angular_speed=torch.tensor([.1]))
        e.cycle_motion_ok=torch.tensor([True]);e.cycle_monitor=SimpleNamespace(failed=torch.tensor([False]),done=torch.tensor([False]))
        events=[]
        parent=SimpleNamespace(started=True,trial=0,env=0,out=Path(d)/'probe',path=path,start_step=0,targets=[],feedback=[],events=events,_event=events.append,capture_arrays=lambda e:{})
        return parent,e,path

    def test_four_frames_then_rejection_never_retries_or_credits(self):
        with tempfile.TemporaryDirectory() as d:
            parent,e,path=self.parent_env(d);ret=LiveReadyReturn(parent);clock=e.cycle_step.clone();q=e.snapshot['finger_pos'].clone()
            with patch('rubik_lab.diagnostics.ready_return.write_candidate',side_effect=ValueError('injected audit preparation rejection')) as build:
                for step in range(1,4):self.assertFalse(ret.prepare(e,0,step,torch.tensor([True]),torch.tensor([1])))
                self.assertFalse(ret.attempted);build.assert_not_called()
                self.assertFalse(ret.prepare(e,0,4,torch.tensor([True]),torch.tensor([1])))
                self.assertTrue(ret.attempted);self.assertFalse(ret.permits_completion)
                self.assertFalse(ret.prepare(e,0,5,torch.tensor([True]),torch.tensor([1])));self.assertEqual(build.call_count,1)
            self.assertTrue(torch.equal(clock,e.cycle_step));self.assertTrue(torch.equal(q,e.snapshot['finger_pos']))

    def test_velocity_or_motion_failure_cannot_trigger_return(self):
        with tempfile.TemporaryDirectory() as d:
            for kind in ('layer','core','motion'):
                parent,e,path=self.parent_env(d);ret=LiveReadyReturn(parent)
                if kind=='layer':e.snapshot['twist_velocity'].fill_(.3)
                if kind=='core':e.snapshot['core_angular_speed'].fill_(1.1)
                if kind=='motion':e.cycle_motion_ok.fill_(False)
                for step in range(1,6):self.assertFalse(ret.prepare(e,0,step,torch.tensor([True]),torch.tensor([1])))
                self.assertFalse(ret.attempted)

    def test_delegation_preserves_global_clock_and_separate_targets(self):
        with tempfile.TemporaryDirectory() as d:
            parent,e,path=self.parent_env(d);e.snapshot['twist_error'].fill_(np.deg2rad(path.source_error_deg))
            parent.start_step=-10;ret=LiveReadyReturn(parent);ret.probe=WholeHandProbe(path,0,0)
            clock=e.cycle_step.clone();q=e.snapshot['finger_pos'].clone()
            self.assertTrue(ret.prepare(e,0,1,torch.tensor([True]),torch.tensor([1])))
            self.assertEqual(len(parent.targets),1);self.assertEqual(len(ret.targets),1)
            self.assertAlmostEqual(parent.targets[0][3],.3);self.assertAlmostEqual(ret.targets[0][3],.025)
            self.assertFalse(ret.permits_completion);self.assertEqual(parent.feedback,[])
            final_step=int(np.ceil(path.duration/e.step_dt))
            ret.prepare(e,0,final_step,torch.tensor([True]),torch.tensor([1]))
            self.assertTrue(ret.permits_completion)
            self.assertFalse(bool(e.cycle_monitor.done[0])) # Permission never sets physical success.
            self.assertTrue(torch.equal(clock,e.cycle_step));self.assertTrue(torch.equal(q,e.snapshot['finger_pos']))

    def test_audited_path_guard_records_separate_return_events(self):
        from dataclasses import replace
        import json
        from rubik_lab.diagnostics.live_whole_hand import LiveWholeHandProbe
        with tempfile.TemporaryDirectory() as d:
            parent,e,path=self.parent_env(d);ret=LiveReadyReturn(parent)
            path=replace(path,source_step=3,source_error_deg=2.)
            e.snapshot['twist_error'].fill_(np.deg2rad(2.))
            sample=dict(cube_clearance={'pass_at_0p1mm':True},support_contact={'passed':True},collisions=[])
            audit=dict(support_contact_audited=True,results=[dict(waypoint=0,cube_clearance=sample['cube_clearance'],support_contact=sample['support_contact'],candidate_collisions=[],segment_samples=[dict(sample) for _ in range(19)])])
            def child(args,*rest,**kwargs):
                prefix=Path(args[args.index('--out')+1]);prefix.with_suffix('.json').write_text(json.dumps(audit))
            parent._child=child;parent.validate_audit=LiveWholeHandProbe.validate_audit
            candidate=dict(results=[dict(measured_feasible=False,segment_pass=False)],duration_seconds=path.duration,remaining_original_seconds=5.)
            with patch('rubik_lab.diagnostics.ready_return.write_candidate',return_value=candidate),patch('rubik_lab.diagnostics.ready_return.WholeHandPath.from_result',return_value=path):
                for step in range(1,4):ret.prepare(e,0,step,torch.tensor([True]),torch.tensor([1]))
                self.assertTrue(ret.prepare(e,0,4,torch.tensor([True]),torch.tensor([1])))
            names=[v['event'] for v in parent.events]
            self.assertIn('return_plan_audited',names);self.assertIn('return_start_accepted',names);self.assertNotIn('start_accepted',names)
            self.assertFalse(ret.permits_completion);self.assertFalse(ret.probe.index_feedback)
            self.assertFalse(bool(e.cycle_monitor.done[0]))


    def test_refinement_requires_second_independent_audit(self):
        import json
        from dataclasses import replace
        from rubik_lab.diagnostics.live_whole_hand import LiveWholeHandProbe
        for final_collision in (False,True):
            with tempfile.TemporaryDirectory() as d:
                parent,e,path=self.parent_env(d);ret=LiveReadyReturn(parent)
                path=replace(path,source_step=3,source_error_deg=2.)
                e.snapshot['twist_error'].fill_(np.deg2rad(2.));calls=[]
                candidate=dict(results=[dict(measured_feasible=False,segment_pass=False)],duration_seconds=path.duration,remaining_original_seconds=5.)
                def child(args,*rest,**kwargs):
                    calls.append(args[0]);out=Path(args[args.index('--out')+1])
                    if args[0]=='ready-return-geometry':
                        out.with_suffix('.json').write_text(json.dumps(candidate));return
                    collision=final_collision or not str(out).endswith('_constrained_audit')
                    c=[dict(rigid_pair=True,sat_signed_overlap_m=.0005)] if collision else []
                    sample=dict(cube_clearance={'pass_at_0p1mm':True},support_contact={'passed':True},collisions=c)
                    audit=dict(support_contact_audited=True,results=[dict(waypoint=0,cube_clearance=sample['cube_clearance'],support_contact=sample['support_contact'],candidate_collisions=c,segment_samples=[dict(sample) for _ in range(19)])])
                    out.with_suffix('.json').write_text(json.dumps(audit))
                parent._child=child;parent.validate_audit=LiveWholeHandProbe.validate_audit
                clock=e.cycle_step.clone()
                with patch('rubik_lab.diagnostics.ready_return.write_candidate',return_value=candidate),patch('rubik_lab.diagnostics.ready_return.WholeHandPath.from_result',return_value=path):
                    for step in range(1,4):ret.prepare(e,0,step,torch.tensor([True]),torch.tensor([1]))
                    accepted=ret.prepare(e,0,4,torch.tensor([True]),torch.tensor([1]))
                self.assertEqual(calls,['hand-collision','ready-return-geometry','hand-collision'])
                self.assertEqual(accepted,not final_collision);self.assertFalse(ret.permits_completion)
                self.assertTrue(torch.equal(clock,e.cycle_step));self.assertFalse(bool(e.cycle_monitor.done[0]))
                names=[x['event'] for x in parent.events]
                self.assertIn('return_geometry_refinement',names)
                self.assertIn('return_planning_rejected' if final_collision else 'return_start_accepted',names)

if __name__=='__main__':unittest.main()
