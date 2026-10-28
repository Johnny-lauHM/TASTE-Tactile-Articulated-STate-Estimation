import unittest
import numpy as np
from rubik_lab.diagnostics.probe_readout import summarize_probe


class ProbeReadoutTests(unittest.TestCase):
    def inputs(self):
        report={'control_hz':40,'whole_hand_probe':{'events':[{'event':'start_accepted','trial':0,'env':0,'step':1},{'event':'prefix_targets_finished_holding_endpoint','step':2}]}}
        trace=np.zeros((3,1,16));trace[:,0,1]=[0,1,2];trace[:,0,5]=[-7,-4,-3]
        targets=np.zeros((2,33));targets[:,1]=[1,2];targets[:,3]=[.025,.05];targets[:,29]=1
        wrist=np.zeros((3,1,2,7));wrist[:,:,:,3]=1
        contacts=np.zeros((3,1,2,2,27));contacts[:,0,0,0,5]=1;contacts[:,0,0,1,15]=1;contacts[1:,0,1,1,10]=2
        raw={'trace':trace,'whole_hand_targets':targets,'motor_targets':np.zeros((3,1,2,22)),'wrist_targets':wrist,'contact':contacts}
        return report,raw

    def test_requested_and_measured_motion_are_separate(self):
        report,raw=self.inputs();result=summarize_probe(report,raw,(.025,.05))
        self.assertTrue(result['confirmed_target_delivery']);self.assertEqual(result['matched_actuator_frames'],2)
        self.assertEqual(result['final_desired_progress_deg'],4)
        self.assertAlmostEqual(result['active_pad_top_impulse_Ns'],.1)
        self.assertEqual(result['per_hand_contact_missing_frames'],[0,0])
        self.assertEqual(result['per_planned_phase']['approach']['measured_progress_deg'],3)
        self.assertEqual(result['per_planned_phase']['nominal_turn']['measured_progress_deg'],1)

    def test_baseline_motion_without_accepted_probe_not_credited(self):
        report,raw=self.inputs();report['whole_hand_probe']['events']=[]
        self.assertFalse(summarize_probe(report,raw)['confirmed_target_delivery'])

    def test_requests_without_matching_applied_targets_not_credited(self):
        report,raw=self.inputs();raw['motor_targets']+=.1
        result=summarize_probe(report,raw)
        self.assertFalse(result['confirmed_target_delivery']);self.assertEqual(result['matched_actuator_frames'],0)

    def test_nonfinite_contact_rejected(self):
        report,raw=self.inputs();raw['contact'][1,0,1,1,10]=np.nan
        with self.assertRaises(ValueError):summarize_probe(report,raw)

class WrenchReadoutTests(unittest.TestCase):
    def test_opposing_normal_and_friction_moments(self):
        from rubik_lab.diagnostics.probe_readout import summarize_wrenches
        frame=np.zeros((1,19));frame[0,:3]=[0,20,7];frame[0,8]=1;frame[0,14:16]=1
        normal=np.array([[0,20,7,37,2,.035,-.02,0,-1,0,0,0]],dtype=float)
        friction=np.array([[0,20,7,37,0,.04/.035,0,.035,-.02,0]],dtype=float)
        targets=np.zeros((1,33));targets[0,:3]=[0,20,7]
        raw={'probe_wrench_frames':frame,'probe_normal_contacts':normal,'probe_friction_contacts':friction,'whole_hand_targets':targets}
        report={'control_hz':40,'whole_hand_probe':{'events':[{'event':'prefix_targets_finished_holding_endpoint','step':10}]}}
        result=summarize_wrenches(report,raw)['groups']['endpoint_hold_after_0p25s']
        self.assertAlmostEqual(result['normal_torque_mean_Nm'],-.04)
        self.assertAlmostEqual(result['friction_torque_mean_Nm'],.04)
        self.assertAlmostEqual(result['contact_torque_mean_Nm'],0.)
        raw['whole_hand_targets']=np.empty((0,33))
        self.assertEqual(summarize_wrenches(report,raw)['groups']['endpoint_hold_after_0p25s']['frames'],0)
        frame[0,14]=2
        with self.assertRaises(ValueError):summarize_wrenches(report,raw)

class FeedbackReadoutTests(unittest.TestCase):
    def inputs(self):
        from types import SimpleNamespace
        trace=np.zeros((2,1,16));trace[:,0,1]=[11,12]
        sensory=np.zeros((2,1,148));sensory[0,0,27:31]=-.04;sensory[1,0,27:31]=-.02
        motors=np.zeros((2,1,2,22));motors[1,0,1,5:9]=.025
        targets=np.zeros((1,33));targets[0,:4]=[0,12,0,.3]
        row=np.r_[0,12,0,np.zeros(4),np.full(4,-.04),np.full(4,.04),np.full(4,.025),np.ones(4)]
        raw=dict(trace=trace,sensory=sensory,motor_targets=motors,whole_hand_targets=targets,whole_hand_feedback=row[None],actuator_lower=np.full((1,2,22),-.5),actuator_upper=np.full((1,2,22),.5))
        report=dict(control_hz=40,whole_hand_probe=dict(events=[dict(event='start_accepted',trial=0,env=0,finger_bias_rad=[0.]*22)]))
        path=SimpleNamespace(duration=0.,sample=lambda elapsed:dict(fingers=np.zeros(22)))
        return report,raw,path

    def test_previous_state_and_applied_rate_limited_target(self):
        from rubik_lab.diagnostics.probe_readout import summarize_feedback
        report,raw,path=self.inputs();result=summarize_feedback(report,raw,path)
        self.assertTrue(result['prior_state_reference_and_actual_command_verified'])
        self.assertEqual(result['limited_joint_frames'],4);self.assertEqual(result['max_command_rate_rad_s'],1.)
        raw['whole_hand_feedback'][0,19]=0
        with self.assertRaises(ValueError):summarize_feedback(report,raw,path)
        raw['whole_hand_feedback'][0,19]=1
        # A log made from the post-action state must not pass the causal timing check.
        raw['whole_hand_feedback'][0,7:11]=-.02
        with self.assertRaises(ValueError):summarize_feedback(report,raw,path)

    def test_early_or_unapplied_feedback_rejected(self):
        from rubik_lab.diagnostics.probe_readout import summarize_feedback
        report,raw,path=self.inputs();raw['whole_hand_targets'][0,3]=.2
        with self.assertRaises(ValueError):summarize_feedback(report,raw,path)
        report,raw,path=self.inputs();raw['motor_targets'][1,0,1,5:9]=.04
        with self.assertRaises(ValueError):summarize_feedback(report,raw,path)

class ReturnReadoutTests(unittest.TestCase):
    def inputs(self):
        from types import SimpleNamespace
        report,raw=ProbeReadoutTests().inputs()
        report['whole_hand_probe']['ready_return']=True
        report['whole_hand_probe']['events'] += [dict(event='return_start_accepted',trial=0,env=0,step=1,finger_bias_rad=[0.]*22),dict(event='return_targets_finished_holding_ready',trial=0,env=0,step=2)]
        raw['whole_hand_return_targets']=raw['whole_hand_targets'].copy()
        raw['sensory']=np.zeros((3,1,148));raw['ready_targets']=np.zeros((1,1,2,22));raw['trace'][:,0,3]=1
        path=SimpleNamespace(sample=lambda t:dict(fingers=np.zeros(22)))
        return report,raw,path

    def test_return_finish_does_not_invent_physical_completion(self):
        from rubik_lab.diagnostics.probe_readout import summarize_return
        report,raw,path=self.inputs();result=summarize_return(report,raw,path)
        self.assertTrue(result['actual_commands_verified']);self.assertTrue(result['target_sequence_finished'])
        self.assertFalse(result['original_full_physical_monitor_passed'])
        raw['trace'][2,0,10]=1
        self.assertTrue(summarize_return(report,raw,path)['original_full_physical_monitor_passed'])
        result=summarize_probe(report,raw,(.025,.05))
        self.assertEqual(result['per_planned_phase']['nominal_turn']['frames'],0)
        self.assertEqual(result['per_planned_phase']['withdrawal']['frames'],2)

    def test_unpreserved_support_or_continued_feedback_reject(self):
        from rubik_lab.diagnostics.probe_readout import summarize_return
        report,raw,path=self.inputs();raw['motor_targets'][1,0,1,0]=.01
        with self.assertRaises(ValueError):summarize_return(report,raw,path)
        report,raw,path=self.inputs();raw['whole_hand_feedback']=np.zeros((1,23));raw['whole_hand_feedback'][0,:3]=[0,1,0]
        with self.assertRaises(ValueError):summarize_return(report,raw,path)


class PostReturnReadoutTests(unittest.TestCase):
    def inputs(self,move):
        t=np.zeros((10,1,16));t[:,0,1]=np.arange(11,21);t[:,0,3]=2;t[:,0,4]=1;t[:,0,9]=1;t[-1,0,10]=1
        c=np.zeros((10,1,2,2,27));c[:,0,0,:,5]=1;c[0,0,1,0,10]=3;c[0,0,1,1,10]=7
        q=np.zeros((10,1,148));q[-1,0,49:53]=.2;q[-1,0,71:75]=.4
        report=dict(control_hz=40,timeout_per_primitive_s=6,post_return_events=[dict(event='post_return_handoff',trial=0,envs=[0],step=10,move=move)],episodes=[dict(trial=0,env=0,reason='success',completed_stages=3)])
        raw=dict(trace=t,contact=c,sensory=q,commands=np.zeros((10,1,4)),motor_targets=np.zeros((10,1,2,22)),whole_hand_targets=np.empty((0,33)))
        return report,raw
    def test_direction_selects_real_participating_hand(self):
        from rubik_lab.diagnostics.probe_readout import summarize_post_return
        for move,hand,force,speed in [('U','right',7,.4),("U'",'left',3,.2)]:
            r,z=self.inputs(move);out=summarize_post_return(r,z)
            self.assertEqual(out['hand'],hand);self.assertEqual(out['peak_pad_force_N'],force);self.assertAlmostEqual(out['final_index_speed_rad_s'],speed)
            self.assertTrue(out['original_full_monitor_passed']);self.assertEqual(out['original_elapsed_seconds'],.25)
            r['post_return_events'].append(dict(event='post_return_motor_hold',trial=0,env=0,step=11));self.assertFalse(summarize_post_return(r,z)['original_full_monitor_passed']);r['post_return_events'].pop()
            z['trace'][-1,0,10]=0;self.assertFalse(summarize_post_return(r,z)['original_full_monitor_passed'])
    def test_stale_targets_and_missing_frames_are_rejected(self):
        from rubik_lab.diagnostics.probe_readout import summarize_post_return
        r,z=self.inputs('U');z['whole_hand_targets']=np.zeros((1,33));z['whole_hand_targets'][0,1]=11
        with self.assertRaises(ValueError):summarize_post_return(r,z)
        z['whole_hand_targets']=np.empty((0,33));z['trace'][3,0,1]+=1
        with self.assertRaises(ValueError):summarize_post_return(r,z)


class EscapeReadoutTests(unittest.TestCase):
    def inputs(self):
        t=np.zeros((22,1,16));t[:,0,1]=np.arange(22);t[:,0,3]=2;t[:,0,4]=1
        raw=dict(trace=t,sensory=np.zeros((22,1,148)),poses=np.zeros((22,1,21)),motor_targets=np.zeros((22,1,2,22)),wrist_targets=np.zeros((22,1,2,7)),contact=np.zeros((22,1,2,2,27)),core_angular_speed=np.zeros((22,1)),actuator_lower=np.full((1,2,22),-1.),actuator_upper=np.ones((1,2,22)))
        raw['contact'][:,0,0,:,5]=1
        control={k:v.copy() for k,v in raw.items()}
        raw['motor_targets'][2:,0,0,6]=-np.minimum(np.arange(1,21)/40,.4)
        report=dict(control_hz=40,post_return_events=[dict(event='post_return_escape_start',trial=0,env=0,step=2,source_motor_target=[0.,0.,0.,0.])])
        return report,raw,control
    def test_relaxation_reference_must_come_from_previous_proprioception(self):
        from rubik_lab.diagnostics.probe_readout import summarize_escape
        r,z,b=self.inputs();reference=np.array([.06,-.04,.31,.14]);z['sensory'][1,0,5:9]=reference;b['sensory'][1,0,5:9]=reference
        event=r['post_return_events'][0];event['event']='post_return_relax_start';event['physical_reference']=reference.tolist()
        elapsed=np.arange(1,21)[:,None]/40
        z['motor_targets'][2:,0,0,5:9]=np.clip(reference,-elapsed,elapsed)
        self.assertTrue(summarize_escape(r,z,b)['first_segment_supported'])
        event['physical_reference'][2]+=.01
        with self.assertRaises(ValueError):summarize_escape(r,z,b)

    def test_matching_prefix_and_actual_targets_required(self):
        from rubik_lab.diagnostics.probe_readout import summarize_escape
        r,z,b=self.inputs();out=summarize_escape(r,z,b)
        self.assertTrue(out['first_segment_supported'])
        z['poses'][0,0,0]=.001
        with self.assertRaises(ValueError):summarize_escape(r,z,b)
        r,z,b=self.inputs();z['motor_targets'][3,0,0,6]=0
        with self.assertRaises(ValueError):summarize_escape(r,z,b)
    def test_core_contact_cannot_be_reported_as_whole_finger_release(self):
        from rubik_lab.diagnostics.probe_readout import summarize_escape
        r,z,b=self.inputs();z['contact'][2:,0,0,0,9]=2
        out=summarize_escape(r,z,b)
        self.assertEqual(out['final_release_force_N'],0)
        self.assertEqual(out['final_index_total_force_N'],2)
        self.assertFalse(out['first_segment_supported'])

    def test_force_and_angle_failure_cannot_be_hidden_by_final_state(self):
        from rubik_lab.diagnostics.probe_readout import summarize_escape
        r,z,b=self.inputs();z['trace'][3,0,5]=6
        self.assertFalse(summarize_escape(r,z,b)['first_segment_supported'])
        r,z,b=self.inputs();z['trace'][-1,0,8]=.6
        self.assertFalse(summarize_escape(r,z,b)['first_segment_supported'])
        r,z,b=self.inputs();z['motor_targets'][5,0,1,0]=.01
        self.assertFalse(summarize_escape(r,z,b)['first_segment_supported'])


class LeftFullReturnReadoutTests(unittest.TestCase):
    def inputs(self):
        from types import SimpleNamespace
        r,z,b=EscapeReadoutTests().inputs();r['timeout_per_primitive_s']=6
        r['post_return_events']=[dict(event='left_return_start',trial=0,env=0,step=2),dict(event='left_return_targets_finished',trial=0,env=0,step=5),dict(event='post_return_handoff',trial=0,envs=[0],step=0,move="U'")]
        r['episodes']=[dict(trial=0,env=0,reason='success',completed_stages=3)]
        z['trace'][2:,0,9:11]=1;z['ready_targets']=np.zeros((1,1,2,22))
        z['motor_targets'][2:,0,0,6]=-np.minimum(np.arange(1,21)/40,.1)
        motor=np.zeros((2,22));motor[1,6]=-.1
        p=SimpleNamespace(times=np.array([0.,.1]),duration=.1,motor=motor)
        return r,z,b,p
    def test_actual_completion_and_delivery_required(self):
        from rubik_lab.diagnostics.probe_readout import summarize_left_return
        r,z,b,p=self.inputs();self.assertTrue(summarize_left_return(r,z,b,p)['original_full_physical_monitor_passed'])
        z['trace'][-1,0,10]=0;self.assertFalse(summarize_left_return(r,z,b,p)['original_full_physical_monitor_passed'])
        r,z,b,p=self.inputs();z['motor_targets'][4,0,0,6]=0
        with self.assertRaises(ValueError):summarize_left_return(r,z,b,p)
    def test_core_contact_and_early_finish_rejected(self):
        from rubik_lab.diagnostics.probe_readout import summarize_left_return
        r,z,b,p=self.inputs();z['contact'][-1,0,0,0,9]=1
        self.assertFalse(summarize_left_return(r,z,b,p)['original_full_physical_monitor_passed'])
        r,z,b,p=self.inputs();r['post_return_events'][1]['step']=4
        with self.assertRaises(ValueError):summarize_left_return(r,z,b,p)


class WholeFingerReportingTests(unittest.TestCase):
    def test_core_contact_is_included_in_whole_finger_force(self):
        from rubik_lab.diagnostics.probe_readout import summarize_post_return
        r,z=PostReturnReadoutTests().inputs('U')
        z['contact'][-1,0,0,1,9]=2.0
        result=summarize_post_return(r,z)
        self.assertAlmostEqual(result['final_whole_index_release_force_N']-result['final_top_index_release_force_N'],2.0)
        self.assertIn('top-only',result['release_monitor_scope'])


class FourthReadoutTests(unittest.TestCase):
    def inputs(self):
        r,z=PostReturnReadoutTests().inputs('U')
        z={k:(np.concatenate([v[:1],v],axis=0) if len(v) else v) for k,v in z.items()}
        z['trace'][0,0,1]=10;z['trace'][0,0,3]=2;z['trace'][1:,0,3]=3
        z['core_angular_speed']=np.zeros((11,1));z['poses']=np.zeros((11,1,21));z['wrist_targets']=np.zeros((11,1,2,7))
        z['contact'][1:,0,:,1,6:11]=0
        r['post_return_events'][0].update(event='after_left_return_handoff',diagnostic_targets_cleared=True,preserved_fields=42)
        r['episodes'][0]['completed_stages']=4
        return r,z,{k:v.copy() for k,v in z.items()}
    def test_exact_prefix_and_core_release_required(self):
        from rubik_lab.diagnostics.probe_readout import summarize_fourth
        r,z,b=self.inputs();self.assertTrue(summarize_fourth(r,z,b)['original_full_monitor_passed'])
        z['contact'][-1,0,0,1,9]=1
        self.assertFalse(summarize_fourth(r,z,b)['original_full_monitor_passed'])
        z['poses'][0,0,0]=.001
        with self.assertRaises(ValueError):summarize_fourth(r,z,b)
    def test_actual_handoff_does_not_claim_independent_control_match(self):
        from rubik_lab.diagnostics.probe_readout import summarize_fourth
        r,z,b=self.inputs();out=summarize_fourth(r,z)
        self.assertFalse(out['prefix_exact_match']);self.assertTrue(out['original_full_monitor_passed'])
        r['post_return_events'][0]['diagnostic_targets_cleared']=False
        with self.assertRaises(ValueError):summarize_fourth(r,z)

    def test_third_completion_survives_later_failure(self):
        from rubik_lab.diagnostics.probe_readout import summarize_left_return
        r,z,b,p=LeftFullReturnReadoutTests().inputs()
        r['episodes'][0].update(reason='timeout',completed_stages=3)
        r['post_return_events'].append(dict(event='after_left_return_handoff',step=21))
        self.assertTrue(summarize_left_return(r,z,b,p)['original_full_physical_monitor_passed'])
        z['trace'][-1,0,10]=0
        self.assertFalse(summarize_left_return(r,z,b,p)['original_full_physical_monitor_passed'])


class PreloadBrakeReadoutTests(unittest.TestCase):
    def inputs(self):
        r,z,b=EscapeReadoutTests().inputs();reference=np.array([.06,-.04,.21,.14]);z['sensory'][1,0,5:9]=reference
        z['trace'][0,0,5:7]=[30,10];z['trace'][1,0,5:7]=[15,10]
        e=r['post_return_events'][0];e.update(event='post_return_relax_start',physical_reference=reference.tolist())
        r['diagnostic_preload_at_error_deg']=20
        elapsed=np.arange(1,21)[:,None]/40;z['motor_targets'][2:,0,0,5:9]=np.clip(reference,-elapsed,elapsed)
        return r,z
    def test_trigger_and_latched_reference_are_causal(self):
        from rubik_lab.diagnostics.probe_readout import summarize_preload_brake
        r,z=self.inputs();self.assertFalse(summarize_preload_brake(r,z)['matched_control_verified']);self.assertTrue(summarize_preload_brake(r,z)['braking_hypothesis_supported'])
        z['trace'][0,0,5]=19
        with self.assertRaises(ValueError):summarize_preload_brake(r,z)
        r,z=self.inputs();z['sensory'][1,0,7]+=.01
        with self.assertRaises(ValueError):summarize_preload_brake(r,z)
    def test_stable_overshoot_or_core_force_rejects_braking(self):
        from rubik_lab.diagnostics.probe_readout import summarize_preload_brake
        r,z=self.inputs();z['trace'][-10:,0,5]=-6
        self.assertFalse(summarize_preload_brake(r,z)['braking_hypothesis_supported'])
        r,z=self.inputs();z['contact'][-10:,0,0,0,9]=1
        self.assertFalse(summarize_preload_brake(r,z)['braking_hypothesis_supported'])



class ResidualEscapeReadoutTests(unittest.TestCase):
    def inputs(self):
        r,z,b=EscapeReadoutTests().inputs()
        z['motor_targets'][:]=0;b['motor_targets'][:]=0
        # Natural opposite-hand settling is allowed only when exactly paired.
        drift=np.linspace(0,.0003,22)
        z['motor_targets'][:,0,1,0]=drift;b['motor_targets'][:,0,1,0]=drift
        z['motor_targets'][2:,0,0,7]=.02
        r['post_return_events']=[dict(event='preload_residual_escape_start',trial=0,env=0,step=2)]
        return r,z,b,np.array([0.,0.,.02,0.])
    def test_paired_drift_allowed_but_unrelated_change_or_prefix_rejected(self):
        from rubik_lab.diagnostics.probe_readout import summarize_residual_escape
        r,z,b,d=self.inputs();self.assertTrue(summarize_residual_escape(r,z,b,d)['hypothesis_supported'])
        z['motor_targets'][5,0,1,0]+=.00001
        with self.assertRaises(ValueError):summarize_residual_escape(r,z,b,d)
        r,z,b,d=self.inputs();z['poses'][1,0,0]=.00001
        with self.assertRaises(ValueError):summarize_residual_escape(r,z,b,d)
    def test_force_failure_and_motor_mismatch_rejected(self):
        from rubik_lab.diagnostics.probe_readout import summarize_residual_escape
        r,z,b,d=self.inputs();z['contact'][-1,0,0,0,9]=.6
        self.assertFalse(summarize_residual_escape(r,z,b,d)['hypothesis_supported'])
        r,z,b,d=self.inputs();z['motor_targets'][4,0,0,7]=.01
        with self.assertRaises(ValueError):summarize_residual_escape(r,z,b,d)
        r,z,b,d=self.inputs();z['trace'][-10:,0,5]=5.1
        self.assertFalse(summarize_residual_escape(r,z,b,d)['hypothesis_supported'])


class LiveLeftTimingTests(unittest.TestCase):
    def test_first_actual_full_window_required(self):
        from rubik_lab.diagnostics.probe_readout import audit_live_left_timing
        r,z=PreloadBrakeReadoutTests().inputs()
        r['post_return_events'].append(dict(event='left_return_start',step=14))
        self.assertEqual(audit_live_left_timing(r,z)['initial_source_step'],13)
        r['post_return_events'][-1]['step']=15
        with self.assertRaises(ValueError):audit_live_left_timing(r,z)
    def test_residual_then_strict_release_window_and_no_extra_attempts(self):
        from rubik_lab.diagnostics.probe_readout import audit_live_left_timing
        r,z=PreloadBrakeReadoutTests().inputs();z['contact'][10:14,0,0,0,9]=.6
        r['post_return_events'] += [dict(event='preload_residual_escape_start',step=14,duration_seconds=.02),dict(event='left_return_start',step=18)]
        self.assertEqual(audit_live_left_timing(r,z)['attempts'],2)
        r['post_return_events'][-1]['step']=17
        with self.assertRaises(ValueError):audit_live_left_timing(r,z)
        r['post_return_events'][-1]['step']=18;r['post_return_events'].append(dict(event='left_return_start',step=19))
        with self.assertRaises(ValueError):audit_live_left_timing(r,z)

if __name__=='__main__':unittest.main()
