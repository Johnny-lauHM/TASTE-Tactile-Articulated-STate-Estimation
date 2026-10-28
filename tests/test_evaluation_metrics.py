import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source/rubik_lab'))
from rubik_lab.evaluation_metrics import summarize_stages, summarize_mechanical, summarize_continuation

class StageMetricsTests(unittest.TestCase):
    def test_later_skill_denominator_excludes_trials_that_never_reached_it(self):
        completed=[{'seconds':1.},{'seconds':2.}]
        rows=[{'primitives':completed,'reason':'success'},
              {'primitives':completed,'reason':'success'},
              {'primitives':[],'reason':'timeout'},
              {'primitives':completed[:1],'reason':'premature_trigger'}]
        result=summarize_stages(rows,['U',"U'"])
        self.assertEqual((result[0]['attempted'],result[0]['completed']),(4,3))
        self.assertEqual((result[1]['attempted'],result[1]['completed']),(3,2))
        self.assertAlmostEqual(result[1]['conditional_completion_rate'],2/3)
        self.assertEqual(result[1]['completion_rate_over_all_trials'],.5)
        self.assertEqual(result[1]['premature_requests'],1)
        self.assertEqual(result[0]['premature_requests'],0)

class MechanicalMetricsTests(unittest.TestCase):
    def report(self):
        return dict(rounds=2,num_envs=2,episodes=[dict(trial=t,env=e,move='U' if e==0 else "U'",reason='success' if e else 'timeout') for t in range(2) for e in range(2)])
    def test_round_counts_and_edge_intervals(self):
        result=summarize_mechanical(self.report())
        self.assertTrue(result['quota_complete'])
        self.assertEqual(result['by_move']["U'"]['overall']['success'],2)
        self.assertEqual(result['by_move']['U']['overall']['failure_types'],{'timeout':2})
        self.assertAlmostEqual(result['by_move']["U'"]['rounds'][0]['wilson95'][0],0.20654931437723745)
        self.assertIsNone(result['premature_confirmation_rate'])
    def test_missing_trial_is_not_complete_quota(self):
        report=self.report();report['episodes'].pop()
        self.assertFalse(summarize_mechanical(report)['quota_complete'])
    def test_wrong_move_assignment_does_not_satisfy_condition_quota(self):
        report=self.report();report["episodes"][0]["move"]="U'"
        self.assertFalse(summarize_mechanical(report)["quota_complete"])
    def test_duplicates_rejected(self):
        report=self.report();report['episodes'].append(report['episodes'][0])
        with self.assertRaises(ValueError):summarize_mechanical(report)


class ContinuationMetricsTests(unittest.TestCase):
    def report(self):
        pairs=[['U','U'],['U',"U'"],["U'",'U'],["U'","U'"]]
        return dict(rounds=2,num_envs=4,episodes=[dict(trial=t,env=e,pair=pairs[e],completed_stages=2 if t else 0,reason='success' if t else 'timeout') for t in range(2) for e in range(4)])
    def test_conditional_denominator_and_complete_quota(self):
        result=summarize_continuation(self.report());row=result['chains']['U->U']['overall']
        self.assertTrue(result['quota_complete']);self.assertEqual(row['success_rate'],.5)
        self.assertEqual(row['second_conditional_rate'],1.);self.assertIsNone(result['premature_confirmation_rate'])
    def test_missing_duplicate_wrong_chain_and_false_success_rejected(self):
        for kind in ('missing','duplicate','chain','success'):
            with self.subTest(kind=kind):
                r=self.report()
                if kind=='missing':r['episodes'].pop()
                elif kind=='duplicate':r['episodes'].append(r['episodes'][0])
                elif kind=='chain':r['episodes'][0]['pair']=['U',"U'"]
                else:r['episodes'][0]['reason']='success'
                with self.assertRaises(ValueError):summarize_continuation(r)

class WithdrawalMetricsTests(unittest.TestCase):
    def test_signed_error_and_stage_are_not_success_gates(self):
        import numpy as np
        from rubik_lab.evaluation_metrics import summarize_withdrawal
        from rubik_lab.constants import FINGER_JOINTS
        t=np.zeros((5,1,16));t[:,:,3:5]=1
        t[:,0,5]=[-90,-3,3,3,3]
        t[:,0,7]=[0,.8,0,0,0]
        t[:,0,8]=[0,0,0,.6,0]
        sensory=np.zeros((5,1,148))
        col=next(i for i,j in enumerate(FINGER_JOINTS) if j.startswith('index'))
        sensory[2,0,66+col]=1.1
        raw=dict(trace=t,sensory=sensory,withdrawal_stage=np.full((5,1),3))
        report=dict(episodes=[dict(pair=['U','U'],completed_stages=1,env=0,trial=0,reason='timeout')])
        row=summarize_withdrawal(report,raw)['rows'][0]
        self.assertEqual(row['near_slow_frames'],4)
        self.assertEqual(row['near_slow_actual_ready_frames'],1)
        self.assertEqual(row['renewed_contact_frames'],1)
        self.assertFalse(row['success'])

class TerminalHoldMetricsTests(unittest.TestCase):
    def test_reject_stale_handoff_and_changed_command(self):
        import numpy as np
        from rubik_lab.evaluation_metrics import summarize_terminal_hold
        t=np.zeros((3,1,16));t[:,:,3:5]=1;t[:,0,5]=[8,8,8]
        raw=dict(trace=t,terminal_hold=np.ones((3,1),dtype=bool),commands=np.zeros((3,1,4)))
        report=dict(episodes=[dict(pair=['U','U'],completed_stages=1,env=0,trial=0,reason='timeout')])
        with self.assertRaises(ValueError):summarize_terminal_hold(report,raw)
        raw['terminal_hold'][0]=False
        self.assertEqual(summarize_terminal_hold(report,raw)['triggered'],1)
        raw['commands'][2,0,0]=.1
        with self.assertRaises(ValueError):summarize_terminal_hold(report,raw)

class ReleaseSubstepMetricsTests(unittest.TestCase):
    def test_constant_velocity_and_missing_step(self):
        import numpy as np
        from rubik_lab.evaluation_metrics import summarize_release_substeps
        z=np.zeros((5,7));z[:,1]=np.arange(5);z[:,3]=np.arange(5)*.01;z[:,4]=1
        result=summarize_release_substeps(dict(physics_hz=100),dict(release_substeps=z))
        self.assertLess(result['max_abs_integral_residual_deg'],1e-10)
        z[3,1]+=1
        with self.assertRaises(ValueError):summarize_release_substeps(dict(physics_hz=100),dict(release_substeps=z))
    def test_empty_is_not_evidence_of_consistency(self):
        from rubik_lab.evaluation_metrics import summarize_release_substeps
        self.assertIsNone(summarize_release_substeps(dict(physics_hz=240),{})['max_abs_integral_residual_deg'])

if __name__=='__main__':unittest.main()


class RecontactRawAuditTests(unittest.TestCase):
    def fixture(self):
        import numpy as np
        raw={k:np.zeros(shape) for k,shape in dict(trace=(10,1,16),recontact=(10,1,7),sensory=(10,1,148),contact=(10,1,2,2,27),poses=(10,1,21),ready_targets=(1,1,2,22),core_angular_speed=(10,1)).items()}
        raw['trace'][:,0,1]=np.arange(10)
        raw['trace'][:,0,3:5]=1
        raw['recontact'][:,0,2]=1
        raw['contact'][:,0,0,:,15]=1
        raw['poses'][:,0,3]=1
        return {'episodes':[dict(trial=0,env=0,reason='success')]},raw

    def audit(self,report,raw):
        from rubik_lab.evaluation_metrics import audit_recontact_completion
        return audit_recontact_completion(report,raw)

    def test_valid_measurements_do_not_need_success_flags(self):
        report,raw=self.fixture()
        self.assertTrue(self.audit(report,raw)['rows'][0]['passed'])
        # No simulator ready/done booleans were supplied as true.
        self.assertFalse(raw['trace'][:,:,9:11].any())

    def test_raw_counterexamples_override_claimed_success(self):
        cases=[('sensory',(0,0,5),.21),('sensory',(0,0,44+22+5),1.01),
               ('contact',(0,0,1,0,6),.51),('core_angular_speed',(0,0),1.01),
               ('poses',(9,0,0),.0021),('trace',(9,0,1),11),('trace',(0,0,5),5.),
               ('sensory',(0,0,0),float('nan'))]
        for key,index,value in cases:
            with self.subTest(key=key,index=index):
                report,raw=self.fixture();raw[key][index]=value
                self.assertFalse(self.audit(report,raw)['rows'][0]['passed'])

    def test_missing_measurement_is_unknown_and_empty_is_not_a_pass(self):
        report,raw=self.fixture();del raw['ready_targets']
        self.assertEqual(self.audit(report,raw)['status'],'unknown')
        report,raw=self.fixture();raw['recontact'].fill(0)
        self.assertEqual(self.audit(report,raw)['rows'],[])
