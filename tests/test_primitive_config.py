import sys,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source/rubik_lab'))
from rubik_lab.primitive_config import validate_sequence_policy_settings

class SequenceConfigTests(unittest.TestCase):
    def test_reset_distribution_can_differ_but_live_controls_must_match(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            saved={'grasp_cache_path':'training.npz','episode_length_s':6.,
                'cycle_terminate_on_success':True,'cycle_stable_s':.25,'residual_finger_range':.6}
            (path/'env.json').write_text(json.dumps(saved))
            live=SimpleNamespace(grasp_cache_path='evaluation.npz',episode_length_s=12.,
                cycle_terminate_on_success=False,cycle_stable_s=.25,residual_finger_range=.6)
            self.assertEqual(validate_sequence_policy_settings(live,path/'model.pt'),str(path/'env.json'))
            saved['residual_finger_range']=.9
            (path/'env.json').write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError,'residual_finger_range'):
                validate_sequence_policy_settings(live,path/'model.pt')

    def test_missing_legacy_metadata_is_not_claimed_verified(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(validate_sequence_policy_settings(SimpleNamespace(),Path(folder)/'model.pt'))

if __name__=='__main__':unittest.main()
