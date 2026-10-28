"""Regression tests: readiness labels, episode isolation, grasp holdout and 21 move IDs."""
import sys
from pathlib import Path
import tempfile
import unittest
import h5py
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/rubik_lab"))
from rubik_lab.detector.data import TactileWindows, split_episodes, N_MOVES
from rubik_lab.detector.model import CompletionDetector


class DetectorDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "data.h5")
        with h5py.File(self.path, "w") as f:
            n = 12
            f["episode_uid"] = np.repeat(np.arange(4), 3)
            f["grasp_idx"] = np.repeat([10, 10, 20, 20], 3)
            f["step"] = np.tile(np.arange(3), 4)
            f["tactile_ft"] = np.arange(n)[:, None, None, None] * np.ones((n, 2, 5, 6))
            f["finger_pos"] = np.zeros((n, 2, 22))
            f["finger_tgt"] = np.zeros((n, 2, 22))
            f["move_id"] = np.full(n, 20)
            f["move_done"] = np.ones(n, bool)
            f["primitive_done"] = np.tile([False, False, True], 4)
            f["dropped"] = np.zeros(n, bool)
            f["timeout"] = np.zeros(n, bool)
            f["cycle_failed"] = np.zeros(n, bool)

    def tearDown(self):
        self.tmp.cleanup()

    def test_image_mode_requires_actual_images(self):
        with self.assertRaisesRegex(ValueError, "no tactile_image"):
            TactileWindows(self.path, images=True)

    def test_legacy_images_rejected_but_ft_data_remains_usable(self):
        with h5py.File(self.path, 'a') as f:
            f['tactile_image'] = np.zeros((12,2,5,60,60), dtype=np.uint8)
        for version in (None, 1):
            if version is not None:
                with h5py.File(self.path, 'a') as f: f.attrs['tactile_geometry_version']=version
            with self.assertRaisesRegex(ValueError, "geometry_version=2"):
                TactileWindows(self.path, images=True)
        ds=TactileWindows(self.path);self.assertFalse(ds.images);ds.f.close()

    def test_contradictory_geometry_metadata_is_rejected(self):
        with h5py.File(self.path,'a') as f:
            f['tactile_image']=np.zeros((12,2,5,60,60),dtype=np.uint8)
            f.attrs['tactile_geometry_version']=2
            f.attrs['config_json']='{"tactile":{"geometry_version":1}}'
        with self.assertRaisesRegex(ValueError,"geometry_version=2"):
            TactileWindows(self.path,images=True)

    def test_corrected_image_geometry_is_explicitly_required(self):
        with h5py.File(self.path, 'a') as f:
            f['tactile_image'] = np.zeros((12,2,5,60,60), dtype=np.uint8)
            f.attrs['config_json']='{"tactile":{"geometry_version":2}}'
        ds=TactileWindows(self.path,images=True,window=2)
        self.assertEqual(tuple(ds[0][0]['image'].shape),(2,2,5,60,60));ds.f.close()

    def test_layer_arrival_is_not_full_completion(self):
        ds = TactileWindows(self.path)
        self.assertEqual(ds.label[:3].tolist(), [0, 0, 1])
        ds.f.close()

    def test_completed_window_labels_do_not_become_instantaneous_ready(self):
        with h5py.File(self.path,"a") as f:
            f["cycle_ready"]=np.tile([False,True,True],4)
        instant=TactileWindows(self.path,label_key='cycle_ready')
        complete=TactileWindows(self.path,label_key='primitive_done')
        self.assertEqual(instant.label[:3].tolist(),[0,1,1])
        self.assertEqual(complete.label[:3].tolist(),[0,0,1])
        instant.f.close();complete.f.close()

    def test_grasp_holdout_has_no_shared_grasp(self):
        tr, va = split_episodes(self.path, group_key="grasp_idx")
        mapping = {0: 10, 1: 10, 2: 20, 3: 20}
        self.assertFalse({mapping[i] for i in tr} & {mapping[i] for i in va})
        self.assertEqual(len(tr) + len(va), 4)

    def test_window_does_not_cross_episode(self):
        ds = TactileWindows(self.path, window=4)
        x, _ = ds[3]
        self.assertTrue(torch.allclose(x["ft"], torch.full((4, 60), 0.6)))
        ds.f.close()

    def test_failure_overrides_ready(self):
        with h5py.File(self.path, "a") as f:
            f["cycle_failed"][2] = True
            f["timeout"][1] = True
        ds = TactileWindows(self.path)
        self.assertEqual(ds.label[:3].tolist(), [0, 0, 2])
        legacy = TactileWindows(self.path, timeout_is_failure=True)
        self.assertEqual(legacy.label[:3].tolist(), [0, 2, 2])
        legacy.f.close()
        ds.f.close()

    def test_full_move_vocabulary_matches_model(self):
        ds = TactileWindows(self.path, window=4)
        x, _ = ds[0]
        self.assertEqual(len(x["move"]), N_MOVES)
        self.assertEqual(N_MOVES, 21)
        model = CompletionDetector()
        out = model({k: v.unsqueeze(0) for k, v in x.items()})
        self.assertEqual(tuple(out.shape), (1, 3))
        ds.f.close()

    def test_merged_stages_of_one_raw_trial_never_cross_split(self):
        with h5py.File(self.path, "a") as f:
            f["source_file"] = np.zeros(12, dtype=int)
            f["source_episode_uid"] = np.repeat([7, 7, 8, 8], 3)
        for seed in range(10):
            train, val = split_episodes(self.path, seed=seed)
            self.assertEqual(set(train), {0, 1} if 0 in train else {2, 3})
            self.assertEqual(set(val), {0, 1} if 0 in val else {2, 3})
            self.assertFalse(set(train) & set(val))

    def test_source_holdout_keeps_recording_runs_separate(self):
        with h5py.File(self.path, "a") as f:
            f["source_file"] = np.repeat([0, 0, 1, 1], 3)
            f["source_episode_uid"] = np.repeat([0, 1, 0, 1], 3)
        train, val = split_episodes(self.path, group_key="source_file")
        self.assertFalse({int(i)//2 for i in train} & {int(i)//2 for i in val})

    def test_single_grasp_cannot_claim_holdout_validation(self):
        with h5py.File(self.path, "a") as f:
            f["grasp_idx"][:] = 10
        with self.assertRaises(ValueError):
            split_episodes(self.path, group_key="grasp_idx")


class TriggerMetricsTests(unittest.TestCase):
    def test_premature_trigger_is_failure_even_if_later_prediction_is_correct(self):
        from rubik_lab.detector.metrics import trigger_metrics
        result = trigger_metrics([0,0,0,1,1,1,2,2], [0,0,1,0,0,1,0,2],
                                 [0.1,0.99,0.99,0.1,0.2,0.99,0.1,0.1], 0.95)
        self.assertEqual(result['premature_triggers'], 1)
        self.assertEqual(result['correct_first_triggers'], 1)
        self.assertEqual(result['positive_episode_recall'], 0.5)
        self.assertEqual(result['missed_positive_episodes'], 1)

    def test_consecutive_gate_rejects_a_brief_premature_spike(self):
        from rubik_lab.detector.metrics import trigger_metrics
        result = trigger_metrics([0]*6, [0,0,0,0,0,1], [0.99,0.1,0.1,0.99,0.99,0.99], 0.95, 3)
        self.assertEqual(result['premature_triggers'], 0)
        self.assertEqual(result['correct_first_triggers'], 1)

    def test_streaming_matches_full_history_without_future_input(self):
        torch.manual_seed(10)
        model=CompletionDetector().eval()
        x={'ft':torch.randn(2,7,60),'prop':torch.randn(2,7,88),'move':torch.zeros(2,N_MOVES)}
        with torch.no_grad():
            full,_=model.forward_sequence(x)
            hidden=None
            pieces=[]
            for step in range(7):
                one={'ft':x['ft'][:,step:step+1],'prop':x['prop'][:,step:step+1],'move':x['move']}
                value,hidden=model.forward_sequence(one,hidden)
                pieces.append(value)
        self.assertTrue(torch.allclose(full,torch.cat(pieces,1),atol=1e-6))

    def test_runtime_readiness_gate_counts_and_resets_independently(self):
        from rubik_lab.detector.streaming import StreamingReadiness
        model=CompletionDetector()
        with torch.no_grad():
            for p in model.parameters():p.zero_()
            model.head[-1].bias[1]=10
        gate=StreamingReadiness({'format':'streaming_tactile_ready_v1','model':model.state_dict(),'hold_steps':3},2,'cpu')
        inputs={'tactile_ft':torch.zeros(2,2,5,6),'finger_pos':torch.zeros(2,2,22),
                'finger_tgt':torch.zeros(2,2,22),'move_id':torch.zeros(2,dtype=torch.long)}
        self.assertFalse(gate.step(**inputs)['done'].any())
        self.assertFalse(gate.step(**inputs)['done'].any())
        self.assertTrue(gate.step(**inputs)['done'].all())
        gate.reset(torch.tensor([0]))
        self.assertEqual(gate.step(**inputs)['done'].tolist(),[False,True])

    def test_angle_gate_blocks_confident_but_off_angle_prediction(self):
        from rubik_lab.detector.model import AngleCompletionDetector
        from rubik_lab.detector.streaming import StreamingReadiness
        model=AngleCompletionDetector()
        with torch.no_grad():
            for parameter in model.parameters():parameter.zero_()
            model.head[-1].bias[1]=10
            model.angle_head[-2].bias.fill_(torch.atanh(torch.tensor(10./180.)))
        gate=StreamingReadiness({'format':'streaming_tactile_ready_angle_v1',
            'model':model.state_dict(),'hold_steps':2,'angle_gate_deg':3.},1,'cpu')
        inputs={'tactile_ft':torch.zeros(1,2,5,6),'finger_pos':torch.zeros(1,2,22),
                'finger_tgt':torch.zeros(1,2,22),'move_id':torch.zeros(1,dtype=torch.long)}
        for _ in range(4):
            result=gate.step(**inputs)
            self.assertFalse(result['done'].item())
            self.assertAlmostEqual(result['angle_error_deg'].item(),10.,places=4)
        with torch.no_grad():gate.model.angle_head[-2].bias.zero_()
        self.assertFalse(gate.step(**inputs)['done'].item())
        self.assertTrue(gate.step(**inputs)['done'].item())
        with torch.no_grad():gate.model.angle_head[-2].bias.fill_(1.)
        self.assertFalse(gate.step(**inputs)['done'].item())

    def test_runtime_restores_training_channel_normalization(self):
        from rubik_lab.detector.streaming import StreamingReadiness
        torch.manual_seed(29)
        model=CompletionDetector().eval()
        stats={'ft_mean':torch.randn(60).tolist(),'ft_scale':torch.linspace(.1,2,60).tolist(),
               'prop_mean':torch.randn(88).tolist(),'prop_scale':torch.linspace(.2,3,88).tolist()}
        model.set_normalization(stats)
        gate=StreamingReadiness({'format':'streaming_tactile_ready_normalized_v1',
            'model':model.state_dict(),'hold_steps':2,'feature_normalization':stats},2,'cpu')
        ft=torch.randn(2,2,5,6)
        q=torch.randn(2,2,22)
        target=torch.randn_like(q)
        moves=torch.tensor([6,7])
        with torch.no_grad():
            logits,_=model.forward_sequence({'ft':ft.flatten(1)[:,None]*.2,
                'prop':torch.cat([q.flatten(1),(target-q).flatten(1)],-1)[:,None],
                'move':torch.nn.functional.one_hot(moves,N_MOVES).float()})
        actual=gate.step(tactile_ft=ft,finger_pos=q,finger_tgt=target,move_id=moves)
        self.assertTrue(torch.allclose(actual['ready_probability'],logits[:,0].softmax(-1)[:,1],atol=1e-6))
        stats['ft_scale'][0]=0
        with self.assertRaises(ValueError):model.set_normalization(stats)

    def test_continuous_observer_keeps_history_but_restarts_completion_dwell(self):
        from rubik_lab.detector.streaming import StreamingReadiness
        torch.manual_seed(37)
        model=CompletionDetector().eval()
        gate=StreamingReadiness({'format':'streaming_tactile_ready_v1','model':model.state_dict(),
            'hold_steps':3,'preserve_history_between_moves':True},2,'cpu')
        ft=torch.randn(2,6,60)
        prop=torch.randn(2,6,88)
        moves=torch.tensor([[6,6,6,7,7,7],[6,6,6,6,6,6]])
        with torch.no_grad():
            expected,_=model.forward_sequence({'ft':ft,'prop':prop,
                'move':torch.nn.functional.one_hot(moves,N_MOVES).float()})
        actual=[]
        for t in range(6):
            if t==3:
                old=gate.hidden.clone()
                with torch.inference_mode():gate.stable_steps[:]=2
                gate.advance(torch.tensor([0]))
                self.assertTrue(torch.equal(gate.hidden,old))
                self.assertEqual(gate.stable_steps.tolist(),[0,2])
            q=prop[:,t,:44].reshape(2,2,22)
            target=q+prop[:,t,44:].reshape(2,2,22)
            actual.append(gate.step(tactile_ft=(ft[:,t]/.2).reshape(2,2,5,6),
                finger_pos=q,finger_tgt=target,move_id=moves[:,t])['ready_probability'])
        self.assertTrue(torch.allclose(torch.stack(actual,1),expected.softmax(-1)[:,:,1],atol=1e-6))
        gate.reset()
        self.assertTrue((gate.hidden==0).all())
        self.assertTrue((gate.stable_steps==0).all())

    def test_per_move_readiness_routes_command_and_resets_boundaries(self):
        from rubik_lab.detector.streaming import StreamingReadiness, PerMoveReadiness
        observers={}
        for move,bias in [(6,10.),(7,-10.)]:
            model=CompletionDetector()
            with torch.no_grad():
                for parameter in model.parameters():parameter.zero_()
                model.head[-1].bias[1]=bias
            observers[move]=StreamingReadiness({'format':'streaming_tactile_ready_v1',
                'model':model.state_dict(),'hold_steps':2},2,'cpu')
        router=PerMoveReadiness(observers)
        inputs={'tactile_ft':torch.zeros(2,2,5,6),'finger_pos':torch.zeros(2,2,22),
            'finger_tgt':torch.zeros(2,2,22),'move_id':torch.tensor([6,7])}
        self.assertEqual(router.step(**inputs)['done'].tolist(),[False,False])
        self.assertEqual(router.step(**inputs)['done'].tolist(),[True,False])
        router.advance(torch.tensor([1]))
        inputs['move_id'][1]=6
        self.assertEqual(router.step(**inputs)['done'].tolist(),[True,False])
        self.assertEqual(router.step(**inputs)['done'].tolist(),[True,True])
        inputs['move_id'][0]=0
        with self.assertRaises(ValueError):router.step(**inputs)
        observers[6].preserve_history_between_moves=True
        with self.assertRaises(ValueError):PerMoveReadiness(observers)

    def test_completed_window_observer_metadata_is_explicit(self):
        from rubik_lab.detector.streaming import StreamingReadiness,PerMoveReadiness
        model=CompletionDetector()
        payload={'format':'streaming_tactile_ready_v1','model':model.state_dict(),
                 'hold_steps':1,'label_key':'primitive_done'}
        complete=StreamingReadiness(payload,1,'cpu')
        self.assertEqual(complete.label_key,'primitive_done')
        payload['label_key']='cycle_ready'
        instant=StreamingReadiness(payload,1,'cpu')
        with self.assertRaises(ValueError):PerMoveReadiness({6:complete,7:instant})
        payload['hold_steps']=0
        with self.assertRaises(ValueError):StreamingReadiness(payload,1,'cpu')

    def test_interleaved_episode_rows_preserve_time_order(self):
        from rubik_lab.detector.metrics import trigger_metrics
        result=trigger_metrics([2,1,2,1,2,1],[0,0,0,1,1,1],[.99]*6,.95,2)
        self.assertEqual(result['premature_triggers'],1)
        self.assertEqual(result['correct_first_triggers'],1)
        result=trigger_metrics([2,1,2,1,2,1],[0,0,0,1,1,1],[.99]*6,.95,4)
        self.assertEqual(result['missed_positive_episodes'],2)

    def test_no_positive_examples_cannot_claim_recall(self):
        from rubik_lab.detector.metrics import trigger_metrics
        result = trigger_metrics([0,0], [0,2], [0.1,0.1], 0.95)
        self.assertIsNone(result['positive_episode_recall'])


if __name__ == "__main__":
    unittest.main()
