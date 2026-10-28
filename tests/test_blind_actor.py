import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source/rubik_lab'))
from rubik_lab.blind_actor import student_features, BlindPrimitiveActor, BlindInference, TrajectoryInference


class BlindActorTests(unittest.TestCase):
    def test_privileged_columns_cannot_change_student_input(self):
        x = torch.randn(3, 250)
        changed = x.clone()
        for start, end in ((44,72), (146,174), (204,220), (241,246)):
            changed[:, start:end] += 100
        self.assertTrue(torch.equal(student_features(x), student_features(changed)))
        self.assertEqual(student_features(x).shape[-1], 173)

    def test_tactile_and_joint_measurements_are_not_discarded(self):
        x = torch.zeros(1,250)
        before = student_features(x)
        x[:,72] = 1
        x[:,102] = 1
        self.assertEqual(int((before != student_features(x)).sum()), 2)

    def test_partial_reset_preserves_other_history(self):
        model = BlindPrimitiveActor(16)
        policy = BlindInference({'hidden':16, 'model':model.state_dict()},2,'cpu')
        policy({'policy':torch.randn(2,250)})
        previous = policy.hidden.clone()
        policy.reset(torch.tensor([0]))
        self.assertTrue((policy.hidden[:,0] == 0).all())
        self.assertTrue(torch.equal(previous[:,1],policy.hidden[:,1]))

    def test_trajectory_uses_initial_joints_and_holds_last_action(self):
        policy=TrajectoryInference({'anchors':torch.stack([torch.zeros(44),torch.ones(44)]),
            'trajectories':torch.tensor([[[.1]*4,[.2]*4],[[.3]*4,[.4]*4]]),
            'lengths':torch.tensor([2,2])},2,'cpu')
        observation=torch.zeros(2,250)
        observation[1,:22]=1
        observation[1,102:124]=1
        x={'policy':observation}
        self.assertTrue(torch.allclose(policy(x)[:,0],torch.tensor([.1,.3])))
        self.assertTrue(torch.allclose(policy(x)[:,0],torch.tensor([.2,.4])))
        self.assertTrue(torch.allclose(policy(x)[:,0],torch.tensor([.2,.4])))
        policy.reset(torch.tensor([0]))
        self.assertTrue(torch.allclose(policy(x)[:,0],torch.tensor([.1,.4])))

    def test_settle_prefix_consumes_real_steps_before_replay(self):
        policy=TrajectoryInference({'anchors':torch.zeros(1,44),
            'trajectories':torch.tensor([[[.1]*4,[.2]*4,[.7]*4]]),
            'lengths':torch.tensor([3]),'settle_steps':2},2,'cpu')
        x={'policy':torch.zeros(2,250)}
        observed=[float(policy(x)[0,0]) for _ in range(6)]
        self.assertTrue(torch.allclose(torch.tensor(observed),torch.tensor([.7,.7,.1,.2,.7,.7])))
        policy.reset(torch.tensor([0]))
        self.assertTrue(torch.allclose(policy(x)[:,0],torch.tensor([.7,.7])))
        policy(x)
        self.assertTrue(torch.allclose(policy(x)[:,0],torch.tensor([.1,.7])))


if __name__ == '__main__':
    unittest.main()
