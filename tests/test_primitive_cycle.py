"""CPU regression tests for the real batched acceptance monitor, no Isaac import."""
import importlib.util
from pathlib import Path
import unittest
import torch

path = Path(__file__).resolve().parents[1] / "source/rubik_lab/rubik_lab/primitive_cycle.py"
spec = importlib.util.spec_from_file_location("primitive_cycle_test_module", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.monitor = module.CycleMonitor(2, "cpu", 0.025, 0.25)

    def step(self, layer=(True, True), fingers=(True, True), grasp=(True, True),
             motion=(True, True), failed=(False, False)):
        return self.monitor.update(**{k: torch.tensor(v) for k, v in dict(
            layer_ok=layer, fingers_ready=fingers, grasp_ok=grasp,
            motion_ok=motion, failed=failed).items()})

    def test_requires_ten_consecutive_frames(self):
        for _ in range(9):
            self.assertFalse(self.step().any())
        self.assertTrue(self.step().all())

    def test_brief_alignment_with_finger_still_pushing_is_not_success(self):
        for _ in range(20):
            self.assertFalse(self.step(fingers=(False, False)).any())
        self.assertTrue(self.monitor.reached.all())
        self.assertEqual(self.monitor.stable_steps.tolist(), [0, 0])

    def test_rebound_resets_hold_timer(self):
        for _ in range(9):
            self.step()
        self.step(layer=(False, True))
        self.assertEqual(self.monitor.stable_steps.tolist(), [0, 10])
        self.assertEqual(self.step().tolist(), [False, True])

    def test_motion_violation_remains_failure_after_returning_home(self):
        self.step(motion=(False, True))
        for _ in range(15):
            result = self.step()
        self.assertEqual(result.tolist(), [False, True])
        self.assertEqual(self.monitor.failed.tolist(), [True, False])

    def test_drop_takes_precedence_over_success(self):
        for _ in range(9):
            self.step()
        self.assertEqual(self.step(failed=(True, False)).tolist(), [False, True])

    def test_partial_reset_does_not_reset_other_environment(self):
        for _ in range(10):
            self.step()
        self.monitor.reset(torch.tensor([0]))
        self.assertEqual(self.monitor.done.tolist(), [False, True])
        self.assertEqual(self.monitor.stable_steps.tolist(), [0, 10])

    def test_loss_of_support_restarts_hold(self):
        for _ in range(9):
            self.step()
        self.assertFalse(self.step(grasp=(False, False)).any())
        self.assertEqual(self.monitor.stable_steps.tolist(), [0, 0])


class PoseWindowTests(unittest.TestCase):
    def test_drift_is_measured_from_window_start(self):
        window = module.PoseWindow(1, 'cpu', 0.002, 0.035)
        q = torch.tensor([[1., 0., 0., 0.]])
        self.assertTrue(window.update(torch.zeros(1, 3), q, torch.tensor([True])).item())
        for x in [0.0008, 0.0016]:
            self.assertTrue(window.update(torch.tensor([[x, 0., 0.]]), q, torch.tensor([False])).item())
        self.assertFalse(window.update(torch.tensor([[0.0024, 0., 0.]]), q, torch.tensor([False])).item())
        self.assertTrue(window.update(torch.tensor([[0.0024, 0., 0.]]), q, torch.tensor([True])).item())

    def test_quaternion_sign_and_rotation_limit(self):
        import math
        window = module.PoseWindow(2, 'cpu', 0.002, math.radians(2))
        p = torch.zeros(2, 3)
        q = torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]])
        window.update(p, q, torch.tensor([True, True]))
        q[0] *= -1
        q[1] = torch.tensor([math.cos(math.radians(1.5)), 0., 0., math.sin(math.radians(1.5))])
        self.assertEqual(window.update(p, q, torch.tensor([False, False])).tolist(), [True, False])


class ParticipantFingerTests(unittest.TestCase):
    def setUp(self):
        self.q = torch.zeros(1, 2, 4)
        self.qd = torch.zeros_like(self.q)
        self.ready = torch.zeros_like(self.q)
        self.force = torch.zeros(1, 2, 5)
        self.hands = torch.tensor([[0, 1]])
        self.cols = torch.tensor([[[0, 1], [0, 1]]])
        self.links = torch.tensor([[[1., 1., 0., 0., 0.], [1., 1., 0., 0., 0.]]])
        self.part = torch.ones(1, 2, dtype=torch.bool)

    def metrics(self):
        return module.participant_finger_metrics(self.q, self.qd, self.ready,
            self.force, self.hands, self.cols, self.links, self.part)

    def test_second_finger_blocks_dwell_and_restarts_full_window(self):
        monitor = module.CycleMonitor(1, 'cpu', .025, .25)
        true = torch.tensor([True])
        def update():
            error, speed, release = self.metrics()
            return monitor.update(layer_ok=true, fingers_ready=(error<=.2)&(speed<=1)&(release<=.5),
                grasp_ok=true, motion_ok=true, failed=~true).item()
        for _ in range(9): self.assertFalse(update())
        # Original right index is ready; correcting left index is not released.
        self.force[0, 0, 0] = .51
        self.assertFalse(update())
        self.assertEqual(monitor.stable_steps.item(), 0)
        self.force.zero_()
        for _ in range(9): self.assertFalse(update())
        self.assertTrue(update())
        self.q[0, 0, 1] = .21
        self.assertFalse(update())
        self.q.zero_(); self.qd[0, 0, 1] = 1.01
        self.assertFalse(update())

    def test_each_finger_has_original_force_limit_and_whole_finger_sum(self):
        self.force[0, :, 0] = .3
        self.force[0, :, 1] = .2
        self.force[0, :, 4] = 100.  # Other support link is not the active finger.
        self.assertEqual(self.metrics()[2].item(), .5)
        self.force[0, 0, 1] = .3
        self.assertGreater(self.metrics()[2].item(), .5)

    def test_nonfinite_participant_cannot_pass_or_contaminate_inactive_finger(self):
        self.q[0, 0, 0] = float('nan')
        self.assertTrue(torch.isinf(self.metrics()[0]).item())
        self.part[0, 0] = False
        self.assertEqual(self.metrics()[0].item(), 0.)
        self.part.zero_()
        with self.assertRaises(ValueError): self.metrics()

    def test_single_participant_matches_previous_gather_formula(self):
        gen = torch.Generator().manual_seed(82)
        q = torch.randn(8, 2, 22, generator=gen)
        qd = torch.randn(8, 2, 22, generator=gen)
        ready = torch.randn(8, 2, 22, generator=gen)
        force = torch.rand(8, 2, 27, generator=gen)
        hands = torch.arange(8) % 2
        cols = torch.tensor([[2, 3, 4, 5]]).expand(8, -1)
        links = torch.zeros(8, 27); links[:, 6:11] = 1
        rows = torch.arange(8)
        old_index = (rows[:, None], hands[:, None], cols)
        expected = ((q[old_index]-ready[old_index]).abs().amax(-1), qd[old_index].abs().amax(-1),
                    (force[rows, hands]*links).sum(-1))
        actual = module.participant_finger_metrics(q, qd, ready, force, hands[:, None],
            cols[:, None], links[:, None], torch.ones(8, 1, dtype=torch.bool))
        for old, new in zip(expected, actual): torch.testing.assert_close(new, old, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()


def test_command_slew_limits_requests_without_changing_disabled_behavior():
    from rubik_lab.primitive_cycle import limit_finger_command
    previous=torch.tensor([[0.,.3,-.5,1.],[-1.,0.,0.,0.]])
    requested=torch.tensor([[1.,-.5,1.,0.],[1.,1.,-.5,.02]])
    assert limit_finger_command(previous,requested,0.,.025,1.) is requested
    result=limit_finger_command(previous,requested,2.,.025,1.)
    assert torch.all((result-previous).abs()<=.050001)
    assert torch.isclose(result[1,3],requested[1,3])
    assert torch.equal(previous,torch.tensor([[0.,.3,-.5,1.],[-1.,0.,0.,0.]]))


def test_pad_progress_rewards_new_progress_only():
    from rubik_lab.primitive_cycle import pad_progress_bonus
    best=torch.tensor([1.,1.,1.])
    best,reward=pad_progress_bonus(best,torch.tensor([.9,.9,.9]),torch.tensor([2.,0.,.1]),torch.tensor([2.,2.,.1]),20.)
    torch.testing.assert_close(reward,torch.tensor([2.,0.,0.]))
    best,reward=pad_progress_bonus(best,torch.tensor([1.,1.,1.]),torch.ones(3),torch.ones(3),20.)
    assert not reward.any()
    best,reward=pad_progress_bonus(best,torch.tensor([.9,.9,.9]),torch.ones(3),torch.ones(3),20.)
    assert not reward.any()
