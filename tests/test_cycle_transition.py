"""Exercise the actual sequence-transition method on tensor state, without Isaac."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from test_primitive_cycle import module

path = Path(__file__).resolve().parents[1] / "source/rubik_lab/rubik_lab/tasks/bimanual_twist/cycle_env.py"
cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "advance_cycle")
ns = {"MOVE_INDEX": {"U": 0, "U'": 1, "R": 2},
      "mu": SimpleNamespace(wrap_to_pi=lambda x: (x + torch.pi) % (2 * torch.pi) - torch.pi)}
exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), ns)
advance = ns["advance_cycle"]


class SequenceTransitionTests(unittest.TestCase):
    def make_env(self):
        e = SimpleNamespace()
        e.cfg = SimpleNamespace(cycle_terminate_on_success=False, flick_moves=("U", "U'", "R"))
        e.cycle_monitor = module.CycleMonitor(2, "cpu", 0.025, 0.25)
        e.cycle_monitor.done[:] = True
        e.move_id = torch.tensor([0, 0])
        e.move_face_id = torch.tensor([0, 0, 1])
        e.move_target = torch.tensor([-torch.pi/2, torch.pi/2, -torch.pi/2])
        e.cycle_angle_origin = torch.zeros(2)
        for name in ("move_done", "primitive_done", "layer_done_latch", "first_reach"):
            setattr(e, name, torch.ones(2, dtype=torch.bool))
        e.cycle_step = torch.ones(2, dtype=torch.long)
        e.cycle_sequence_index = torch.zeros(2, dtype=torch.long)
        e.prev_twist_err = torch.zeros(2)
        e.cycle_best_angle_error = torch.zeros(2)
        e.physical_angle = torch.tensor([-1.53, -1.54])
        e._cube_state = lambda: (None, None, e.physical_angle - e.cycle_angle_origin, None)
        return e

    def test_target_advances_without_changing_physics(self):
        e = self.make_env()
        physical = e.physical_angle.clone()
        advance(e, torch.tensor([0]), "U'")
        self.assertTrue(torch.equal(e.physical_angle, physical))
        self.assertAlmostEqual(float(e.cycle_angle_origin[0]), -torch.pi/2, places=5)
        self.assertEqual(e.cycle_angle_origin[1], 0)
        self.assertEqual(e.move_id.tolist(), [1, 0])
        self.assertEqual(e.cycle_sequence_index.tolist(), [1, 0])
        self.assertEqual(e.cycle_monitor.done.tolist(), [False, True])
        # Residual error is preserved, not zeroed by rebasing to measured angle.
        self.assertAlmostEqual(float(e.prev_twist_err[0]), 1.53, places=5)
        self.assertAlmostEqual(float(e.cycle_best_angle_error[0]),1.53,places=5)

    def test_repeated_turns_accumulate_ideal_goal_across_full_revolution(self):
        for move, mid, direction in (("U", 0, -1), ("U'", 1, 1)):
            e = self.make_env()
            e.move_id[0] = mid
            for completed in range(1, 6):
                ideal = direction * completed * torch.pi / 2
                # A measured residual must remain visible at every handoff,
                # including crossing pi and completing a full revolution.
                e.physical_angle[0] = ideal + 0.04
                e.cycle_monitor.done[0] = True
                physical = e.physical_angle.clone()
                advance(e, torch.tensor([0]), move)
                self.assertTrue(torch.equal(e.physical_angle, physical))
                self.assertAlmostEqual(float(e.cycle_angle_origin[0]), ideal, places=5)
                expected = abs(direction * torch.pi / 2 - 0.04)
                self.assertAlmostEqual(float(e.prev_twist_err[0]), expected, places=5)
                self.assertEqual(int(e.cycle_sequence_index[0]), completed)
                self.assertEqual(float(e.cycle_angle_origin[1]), 0)

    def test_cannot_switch_before_full_completion(self):
        e = self.make_env()
        e.cycle_monitor.done[0] = False
        with self.assertRaises(RuntimeError):
            advance(e, torch.tensor([0]), "U'")

    def test_cross_face_transition_is_explicitly_rejected(self):
        e = self.make_env()
        with self.assertRaises(ValueError):
            advance(e, torch.tensor([0]), "R")

    def test_auto_reset_mode_cannot_be_misreported_as_continuous(self):
        e = self.make_env()
        e.cfg.cycle_terminate_on_success = True
        with self.assertRaises(RuntimeError):
            advance(e, torch.tensor([0]), "U'")


if __name__ == "__main__":
    unittest.main()
