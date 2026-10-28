"""Regression against actual teacher source: both turn directions advance equally."""
import ast
from pathlib import Path
from types import SimpleNamespace
import torch

source=Path(__file__).resolve().parents[1]/'source/rubik_lab/rubik_lab/teacher.py'
cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='ScriptedTeacher')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_flick_progress')
namespace={'torch':torch}
exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),namespace)

def progress(degrees):
    target=torch.tensor([-torch.pi/2,torch.pi/2])
    angle=torch.deg2rad(torch.tensor([-degrees,degrees]))
    env=SimpleNamespace(move_target=target,move_id=torch.tensor([0,1]),signed_err=target-angle)
    return namespace['_flick_progress'](SimpleNamespace(env=env))

def test_both_turns_advance_to_same_waypoint():
    for angle in (0.,20.,60.,87.,90.,100.):
        torch.testing.assert_close(progress(angle),torch.deg2rad(torch.tensor([angle,angle])))

def test_wrong_direction_is_not_positive_progress():
    assert (progress(-20.)<0).all()

def test_reverse_move_can_reach_release_threshold():
    assert (progress(88.)>=torch.deg2rad(torch.tensor(87.))).all()

def test_initial_planning_and_tracking_share_direction_rule():
    for name in ('_flick_plan','_act_flick'):
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='_flick_progress' for n in ast.walk(fn))
