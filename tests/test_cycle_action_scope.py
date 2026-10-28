"""Guard the real cycle action expansion against hidden privileged assistance."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
path=Path(__file__).resolve().parents[1]/'source/rubik_lab/rubik_lab/tasks/bimanual_twist/cycle_env.py'
cls=next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.ClassDef))
methods=[x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in ('_idle_offsets','_expand_flick_action')]
ns={'torch':torch};exec(compile(ast.fix_missing_locations(ast.Module(body=methods,type_ignores=[])),str(path),'exec'),ns)
class ForbiddenContacts:
 def forces(self):raise AssertionError('Privileged contact state reached action controller')

class CycleActionScopeTests(unittest.TestCase):
 def test_truth_latch_or_contacts_cannot_retract_or_change_actor_command(self):
  e=SimpleNamespace(num_envs=2,per_hand=28,device='cpu')
  e.cfg=SimpleNamespace(cycle_idle_lift=.3,residual_finger_range=.4,flick_hold_press=.15)
  e.finger_flex={name:torch.zeros(22) for name in ('index','pinky','thumb','middle')}
  for k,name in enumerate(e.finger_flex):e.finger_flex[name][4*k:4*k+4]=1
  e.move_id=torch.tensor([0,1]);e.flick_hand=torch.tensor([1,0]);e.flick_cols=torch.tensor([[0,1,2,3],[0,1,2,3]])
  e._idle_offsets=lambda:ns['_idle_offsets'](e)
  e.cube_contacts=ForbiddenContacts();e.layer_done_latch=torch.zeros(2,dtype=torch.bool);e.primitive_done=torch.zeros(2,dtype=torch.bool)
  actions=torch.tensor([[.1,.2,.3,.4],[-.4,-.3,-.2,-.1]])
  before=ns['_expand_flick_action'](e,actions)
  e.layer_done_latch[:]=True;e.primitive_done[:]=True
  after=ns['_expand_flick_action'](e,actions)
  self.assertTrue(torch.equal(before,after))
  self.assertTrue(torch.equal(after[0,34:38],actions[0]))
  self.assertTrue(torch.equal(after[1,6:10],actions[1]))
  self.assertEqual(int(torch.count_nonzero(after[:,:6])),0)
  self.assertEqual(int(torch.count_nonzero(after[:,28:34])),0)

 def test_terminal_hold_cannot_override_third_action(self):
  runner=path.parents[2]/'diagnostics'/'continuation.py'
  tree=ast.parse(runner.read_text())
  node=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='command' for t in n.targets) and 'held_command' in ast.unparse(n.value))
  actor=torch.arange(16,dtype=torch.float32).reshape(4,4)
  scope=dict(torch=torch,command=actor.clone(),held_command=torch.full((4,4),-3.),hold_latched=torch.ones(4,dtype=torch.bool),active=torch.tensor([True,True,True,False]),stage=torch.tensor([0,1,2,1]))
  exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(runner),'exec'),scope)
  self.assertTrue(torch.equal(scope['command'][[0,2,3]],actor[[0,2,3]]))
  self.assertTrue(torch.equal(scope['command'][1],torch.full((4,),-3.)))

 def test_motor_hold_changes_only_requested_actuator_targets(self):
  runner=path.parents[2]/'diagnostics'/'continuation.py';tree=ast.parse(runner.read_text())
  cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TraceEnv')
  method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_pre_physics_step')
  nodes=method.body[1:3]
  base=torch.arange(44,dtype=torch.float32).reshape(2,22);left=base.clone();right=base.clone()
  cols=torch.tensor([5,6,7,8]);values=torch.tensor([.1,.2,.3,.4]);e=SimpleNamespace(finger_tgt={'left':left,'right':right},post_return_hold_target=(1,'left',cols,values),physical_q=base.clone())
  exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(runner),'exec'),{'self':e})
  expected=base.clone();expected[1,cols]=values
  self.assertTrue(torch.equal(left,expected));self.assertTrue(torch.equal(right,base));self.assertTrue(torch.equal(e.physical_q,base))

 def test_escape_changes_only_lateral_motor_with_rate_and_limit_guard(self):
  runner=path.parents[2]/'diagnostics'/'continuation.py';tree=ast.parse(runner.read_text())
  fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='escape_motor_target')
  scope={};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(runner),'exec'),scope)
  f=scope['escape_motor_target'];original=torch.tensor([.75,.24,.80,.26]);lo=torch.full((4,),-.3);hi=torch.ones(4)
  previous=original.clone()
  for k in range(1,25):
   current=f(original,k*.025,lo,hi)
   self.assertTrue(torch.equal(current[[0,2,3]],original[[0,2,3]]))
   self.assertLessEqual(float((current-previous).abs().max()),.025001)
   previous=current
  self.assertAlmostEqual(float(current[1]),-.16,places=6)
  self.assertTrue(torch.equal(original,torch.tensor([.75,.24,.80,.26])))
  with self.assertRaises(ValueError):f(original,1.,torch.zeros(4),hi)
  with self.assertRaises(ValueError):f(original,-.1,lo,hi)

 def test_preload_removal_uses_latched_reference_and_rate_limits(self):
  runner=path.parents[2]/'diagnostics'/'continuation.py';tree=ast.parse(runner.read_text())
  fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='relax_motor_target')
  scope={};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(runner),'exec'),scope)
  f=scope['relax_motor_target'];original=torch.tensor([.75,.24,.8,.26]);reference=torch.tensor([.81,.19,1.11,.40]);lo=torch.full((4,),-.3);hi=torch.full((4,),1.7)
  previous=original.clone()
  for k in range(1,20):
   current=f(original,reference,k*.025,lo,hi)
   self.assertLessEqual(float((current-previous).abs().max()),.025001)
   previous=current
  self.assertTrue(torch.allclose(current,reference))
  self.assertTrue(torch.equal(original,torch.tensor([.75,.24,.8,.26])))
  with self.assertRaises(ValueError):f(original,reference,1.,lo,torch.ones(4))

 def test_fourth_stage_waits_for_third_completion_and_disables_old_targets(self):
  runner=path.parents[2]/'diagnostics'/'continuation.py';tree=ast.parse(runner.read_text())
  names=['append_fourth','good_third','final_stage','ended']
  nodes=[next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)) for name in names]
  scope=dict(torch=torch,a=SimpleNamespace(after_left_return_move='U'),append_tail=torch.tensor([True,True,False,True]),active=torch.ones(4,dtype=torch.bool),full=torch.tensor([True,False,True,True]),stage=torch.tensor([2,2,1,3]),truncated=torch.zeros(4,dtype=torch.bool),invalid=torch.zeros(4,dtype=torch.bool),timeout=torch.zeros(4,dtype=torch.bool))
  exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(runner),'exec'),scope)
  self.assertEqual(scope['good_third'].tolist(),[True,False,False,False])
  self.assertEqual(scope['ended'].tolist(),[False,False,True,True])
  block=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='good_third.any()')
  clear=[n for n in block.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr in ('post_return_hold_target','whole_hand_probe_target') for t in n.targets)]
  e=SimpleNamespace(post_return_hold_target='stale',whole_hand_probe_target='stale',physical_q=torch.ones(22))
  exec(compile(ast.fix_missing_locations(ast.Module(body=clear,type_ignores=[])),str(runner),'exec'),{'u':e})
  self.assertIsNone(e.post_return_hold_target);self.assertIsNone(e.whole_hand_probe_target);self.assertTrue(torch.equal(e.physical_q,torch.ones(22)))

 def test_brake_trigger_requires_approach_not_overshoot(self):
  import numpy as np
  runner=path.parents[2]/'diagnostics'/'continuation.py';tree=ast.parse(runner.read_text())
  fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='preload_brake_trigger');scope={'np':np}
  exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(runner),'exec'),scope);f=scope['preload_brake_trigger']
  self.assertTrue(f(np.deg2rad(15),10,20,1))
  for error,speed in [(21,10),(-1,10),(10,-1),(10,0),(float('nan'),1)]:self.assertFalse(f(np.deg2rad(error),speed,20,1))

 def test_runtime_brake_call_rejects_previous_primitive_snapshot(self):
  import numpy as np
  runner=path.parents[2]/'diagnostics'/'continuation.py';tree=ast.parse(runner.read_text())
  fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='preload_brake_trigger');scope={'np':np}
  exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(runner),'exec'),scope)
  call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='preload_brake_trigger')
  scope.update(s0={'twist_error':torch.tensor([np.deg2rad(2.1476438)]),'twist_velocity':torch.tensor([1.320454e-5])},i=0,a=SimpleNamespace(diagnostic_preload_at_error_deg=20),u=SimpleNamespace(cycle_step=torch.tensor([0])))
  expression=compile(ast.fix_missing_locations(ast.Expression(body=call)),str(runner),'eval')
  self.assertFalse(eval(expression,scope))
  scope['u'].cycle_step[0]=1
  self.assertTrue(eval(expression,scope))
