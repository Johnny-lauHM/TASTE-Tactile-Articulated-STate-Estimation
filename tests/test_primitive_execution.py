"""A ready hold must not consume recurrent primitive time."""
import importlib.util
from pathlib import Path
import torch

spec=importlib.util.spec_from_file_location("primitive_execution",Path(__file__).resolve().parents[1]/"source/rubik_lab/rubik_lab/primitive_execution.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class StatefulActor:
    def __init__(self):self.state=torch.zeros(2)
    def act(self):
        self.state+=1
        return self.state[:,None].expand(-1,4).clone()
    def reset(self,ids):self.state[ids]=0

def test_wait_does_not_consume_motion_phase_or_reset_running_neighbor():
    actor=StatefulActor();ready=torch.full((2,4),-0.3)
    for k in range(4):
        out=m.apply_ready_hold(actor.act(),ready,torch.tensor([True,False]),[actor])
        torch.testing.assert_close(out[0],ready[0])
        torch.testing.assert_close(out[1],torch.full((4,),float(k+1)))
    out=m.apply_ready_hold(actor.act(),ready,torch.tensor([False,False]),[actor])
    torch.testing.assert_close(out[:,0],torch.tensor([1.,5.]))

def test_multiple_actors_share_wait_boundary_and_stateless_actor_is_supported():
    left,right=StatefulActor(),StatefulActor()
    left.act();right.act();left.act()
    original=torch.ones(2,4);ready=torch.zeros_like(original)
    out=m.apply_ready_hold(original,ready,torch.tensor([False,True]),[left,right,lambda x:x])
    torch.testing.assert_close(left.state,torch.tensor([2.,0.]))
    torch.testing.assert_close(right.state,torch.tensor([1.,0.]))
    torch.testing.assert_close(original,torch.ones(2,4))
    torch.testing.assert_close(out,torch.tensor([[1.,1.,1.,1.],[0.,0.,0.,0.]]))


def test_filtered_target_roundtrip_and_unreachable_target_remains_bounded():
    previous=torch.tensor([-.2,.4,-1.,1.])
    target=torch.tensor([.1,.3,1.,-1.])
    command=m.command_for_filtered_target(target,previous,.5)
    applied=.5*command+.5*previous
    torch.testing.assert_close(applied[:2],target[:2])
    torch.testing.assert_close(command[2:],torch.tensor([1.,-1.]))
    torch.testing.assert_close(applied[2:],torch.zeros(2))
    torch.testing.assert_close(previous,torch.tensor([-.2,.4,-1.,1.]))

def test_filter_compensation_rejects_noninvertible_or_invalid_settings():
    import pytest
    for alpha,limit in [(0,1),(-.1,1),(1.1,1),(.5,0)]:
        with pytest.raises(ValueError):
            m.command_for_filtered_target(torch.zeros(1),torch.zeros(1),alpha,limit)
