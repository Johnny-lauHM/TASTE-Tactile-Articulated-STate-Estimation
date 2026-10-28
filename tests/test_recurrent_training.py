import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source/rubik_lab'))
from rubik_lab.recurrent_training import generalized_advantage,recurrent_actions,continuation_advantage
from rubik_lab.blind_actor import BlindPrimitiveActor

class RecurrentTrainingTests(unittest.TestCase):
    def test_terminal_returns_do_not_bootstrap_reset_episode(self):
        reward=torch.tensor([[1.],[2.],[100.]])
        values=torch.zeros_like(reward)
        done=torch.tensor([[False],[True],[False]])
        advantage,returns=generalized_advantage(reward,values,done,torch.tensor([10.]),gamma=1.,lam=1.)
        self.assertEqual(returns[:,0].tolist(),[3.,2.,110.])

    def test_replay_matches_independent_episodes_after_partial_reset(self):
        torch.manual_seed(12)
        actor=BlindPrimitiveActor(16)
        features=torch.randn(5,2,173)
        reset=torch.zeros(5,2,dtype=torch.bool);reset[3,0]=True
        hidden=torch.zeros(1,2,16)
        actual,_=recurrent_actions(actor,features,hidden,reset)
        before,_=actor(features[:3,0].unsqueeze(0))
        after,_=actor(features[3:,0].unsqueeze(0))
        other,_=actor(features[:,1].unsqueeze(0))
        self.assertTrue(torch.allclose(actual[:,0],torch.cat([before[0],after[0]]),atol=1e-6))
        self.assertTrue(torch.allclose(actual[:,1],other[0],atol=1e-6))
        actual.sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters()))


    def test_prefix_rewards_and_values_cannot_enter_training_returns(self):
        reward=torch.tensor([[1000.],[2000.],[1.],[2.],[3000.],[3.]])
        phase=torch.tensor([[False],[False],[True],[True],[False],[True]])
        values=torch.where(phase,torch.zeros_like(reward),torch.full_like(reward,999.))
        done=torch.zeros_like(phase)
        advantage,returns=continuation_advantage(reward,values,done,torch.tensor([4.]),phase,torch.tensor([True]),gamma=1.,lam=1.)
        torch.testing.assert_close(returns[:,0],torch.tensor([0.,0.,3.,2.,0.,7.]))
        torch.testing.assert_close(advantage,returns)

    def test_terminal_and_prefix_bootstrap_are_both_cut(self):
        reward=torch.tensor([[1.,1.],[2.,2.]])
        phase=torch.ones_like(reward,dtype=torch.bool)
        done=torch.tensor([[True,False],[False,False]])
        _,returns=continuation_advantage(reward,torch.zeros_like(reward),done,torch.tensor([999.,999.]),phase,torch.tensor([False,False]),gamma=1.,lam=1.)
        torch.testing.assert_close(returns,torch.tensor([[1.,3.],[2.,2.]]))

    def test_continuation_mask_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            continuation_advantage(torch.zeros(2,1),torch.zeros(2,1),torch.zeros(2,1,dtype=torch.bool),torch.zeros(1),torch.zeros(1,1,dtype=torch.bool),torch.zeros(1,dtype=torch.bool))

if __name__=='__main__':unittest.main()
