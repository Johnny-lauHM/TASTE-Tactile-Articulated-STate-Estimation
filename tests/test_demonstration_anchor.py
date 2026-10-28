import sys
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import h5py
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source/rubik_lab'))
from rubik_lab.demonstration_anchor import attach_anchor


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor=torch.nn.Linear(250,4)
    def act_inference(self,obs):
        return self.actor(obs['policy'])


class RehearsalTests(unittest.TestCase):
    def test_failed_episodes_and_validation_grasps_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            path=str(Path(directory)/'demo.h5')
            with h5py.File(path,'w') as f:
                f['episode_uid']=np.repeat([0,1,2],3)
                f['grasp_idx']=np.repeat([10,10,20],3)
                f['primitive_done']=np.tile([False,False,True],3)
                failed=np.zeros(9,bool);failed[5]=True
                f['cycle_failed']=failed
                f['policy_obs']=np.zeros((9,250),np.float32)
                f['policy_action']=np.ones((9,4),np.float32)*0.3
            policy=Policy()
            algorithm=SimpleNamespace(policy=policy,update=lambda:{'ppo':1.0})
            info=attach_anchor(algorithm,path,excluded_grasps=[20],steps=20,lr=0.05)
            self.assertEqual(info['rows'],3)
            before=(policy.actor(torch.zeros(1,250))-0.3).square().mean().item()
            result=algorithm.update()
            after=(policy.actor(torch.zeros(1,250))-0.3).square().mean().item()
            self.assertLess(after,before)
            self.assertEqual(result['ppo'],1.0)
            self.assertIn('demo_rehearsal',result)


if __name__=='__main__':
    unittest.main()
