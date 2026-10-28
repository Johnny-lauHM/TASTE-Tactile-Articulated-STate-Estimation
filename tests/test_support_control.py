"""Exercise actual support IK: each selected finger must move toward its own pad target."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import pytest
from rubik_lab.constants import FINGER_JOINTS
from rubik_lab.hand_kinematics import HandFK,quat_to_mat,axis_angle_to_mat,mat_to_quat
from rubik_lab.diagnostics.support_control import SupportTiltController

@pytest.mark.parametrize('actual_pose',[False,True])
@pytest.mark.parametrize('finger',['middle','thumb','ring'])
def test_selected_finger_moves_toward_tilt_correction(finger,actual_pose):
    cache=np.load(Path(__file__).resolve().parents[1]/'cache/g70s_cycle_demo1_v3.npz')
    tensor=lambda key:torch.tensor(cache[key][:1],dtype=torch.float32)
    core=tensor('cube_pos');goal=tensor('cube_quat')
    actual=mat_to_quat(axis_angle_to_mat(torch.tensor([[.1,0.,0.]]))@quat_to_mat(goal))
    env=SimpleNamespace(num_envs=1,device='cpu',per_hand=28,scene=SimpleNamespace(env_origins=torch.zeros(1,3)),cube_quat_goal=goal,cfg=SimpleNamespace(residual_finger_range=1.),finger_grasp={},ctrl={})
    env._cube_state=lambda:(core,actual,None,None)
    fk={};env.hands={};env.finger_ids={}
    for side in ('left','right'):
        fk[side]=HandFK(side,device='cpu',pts_per_link=1)
        env.finger_grasp[side]=tensor(side+'_fingers')
        env.finger_ids[side]=torch.arange(22)
        env.hands[side]=SimpleNamespace(data=SimpleNamespace(joint_pos=(env.finger_grasp[side]-.05).clamp(min=fk[side].lower,max=fk[side].upper)))
        env.ctrl[side]=SimpleNamespace(ee_pose_w=lambda side=side:(tensor(side+'_ee_pos'),tensor(side+'_ee_quat')))
    full=torch.zeros(1,56);out=SupportTiltController(env,.5,finger=finger,actual_pose=actual_pose).apply(full,torch.tensor([True]),SimpleNamespace(fk=fk))
    axis=quat_to_mat(goal)[:,:,2];rot=torch.tensor([[np.sin(.1),0.,0.]],dtype=torch.float32);rot-=axis*(rot*axis).sum(-1,keepdim=True)
    cols=[i for i,n in enumerate(FINGER_JOINTS) if n.startswith(finger+'_')]
    for h,side in enumerate(('left','right')):
        wp,wq=env.ctrl[side].ee_pose_w();q=env.hands[side].data.joint_pos if actual_pose else env.finger_grasp[side];dq=out[:,h*28+6:h*28+28]
        old,_=fk[side].pads(fk[side].forward(wp,quat_to_mat(wq),q))[finger]
        new,_=fk[side].pads(fk[side].forward(wp,quat_to_mat(wq),q+dq))[finger]
        desired=torch.linalg.cross(-.5*rot,old-core)
        assert float(((new-old)*desired).sum())>1e-8
        assert float(dq[:,cols].abs().max())>1e-4
        other=[i for i in range(22) if i not in cols]
        assert torch.equal(dq[:,other],torch.zeros_like(dq[:,other]))
