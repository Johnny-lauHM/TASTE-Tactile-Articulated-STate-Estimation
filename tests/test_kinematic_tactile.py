"""Kinematic tactile features must match hand FK and streaming preprocessing."""
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source/rubik_lab'))
from rubik_lab.detector.kinematic_features import PadWrenchRotation
from rubik_lab.hand_kinematics import HandFK
from rubik_lab.detector.model import CompletionDetector
from rubik_lab.detector.streaming import StreamingReadiness
from rubik_lab.moves import ALL_MOVES


def test_axis_rotation_matches_existing_hand_fk():
    torch.manual_seed(71)
    q=torch.randn(2,2,22)*.2
    wrench=torch.randn(2,2,5,6)
    transform=PadWrenchRotation()
    actual=transform(wrench,q)
    for h,side in enumerate(('left','right')):
        fk=HandFK(side,device='cpu',pts_per_link=1)
        frames=fk.forward(torch.zeros(2,3),torch.eye(3).repeat(2,1,1),q[:,h])
        for f,finger in enumerate(('thumb','index','middle','ring','pinky')):
            rotation=frames[side+'_'+finger+'_elastomer'][1]
            expected=torch.cat([(rotation@wrench[:,h,f,:3,None]).squeeze(-1),
                                (rotation@wrench[:,h,f,3:,None]).squeeze(-1)],-1)
            assert torch.allclose(actual[:,h,f],expected,atol=1e-6)
    assert torch.allclose(actual[...,:3].norm(dim=-1),wrench[...,:3].norm(dim=-1),atol=1e-6)
    assert torch.allclose(actual[...,3:].norm(dim=-1),wrench[...,3:].norm(dim=-1),atol=1e-6)


def test_kinematic_streaming_matches_training_features():
    torch.manual_seed(73)
    transform=PadWrenchRotation()
    model=CompletionDetector(ft_dim=120).eval()
    payload={'format':'streaming_tactile_ready_v1','model':model.state_dict(),'hold_steps':2,
             'kinematic_ft':True,'ft_dim':120,'kinematics_urdf_sha256':transform.urdf_sha256}
    gate=StreamingReadiness(payload,2,'cpu')
    ft=torch.randn(2,4,2,5,6);q=torch.randn(2,4,2,22);target=q+.1
    moves=torch.tensor([6,7])
    x={'ft':torch.cat([ft.flatten(2),transform(ft,q).flatten(2)],-1)*.2,
       'prop':torch.cat([q.flatten(2),(target-q).flatten(2)],-1),
       'move':torch.nn.functional.one_hot(moves,len(ALL_MOVES)).float()}
    with torch.no_grad():expected=model.forward_sequence(x)[0].softmax(-1)[:,:,1]
    actual=[]
    for step in range(4):
        actual.append(gate.step(tactile_ft=ft[:,step],finger_pos=q[:,step],finger_tgt=target[:,step],move_id=moves)['ready_probability'])
    assert torch.allclose(torch.stack(actual,1),expected,atol=1e-6)
