from types import SimpleNamespace
import torch
import pytest
from rubik_lab.continuous_prefix import advance_preserving_state


def fixture(changed=None):
    e=SimpleNamespace()
    for name in ('cube_center','cube_quat_goal','cube_quat0','action_smooth','peak_pos','peak_rot','cycle_wrist_pos_peak','cycle_wrist_rot_peak'):
        setattr(e,name,torch.ones(2,3))
    for name in ('ee_home_pos','ee_home_quat','ee_grasp_tgt','ee_grasp_quat','finger_grasp','finger_tgt','ee_pos_tgt','ee_quat_tgt'):
        setattr(e,name,{s:torch.ones(2,3) for s in ('left','right')})
    def asset():return SimpleNamespace(data=SimpleNamespace(**{k:torch.ones(2,3) for k in ('root_state_w','body_state_w','joint_pos','joint_vel')}))
    e.cube=asset();e.hands={s:asset() for s in ('left','right')}
    e.tactile=SimpleNamespace(_ft=[torch.ones(2,6),torch.ones(2,6)],_img=None,timing=torch.ones(2,2),step_count=12,_pending_step=10,_has_pending=True)
    e.cycle_angle_origin=torch.tensor([0.,2.]);e.move_id=torch.tensor([0,0]);e.move_target=torch.tensor([-1.57])
    def advance(ids,move):
        e.cycle_angle_origin[ids]+=e.move_target[e.move_id[ids]]
        if changed=='velocity':e.cube.data.joint_vel[ids]=0
        if changed=='sensor':e.tactile._ft[1][ids]=0
        if changed=='reference':e.cube_center[ids]=0
        if changed=='ema':e.action_smooth[ids]=0
        if changed=='neighbor':e.cycle_angle_origin[1]=0
    e.advance_cycle=advance
    return e


def test_preserves_live_state_and_updates_only_selected_ideal_goal():
    e=fixture();result=advance_preserving_state(e,torch.tensor([0]),'U')
    assert result['environments']==1
    assert e.cycle_angle_origin[1]==2
    assert torch.equal(e.cube.data.joint_vel,torch.ones(2,3))

@pytest.mark.parametrize('changed',['velocity','sensor','reference','ema','neighbor'])
def test_rejects_reset_or_reference_refresh(changed):
    with pytest.raises(RuntimeError):advance_preserving_state(fixture(changed),torch.tensor([0]),'U')
