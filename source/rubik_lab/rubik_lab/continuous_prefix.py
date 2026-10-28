"""Checked continuous handoff shared by diagnostic and training rollouts.
Only the ideal move goal and primitive completion bookkeeping may change.
"""
import torch


def _preserved_state(env):
    state={name:getattr(env,name).clone() for name in (
        'cube_center','cube_quat_goal','cube_quat0','action_smooth',
        'peak_pos','peak_rot','cycle_wrist_pos_peak','cycle_wrist_rot_peak')}
    for name in ('ee_home_pos','ee_home_quat','ee_grasp_tgt','ee_grasp_quat',
                 'finger_grasp','finger_tgt','ee_pos_tgt','ee_quat_tgt'):
        for side,value in getattr(env,name).items():state[name+'/'+side]=value.clone()
    for name,asset in [('cube',env.cube),*env.hands.items()]:
        for field in ('root_state_w','body_state_w','joint_pos','joint_vel'):
            state[name+'/'+field]=getattr(asset.data,field).clone()
    for i,value in enumerate(env.tactile._ft):state['tactile_ft/'+str(i)]=value.clone()
    if env.tactile._img is not None:
        for i,value in enumerate(env.tactile._img):state['tactile_image/'+str(i)]=value.clone()
    state['tactile_timing']=env.tactile.timing.clone()
    for key in ('step_count','_pending_step','_has_pending'):
        state['tactile/'+key]=getattr(env.tactile,key)
    return state


def advance_preserving_state(env,ids,move):
    """Diagnostic/training truth-gated handoff; never a blind trigger."""
    before=_preserved_state(env)
    origin=env.cycle_angle_origin.clone();old_move=env.move_id.clone()
    expected=origin.clone();expected[ids]+=env.move_target[old_move[ids]]
    env.advance_cycle(ids,move)
    after=_preserved_state(env)
    changed=[key for key,value in before.items() if
             not (torch.equal(value,after[key]) if isinstance(value,torch.Tensor) else value==after[key])]
    if changed:raise RuntimeError('Continuous handoff changed preserved state: '+', '.join(changed))
    if not torch.equal(env.cycle_angle_origin,expected):
        raise RuntimeError('Handoff changed ideal angle origin incorrectly')
    return {'environments':len(ids),'preserved_fields':len(before)}
