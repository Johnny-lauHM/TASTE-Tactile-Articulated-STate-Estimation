"""Privileged middle-finger tilt compensation for mechanical diagnosis only."""
import torch
from rubik_lab.constants import FINGER_JOINTS
from rubik_lab.hand_kinematics import quat_to_mat

class SupportTiltController:
    def __init__(self, env, gain, max_shift=0.005, finger="middle", actual_pose=False):
        self.env=env;self.gain=gain;self.max_shift=max_shift;self.finger=finger;self.actual_pose=actual_pose
        self.cols=[i for i,name in enumerate(FINGER_JOINTS) if name.startswith(finger+'_')]

    def apply(self, full, active, planner):
        e=self.env;out=full.clone();ids=active.nonzero().flatten()
        if not len(ids):return out
        core,quat,_,_=e._cube_state();core=core+e.scene.env_origins
        current=quat_to_mat(quat); goal=quat_to_mat(e.cube_quat_goal)
        relative=current@goal.transpose(1,2)
        error=0.5*torch.stack([relative[:,2,1]-relative[:,1,2],relative[:,0,2]-relative[:,2,0],relative[:,1,0]-relative[:,0,1]],-1)
        axis=goal[:,:,2];error=error-axis*(error*axis).sum(-1,keepdim=True)
        for h,side in enumerate(('left','right')):
            fk=planner.fk[side];wp,wq=e.ctrl[side].ee_pose_w();wr=quat_to_mat(wq)
            sl=slice(h*e.per_hand+6,(h+1)*e.per_hand)
            nominal=(e.finger_grasp[side]+full[:,sl]*e.cfg.residual_finger_range).clamp(min=fk.lower,max=fk.upper)
            seed=(e.hands[side].data.joint_pos[:,e.finger_ids[side]].clamp(min=fk.lower,max=fk.upper) if self.actual_pose else nominal)
            with torch.no_grad():
                pad,_=fk.pads(fk.forward(wp[ids],wr[ids],seed[ids]))[self.finger]
                shift=torch.linalg.cross(-self.gain*error[ids],pad-core[ids])
                shift*=torch.clamp(self.max_shift/shift.norm(dim=-1,keepdim=True).clamp(min=1e-9),max=1)
                target=pad+shift
            with torch.enable_grad():
                raw=seed[ids][:,self.cols].detach().clone().requires_grad_(True)
                opt=torch.optim.Adam([raw],lr=0.01)
                for _ in range(20):
                    q=seed[ids].detach().clone()
                    q[:,self.cols]=raw.clamp(min=fk.lower[self.cols],max=fk.upper[self.cols])
                    pos,_=fk.pads(fk.forward(wp[ids],wr[ids],q))[self.finger]
                    loss=((pos-target)**2).sum()+1e-5*((raw-seed[ids][:,self.cols])**2).sum()
                    opt.zero_grad();loss.backward();opt.step()
                solved=raw.detach().clamp(min=fk.lower[self.cols],max=fk.upper[self.cols])
            if self.actual_pose:
                solved=(nominal[ids][:,self.cols]+solved-seed[ids][:,self.cols]).clamp(min=fk.lower[self.cols],max=fk.upper[self.cols])
            columns=torch.tensor(self.cols,device=e.device)+h*e.per_hand+6
            out[ids[:,None],columns[None,:]]=(solved-e.finger_grasp[side][ids][:,self.cols])/e.cfg.residual_finger_range
        return out.clamp(-1,1)


class WristTiltController:
    """Truth-based tilt diagnostic; bounded rotation about each fixed wrist position."""
    def __init__(self, env, gain, max_angle=0.0523598776, rate=0.1745329252):
        self.env=env;self.gain=gain;self.max_angle=max_angle;self.rate=rate
        self.command=torch.zeros(env.num_envs,3,device=env.device)

    def reset(self):
        self.command.zero_()

    def apply(self, active):
        from isaaclab.utils.math import quat_from_angle_axis, quat_apply
        e=self.env
        _,quat,_,_=e._cube_state()
        current=quat_to_mat(quat);goal=quat_to_mat(e.cube_quat_goal)
        relative=current@goal.transpose(1,2)
        error=0.5*torch.stack([relative[:,2,1]-relative[:,1,2],relative[:,0,2]-relative[:,2,0],relative[:,1,0]-relative[:,0,1]],-1)
        axis=goal[:,:,2];error-=axis*(error*axis).sum(-1,keepdim=True)
        desired=-self.gain*error*active[:,None]
        desired*=torch.clamp(self.max_angle/desired.norm(dim=-1,keepdim=True).clamp(min=1e-9),max=1)
        delta=desired-self.command
        delta*=torch.clamp(self.rate*e.step_dt/delta.norm(dim=-1,keepdim=True).clamp(min=1e-9),max=1)
        self.command+=delta
        dq=quat_from_angle_axis(self.command.norm(dim=-1),self.command)
        result=torch.zeros(e.num_envs,2,6,device=e.device)
        for h,side in enumerate(('left','right')):
            offset=e.ee_grasp_tgt[side]-e.cube_center
            # Cancel the environment's rotation-about-cube translation exactly.
            result[:,h,:3]=(offset-quat_apply(dq,offset))/e.cfg.residual_pos_range
            result[:,h,3:]=self.command/e.cfg.residual_rot_range
        if bool((result.abs()>1).any()):raise RuntimeError('Wrist correction exceeds residual command range')
        return result
