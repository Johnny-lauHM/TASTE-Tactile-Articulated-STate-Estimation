"""Privileged diagnostic withdrawal: outward pad retreat, then rate-limited ready.

Uses simulator cube pose/contact only to diagnose mechanics; not a blind policy.
"""
import torch
from rubik_lab.hand_kinematics import quat_to_mat

class RadialWithdrawal:
    def __init__(self, env, distance=0.012, duration=0.3, return_rate=1.0, proximal_first=False, move_ids=None):
        self.move_ids=move_ids
        self.proximal_first=proximal_first
        self.env=env; self.distance=distance; self.duration=duration; self.return_rate=return_rate
        n,dev=env.num_envs,env.device
        self.stage=torch.zeros(n,dtype=torch.long,device=dev)
        self.was_push=torch.zeros(n,dtype=torch.bool,device=dev)
        self.elapsed=torch.zeros(n,device=dev)
        self.start=torch.zeros(n,3,device=dev); self.offset=torch.zeros_like(self.start)
        self.qcmd=torch.zeros(n,22,device=dev)
        self.ready_command=torch.zeros(n,4,device=dev)

    def apply(self, actions, ready, push, planner):
        e=self.env; rows=torch.arange(e.num_envs,device=e.device)
        moves=e.move_id if self.move_ids is None else self.move_ids
        hand=e.flick_hand[moves]; cols=e.flick_cols[moves]
        begin=self.was_push&~push
        self.stage[push]=0; self.stage[begin]=1; self.elapsed[begin]=0
        pos,quat,_,_=e._cube_state(); core=pos+e.scene.env_origins
        axis=quat_to_mat(quat)[:,:,2]
        force=e.cube_contacts.forces()[:,1]
        release=(force[rows,hand]*e.flick_links[moves]).sum(-1)
        out=actions.clone()
        for h,side,move in [(0,'left',"U'"),(1,'right','U')]:
            ids=((hand==h)&(self.stage>0)).nonzero().flatten()
            if not len(ids):continue
            wp,wq=e.ctrl[side].ee_pose_w(); wr=quat_to_mat(wq)
            q=e.hands[side].data.joint_pos[:,e.finger_ids[side]]
            starts=ids[begin[ids]]
            if len(starts):
                pad,_=planner.fk[side].pads(planner.fk[side].forward(wp[starts],wr[starts],q[starts]))['index']
                radial=pad-core[starts];radial-=axis[starts]*(radial*axis[starts]).sum(-1,keepdim=True)
                self.start[starts]=pad
                self.offset[starts]=self.distance*radial/radial.norm(dim=-1,keepdim=True).clamp(min=1e-6)
                self.qcmd[starts]=q[starts]
            clear=ids[(self.stage[ids]==1)&(self.elapsed[ids]>=self.duration)&(release[ids]<=0.5)]
            if len(clear):
                self.stage[clear]=2
                self.ready_command[clear]=(q[clear].gather(1,cols[clear])-e.finger_grasp[side][clear].gather(1,cols[clear]))/e.cfg.residual_finger_range
            retreat=ids[self.stage[ids]==1]
            if len(retreat):
                fraction=((self.elapsed[retreat]+e.step_dt)/self.duration).clamp(max=1)[:,None]
                target=self.start[retreat]+fraction*self.offset[retreat]
                with torch.enable_grad():
                    solved=planner.track(move,wp[retreat],wr[retreat],q[retreat],self.qcmd[retreat],target,iters=20)
                self.qcmd[retreat]=solved
                out[retreat]=(solved.gather(1,cols[retreat])-e.finger_grasp[side][retreat].gather(1,cols[retreat]))/e.cfg.residual_finger_range
            returning=ids[self.stage[ids]>=2]
            if len(returning):
                limit=self.return_rate*e.step_dt/e.cfg.residual_finger_range
                desired=ready[returning].clone()
                if self.proximal_first:
                    actual=(q[returning].gather(1,cols[returning])-e.finger_grasp[side][returning].gather(1,cols[returning]))/e.cfg.residual_finger_range
                    reached=(self.stage[returning]==2)&((actual[:,0]-ready[returning,0]).abs()*e.cfg.residual_finger_range<=0.1)
                    self.stage[returning[reached]]=3
                    first=self.stage[returning]==2
                    desired[first,1:]=self.ready_command[returning[first],1:]
                self.ready_command[returning]+=(desired-self.ready_command[returning]).clamp(-limit,limit)
                out[returning]=self.ready_command[returning]
        self.elapsed+=e.step_dt*(self.stage>0)
        self.was_push=push.clone()
        return out.clamp(-1,1)
