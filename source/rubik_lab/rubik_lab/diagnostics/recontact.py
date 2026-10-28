"""Privileged live UU residual correction; no physical state or goal writes.

Retract the original right index first, plan from the resulting actual pose,
then use the left index for one small Uprime correction. Every rejection stays
in the fixed development quota. This is not a tactile controller.
"""
import math
import torch
from rubik_lab.flick import FlickPlanner, to_cube_frame, rigid_finger_box_penetration
from rubik_lab.hand_kinematics import HandFK, axis_angle_to_mat
from rubik_lab.moves import MOVE_INDEX
from rubik_lab.primitive_execution import command_for_filtered_target
from rubik_lab.primitive_cycle import participant_finger_metrics
from .withdrawal import RadialWithdrawal


class ResidualRecontact:
    def __init__(self, env, contact_press=.003, dense_clearance=False):
        self.env=env
        n,dev=env.num_envs,env.device
        self.phase=torch.zeros(n,dtype=torch.long,device=dev)
        self.stopped=torch.zeros_like(self.phase)
        self.settled=torch.zeros_like(self.phase)
        self.used=torch.zeros(n,dtype=torch.bool,device=dev)
        self.command=torch.zeros(n,4,device=dev)
        self.cursor=torch.zeros_like(self.phase)
        self.end=torch.zeros_like(self.phase)
        self.planner=FlickPlanner(dev,steps_deg=1,max_deg=15,contact_press=contact_press)
        self.planner.fk={s:HandFK(s,dev,pts_per_link=256,seed=0) for s in ('left','right')}
        self.dense_fk=HandFK('left',dev,pts_per_link=8,seed=0,dense_fingers=('index',)) if dense_clearance else None
        self.paths=torch.zeros(n,self.planner.W,22,device=dev)
        self.target=torch.zeros(n,4,device=dev)
        self.moves=torch.full_like(self.phase,MOVE_INDEX["U'"])
        self.withdrawal=RadialWithdrawal(env,proximal_first=True,move_ids=self.moves)
        self.events=[]

    @property
    def request_original_withdrawal(self):
        return self.phase!=0

    def completion_metrics(self,q,qd,ready,forces):
        e=self.env
        hands=torch.stack([e.flick_hand[e.move_id],torch.zeros_like(e.move_id)],-1)
        cols=torch.stack([e.flick_cols[e.move_id],e.flick_cols[self.moves]],1)
        links=torch.stack([e.flick_links[e.move_id],e.flick_links[self.moves]],1)
        participants=torch.stack([torch.ones_like(self.used),self.used],-1)
        return participant_finger_metrics(q,qd,ready,forces,hands,cols,links,participants)

    def apply(self,eligible,error,omega,trial,step):
        e=self.env
        stable=eligible&(error>e.cfg.move_success_tol)&(error<math.radians(15))&(omega.abs()<.25)
        self.stopped=torch.where(stable,self.stopped+1,0)
        begin=(self.phase==0)&(self.stopped>=4)
        self.phase[begin]=1
        for i in begin.nonzero().flatten().tolist():
            self.events.append(dict(trial=trial,step=step,env=i,event='withdraw_original',error_deg=math.degrees(float(error[i]))))
        original_ready=(e.cycle_finger_error<=e.cfg.cycle_finger_tol)&(e.cycle_finger_speed<=e.cfg.cycle_finger_speed)&(e.cycle_release_force<=e.cfg.cycle_release_force)
        candidates=(eligible&(self.phase==1)&original_ready).nonzero().flatten()
        if len(candidates):
            wp,wr,q,_=to_cube_frame(e,'left',mode='actual')
            for i in candidates.tolist():
                if abs(float(error[i]))<e.cfg.move_success_tol:
                    self.phase[i]=5
                    self.events.append(dict(trial=trial,step=step,env=i,event='no_correction_needed'))
                    continue
                if not 0<float(error[i])<math.radians(15):
                    self.phase[i]=-1
                    self.events.append(dict(trial=trial,step=step,env=i,event='residual_outside_path_domain'))
                    continue
                with torch.enable_grad():
                    path,_,_,ok=self.planner.solve("U'",wp[i:i+1],wr[i:i+1],q[i:i+1],phi=-error[i:i+1],iters=60,rigid_clearance=True)
                end=self.planner.n_app+math.ceil(math.degrees(float(error[i])))
                feasible=bool(ok[0,end])
                dense_depth=None
                if feasible and self.dense_fk is not None:
                    with torch.no_grad():
                        qp=path[0,:end+1]
                        turn=torch.cat([torch.zeros(self.planner.n_app,device=e.device),torch.arange(end-self.planner.n_app+1,device=e.device)])
                        w=torch.arange(5,device=e.device,dtype=qp.dtype)/5
                        qp=torch.cat([(qp[:-1,None]+w[None,:,None]*(qp[1:]-qp[:-1])[:,None]).flatten(0,1),qp[-1:]])
                        turn=torch.cat([(turn[:-1,None]+w[None,:]*(turn[1:]-turn[:-1])[:,None]).flatten(),turn[-1:]])
                        angle=-error[i]+torch.deg2rad(turn)
                        rot=axis_angle_to_mat(torch.stack([torch.zeros_like(angle),torch.zeros_like(angle),angle],-1))
                        frames=self.dense_fk.forward(wp[i].expand(len(qp),3),wr[i].expand(len(qp),3,3),qp)
                        dense_depth=float(rigid_finger_box_penetration(self.dense_fk,frames,'index',rot).max())
                        feasible=dense_depth<=.0001

                self.events.append(dict(trial=trial,step=step,env=i,event='plan',error_deg=math.degrees(float(error[i])),feasible=feasible,dense_penetration_m=dense_depth,end=end,first_failed_waypoint=int((~ok[0]).nonzero()[0]) if (~ok[0]).any() else None))
                if not feasible:
                    self.phase[i]=-1
                    continue
                self.paths[i]=path[0];self.end[i]=end;self.phase[i]=2;self.used[i]=True
                cols=e.flick_cols[self.moves[i]]
                self.target[i]=e.finger_tgt['left'][i,cols]
        cols=e.flick_cols[self.moves]
        q=e.hands['left'].data.joint_pos[:,e.finger_ids['left']].gather(1,cols)
        moving=eligible&(self.phase==2)
        ids=moving.nonzero().flatten()
        if len(ids):
            desired=self.paths[ids,self.cursor[ids]].gather(1,cols[ids])
            self.target[ids]+=(desired-self.target[ids]).clamp(-e.step_dt,e.step_dt)
            reached=((q[ids]-desired).abs().amax(-1)<=.1)&((self.target[ids]-desired).abs().amax(-1)<1e-5)
            finished=reached&(self.cursor[ids]>=self.end[ids])
            self.phase[ids[finished]]=3
            advance=ids[reached&~finished]
            self.cursor[advance]+=1
        # Stop advancing only after the unchanged angle/speed band has held.
        near=eligible&self.used&(error.abs()<e.cfg.move_success_tol)&(omega.abs()<.25)
        self.settled=torch.where(near,self.settled+1,0)
        stop=(self.settled>=4)&((self.phase==2)|(self.phase==3))
        self.phase[stop]=4
        desired=(self.target-e.finger_grasp['left'].gather(1,cols))/e.cfg.residual_finger_range
        previous=e.action_smooth[:,6:e.per_hand].gather(1,cols)
        self.command=command_for_filtered_target(desired,previous,e.cfg.action_ema)
        ready=e._idle_offsets()[cols]/e.cfg.residual_finger_range
        push=eligible&self.used&((self.phase==2)|(self.phase==3))
        # A first-command immediate arrival still requires radial withdrawal.
        self.withdrawal.was_push|=stop
        withdrawn=self.withdrawal.apply(self.command,ready,push,self.planner)
        self.command=torch.where((self.phase==4)[:,None],withdrawn,self.command)
        return self.command
