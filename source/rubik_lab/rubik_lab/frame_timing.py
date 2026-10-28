"""Per-environment provenance of sampled, delayed tactile frames (no physics state)."""
import torch

class SensorFrameClock:
    def __init__(self,n,device,physics_dt):
        self.physics_dt=physics_dt
        self.pending=torch.full((n,),-1,dtype=torch.long,device=device)
        self.delivered=self.pending.clone()

    def capture(self,step):
        self.pending.fill_(step)

    def deliver(self):
        self.delivered.copy_(self.pending)

    def reset(self,ids):
        self.pending[ids]=-1
        self.delivered[ids]=-1

    def observe(self,step):
        valid=self.delivered>=0
        age=torch.where(valid,(step-self.delivered)*self.physics_dt,0.)
        return torch.stack([age,valid.float()],-1)
