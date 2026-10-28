"""Recurrent primitive actor restricted to finger proprioception and delivered F/T.

The source observation layout is the versioned 250-value Flick/Cycle layout.
No cube pose/angle/velocity, fingertip-relative-to-cube coordinates, or completion
latch enters the student. Move ID and previous commanded actions are allowed.
"""
import torch
from torch import nn


def student_features(observation):
    if observation.shape[-1] != 250:
        raise ValueError('Expected version-1 250-column teacher observation layout')
    return torch.cat([observation[..., :44], observation[..., 72:102],
                      observation[..., 102:146], observation[..., 174:204],
                      observation[..., 220:241], observation[..., 246:250]], dim=-1)


class BlindPrimitiveActor(nn.Module):
    feature_dim = 173
    def __init__(self, hidden=128):
        super().__init__()
        self.hidden = hidden
        self.encoder = nn.Sequential(nn.Linear(self.feature_dim, hidden), nn.ELU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ELU(), nn.Linear(64, 4), nn.Tanh())

    def forward(self, features, hidden=None):
        encoded = self.encoder(features)
        sequence, hidden = self.gru(encoded, hidden)
        return self.head(sequence), hidden


class BlindInference:
    observation_scope = "tactile_proprioceptive"
    def __init__(self, checkpoint, n, device):
        self.execution_protocol=checkpoint.get('execution_protocol',{})
        self.model = BlindPrimitiveActor(checkpoint['hidden']).to(device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        self.hidden = torch.zeros(1, n, self.model.hidden, device=device)

    @torch.inference_mode()
    def reset(self, ids=None):
        if ids is None:
            self.hidden.zero_()
        else:
            self.hidden[:, ids] = 0

    @torch.inference_mode()
    def __call__(self, obs):
        features = student_features(obs['policy'])
        action, self.hidden = self.model(features.unsqueeze(1), self.hidden)
        return action[:, 0]


class TrajectoryInference:
    """A learned demonstration medoid, selected from initial finger posture only."""
    observation_scope = 'initial_proprioception_and_clock'
    def __init__(self,checkpoint,n,device):
        self.anchors=checkpoint['anchors'].to(device)
        self.trajectories=checkpoint['trajectories'].to(device)
        self.lengths=checkpoint['lengths'].to(device)
        self.settle_steps=int(checkpoint.get('settle_steps',0))
        self.step_index=torch.zeros(n,dtype=torch.long,device=device)
        self.selected=torch.zeros(n,dtype=torch.long,device=device)

    @torch.inference_mode()
    def reset(self,ids=None):
        if ids is None:
            self.step_index.zero_()
        else:
            self.step_index[ids]=0

    @torch.inference_mode()
    def __call__(self,obs):
        initial=self.step_index==0
        if initial.any():
            q=torch.cat([obs['policy'][initial,:22],obs['policy'][initial,102:124]],-1)
            self.selected[initial]=torch.cdist(q,self.anchors).argmin(-1)
        t=torch.minimum((self.step_index-self.settle_steps).clamp(min=0),self.lengths[self.selected]-1)
        action=self.trajectories[self.selected,t].clone()
        settling=self.step_index<self.settle_steps
        action[settling]=self.trajectories[self.selected[settling],self.lengths[self.selected[settling]]-1]
        self.step_index+=1
        return action
