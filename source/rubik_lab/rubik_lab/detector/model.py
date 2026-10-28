from __future__ import annotations

import torch
from torch import nn
from rubik_lab.moves import ALL_MOVES


class PadEncoder(nn.Module):
    """shared CNN for one 60x60 pad image -> embedding."""

    def __init__(self, dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.ELU(),   # 30
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ELU(),  # 15
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ELU(),  # 8
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, dim),
        )  # fmt: skip

    def forward(self, x):  # (B, 1, 60, 60)
        return self.net(x)


class CompletionDetector(nn.Module):
    """(B, T, ...) history -> logits over {in progress, done, failed} at the last step."""

    def __init__(self, ft_dim=60, prop_dim=88, n_moves=len(ALL_MOVES), images=False, hidden=128, n_cls=3):
        super().__init__()
        self.register_buffer("ft_mean", torch.zeros(ft_dim), persistent=False)
        self.register_buffer("ft_scale", torch.ones(ft_dim), persistent=False)
        self.register_buffer("prop_mean", torch.zeros(prop_dim), persistent=False)
        self.register_buffer("prop_scale", torch.ones(prop_dim), persistent=False)
        self.images = images
        self.pad_enc = PadEncoder(32) if images else None
        self.pad_emb = nn.Parameter(torch.zeros(10, 32)) if images else None
        in_dim = 64 + 64 + n_moves + (32 if images else 0)
        self.ft_mlp = nn.Sequential(nn.Linear(ft_dim, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU())
        self.prop_mlp = nn.Sequential(nn.Linear(prop_dim, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU())
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ELU(), nn.Linear(64, n_cls))

    def set_normalization(self, statistics):
        if statistics is None:
            return
        for name in ('ft_mean','ft_scale','prop_mean','prop_scale'):
            value=torch.as_tensor(statistics[name],device=getattr(self,name).device,dtype=torch.float32)
            if value.shape != getattr(self,name).shape or not torch.isfinite(value).all():
                raise ValueError('Invalid detector normalization: '+name)
            if name.endswith('scale') and not (value>0).all():
                raise ValueError('Normalization scales must be positive')
            getattr(self,name).copy_(value)

    def encode(self, x: dict) -> torch.Tensor:
        B, T = x["ft"].shape[:2]
        move=x["move"] if x["move"].ndim==3 else x["move"].unsqueeze(1).expand(B,T,-1)
        feats = [self.ft_mlp((x["ft"]-self.ft_mean)/self.ft_scale), self.prop_mlp((x["prop"]-self.prop_mean)/self.prop_scale), move]
        if self.images:
            img = x["image"].reshape(B * T * 10, 1, 60, 60)
            e = self.pad_enc(img).view(B, T, 10, -1) + self.pad_emb
            feats.append(e.max(2).values)  # pool over the 10 pads
        return torch.cat(feats, -1)

    def forward_sequence(self, x: dict, hidden=None):
        states, hidden = self.gru(self.encode(x), hidden)
        return self.head(states), hidden

    def forward(self, x: dict) -> torch.Tensor:
        logits, _ = self.forward_sequence(x)
        return logits[:, -1]


class AngleCompletionDetector(CompletionDetector):
    """A supervised tactile angle estimate alongside instantaneous readiness."""
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.angle_head=nn.Sequential(nn.Linear(self.gru.hidden_size,64),nn.ELU(),nn.Linear(64,1),nn.Tanh())

    def predict_sequence(self,x,hidden=None):
        states,hidden=self.gru(self.encode(x),hidden)
        return self.head(states),self.angle_head(states).squeeze(-1)*torch.pi,hidden
