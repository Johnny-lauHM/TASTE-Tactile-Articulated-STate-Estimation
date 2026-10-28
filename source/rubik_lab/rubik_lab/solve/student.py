"""Deployable student: recurrent policy over proprioception + delivered tactile F/T (+ commanded skill and phase).

Outputs per control step:
    action  (56)  residual on the skill's nominal targets (imitates the privileged teacher, DAgger)
    done    (1)   logit: the current skill is complete and the next one may start (completion detector)
    err     (1)   estimate of the remaining task error [rad] (layer / cube angle to go): tactile odometry, auxiliary

No cube pose, layer angle or contact ground truth enters the network.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RunningNorm(nn.Module):
    def __init__(self, dim, eps=1e-4, clip=8.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(eps))
        self.clip = clip

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, x.shape[-1])
        b_mean, b_var, b_n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        tot = self.count + b_n
        delta = b_mean - self.mean
        self.mean += delta * b_n / tot
        self.var = (self.var * self.count + b_var * b_n + delta.square() * self.count * b_n / tot) / tot
        self.count = tot

    def forward(self, x):
        return ((x - self.mean) / (self.var.sqrt() + 1e-6)).clamp(-self.clip, self.clip)


class StudentNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int = 56, hidden: int = 384, enc: int = 512, err_dim: int = 1):
        super().__init__()
        self.obs_dim, self.act_dim, self.hidden, self.err_dim = obs_dim, act_dim, hidden, err_dim
        self.norm = RunningNorm(obs_dim)
        self.enc = nn.Sequential(nn.Linear(obs_dim, enc), nn.ELU(), nn.Linear(enc, hidden), nn.ELU())
        self.gru = nn.GRU(hidden, hidden, batch_first=False)
        self.head = nn.Sequential(nn.Linear(hidden + hidden, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU())
        self.act = nn.Linear(256, act_dim)
        self.done = nn.Linear(256, 1)
        self.err = nn.Linear(256, err_dim)  # auxiliary estimate (task error, or the cube offset in the grip)

    def forward(self, obs: torch.Tensor, h: torch.Tensor, resets: torch.Tensor | None = None):
        """obs (T, N, D), h (1, N, H), resets (T, N) bool: hidden state zeroed BEFORE that step."""
        x = self.enc(self.norm(obs))
        outs = []
        for t in range(obs.shape[0]):
            if resets is not None:
                h = h * (~resets[t]).float().view(1, -1, 1)
            o, h = self.gru(x[t : t + 1], h)
            outs.append(o[0])
        o = torch.stack(outs, 0)
        z = self.head(torch.cat([o, x], -1))
        e = self.err(z)
        return self.act(z).clamp(-1.0, 1.0), self.done(z).squeeze(-1), (e.squeeze(-1) if self.err_dim == 1 else e), h

    def init_hidden(self, n, device):
        return torch.zeros(1, n, self.hidden, device=device)


class StudentPolicy:
    """stateful wrapper used by the executor: call(obs) -> actions, keeps the hidden state per env."""

    def __init__(self, net: StudentNet, num_envs: int, device):
        self.net = net.to(device).eval()
        self.h = net.init_hidden(num_envs, device)
        self.last_done = torch.zeros(num_envs, device=device)
        self.last_err = torch.zeros(num_envs, device=device)

    @classmethod
    def from_checkpoint(cls, blob: dict, num_envs: int, device):
        net = StudentNet(**blob["student_kwargs"])
        net.load_state_dict(blob["student_state_dict"])
        return cls(net, num_envs, device)

    @torch.no_grad()
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        a, d, e, self.h = self.net(obs.unsqueeze(0), self.h)
        self.last_done, self.last_err = torch.sigmoid(d[0]), e[0]
        return a[0]

    def reset(self, env_ids):
        self.h[:, env_ids] = 0.0
