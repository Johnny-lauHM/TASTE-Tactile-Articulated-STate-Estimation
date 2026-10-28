"""Blind cube-state estimator: recurrent net over what the hands sense (finger joint angles / velocities, delivered pad
F/T) and what the robot commands (skill, phase, rolls, next skill) -> the cube state the sighted controllers act on.

The estimate replaces the simulator's cube state in every controller (env.cfg.cube_est: the teacher's cube
observation, the regrip servo, the legacy flick policy); the simulator's state then only scores. Outputs per step:
    state (17)  position off the grip center x20, rotation off the skill goal (rot6), layer error to the target [rad],
                core linear velocity, 0.1 x angular velocity, 0.1 x layer velocity (SolveEnv.cube_state_true units)
    done  (1)   logit: the skill's completion criteria hold (the executor's trigger)
Optionally (img_feat > 0) also the delivered 60x60 TacMap images of the 10 pads (obs "tac_img", pooled to 30x30): one
small CNN shared by the pads -> img_feat features per pad, appended to the vector input.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .student import RunningNorm

# label scale per state dimension (the loss and the outputs are in these units): 2 mm, ~1 deg, ~1 deg, 5 cm/s, 1 rad/s
STATE_SCALE = [0.04] * 3 + [0.02] * 6 + [0.02] + [0.05] * 3 + [0.1] * 3 + [0.1]
ROT6_ID = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]


def rot6_to_matrix(r6: torch.Tensor) -> torch.Tensor:
    """(N, 6) first two columns of a rotation matrix, row-major as matrix[..., :2].reshape(-1, 6) -> (N, 3, 3)."""
    m = r6.view(-1, 3, 2)
    b1 = torch.nn.functional.normalize(m[..., 0], dim=-1)
    a2 = m[..., 1]
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    return torch.stack([b1, b2, torch.cross(b1, b2, dim=-1)], -1)


class PadImageEncoder(nn.Module):
    """(..., 30, 30) uint8 tactile images -> (..., feat); weights shared by all pads."""

    def __init__(self, feat: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.ELU(),  # 15x15
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ELU(),  # 8x8
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ELU(),  # 4x4
            nn.Flatten(), nn.Linear(512, feat),
        )

    def forward(self, img: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
        x = img.flatten(0, -3)
        outs = []
        for i in range(0, x.shape[0], chunk):  # recomputed in backward: activations of all T x N x 10 images do not fit
            xi = x[i : i + chunk].unsqueeze(1).float() / 255.0
            outs.append(checkpoint(self.net, xi, use_reentrant=False) if torch.is_grad_enabled() else self.net(xi))
        return torch.cat(outs, 0).view(*img.shape[:-2], -1)


class CubeEstimator(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 384, enc: int = 512, img_feat: int = 0, img_pads: int = 10):
        super().__init__()
        self.obs_dim, self.hidden, self.img_feat = obs_dim, hidden, img_feat
        self.norm = RunningNorm(obs_dim)
        self.img_enc = PadImageEncoder(img_feat) if img_feat else None
        self.enc = nn.Sequential(nn.Linear(obs_dim + img_feat * img_pads, enc), nn.ELU(), nn.Linear(enc, hidden), nn.ELU())
        self.gru = nn.GRU(hidden, hidden, batch_first=False)
        self.head = nn.Sequential(nn.Linear(hidden + hidden, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 18))
        self.register_buffer("scale", torch.tensor(STATE_SCALE))
        self.register_buffer("offset", torch.tensor([0.0] * 3 + ROT6_ID + [0.0] * 8))

    def normalize(self, state: torch.Tensor) -> torch.Tensor:
        return (state - self.offset) / self.scale

    def forward(self, obs: torch.Tensor, h: torch.Tensor, resets: torch.Tensor | None = None, img: torch.Tensor | None = None):
        """obs (T, N, D), h (1, N, H), resets (T, N): hidden state zeroed BEFORE that step, img (T, N, pads, 30, 30) uint8.
        -> state in normalized units (T, N, 17), done logit (T, N), h."""
        x = self.norm(obs)
        if self.img_enc is not None:
            x = torch.cat([x, self.img_enc(img).flatten(-2)], -1)
        x = self.enc(x)
        outs = []
        for t in range(obs.shape[0]):
            if resets is not None:
                h = h * (~resets[t]).float().view(1, -1, 1)
            o, h = self.gru(x[t : t + 1], h)
            outs.append(o[0])
        z = self.head(torch.cat([torch.stack(outs, 0), x], -1))
        return z[..., :17], z[..., 17], h

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.scale + self.offset

    def init_hidden(self, n, device):
        return torch.zeros(1, n, self.hidden, device=device)


class Estimator:
    """stateful wrapper for execution: step(env observations) -> cube state (N, 17) and done probability, hidden state per env."""

    def __init__(self, path: str, num_envs: int, device):
        blob = torch.load(path, map_location=device, weights_only=False)
        self.net = CubeEstimator(**blob["kwargs"]).to(device).eval()
        self.net.load_state_dict(blob["state_dict"])
        self.h = self.net.init_hidden(num_envs, device)
        self.h_prev = self.h.clone()
        self.done = torch.zeros(num_envs, device=device)
        self.obs_key = blob.get("obs_key", "est")  # env observation group the estimator reads
        zc = blob.get("zero_cols", [])  # input ablation (train_estimator --zero_ft): these columns read 0
        self.zero_cols = torch.tensor(zc, dtype=torch.long, device=device) if zc else None

    @torch.no_grad()
    def step(self, obs: dict, ids=None):
        """one control step; with ids: redo this step for these envs only (their observation changed after a skill start)."""
        img = obs["tac_img"] if self.net.img_enc is not None else None
        obs = obs[self.obs_key]
        if self.zero_cols is not None:
            obs = obs.clone()
            obs[:, self.zero_cols] = 0.0
        if ids is None:
            self.h_prev = self.h
            z, d, self.h = self.net(obs.unsqueeze(0), self.h_prev, img=None if img is None else img.unsqueeze(0))
            self.state, self.done = self.net.denormalize(z[0]), torch.sigmoid(d[0])
        else:
            z, d, h = self.net(obs[ids].unsqueeze(0), self.h_prev[:, ids].contiguous(), img=None if img is None else img[ids].unsqueeze(0))
            self.h[:, ids] = h
            self.state[ids], self.done[ids] = self.net.denormalize(z[0]), torch.sigmoid(d[0])
        return self.state

    def reset(self, env_ids):
        self.h[:, env_ids] = 0.0
