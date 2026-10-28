"""Two skill-family policies (flick, wrist) trained together by one PPO in skill chains.

MixActorCritic holds one rsl_rl ActorCritic per family; each env's skill one-hot in the teacher observation selects the
family whose actor, std and critic act for it. The families start from their separately trained checkpoints (extra
observation columns appended later, e.g. the next skill, enter with zero weights); observation normalizers stay frozen.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules import ActorCritic

from . import skills as S

# teacher observation layout (SolveEnv._get_observations): 2 x 102 proprio + 20 cube, then the skill one-hot
SKILL_OFFSET = 2 * 102 + 20
FLICK_IDS = [S.SKILL_INDEX[s] for s in S.FLICK]


class MixActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, obs, obs_groups, num_actions, **kw):
        super().__init__()
        self.fam = nn.ModuleList([ActorCritic(obs, obs_groups, num_actions, **kw) for _ in range(2)])  # 0 flick, 1 wrist
        self.obs_groups = obs_groups
        self.distribution = None
        self.register_buffer("k_flick", torch.zeros(len(S.SKILLS), dtype=torch.bool))
        self.k_flick[FLICK_IDS] = True

    def _flick(self, x):
        return self.k_flick[x[:, SKILL_OFFSET : SKILL_OFFSET + len(S.SKILLS)].argmax(-1)].unsqueeze(-1)

    def _actor(self, obs):
        x = self.fam[0].get_actor_obs(obs)
        m = self._flick(x)
        mean = [f.actor(f.actor_obs_normalizer(x)) for f in self.fam]
        std = [torch.exp(f.log_std).expand_as(mean[0]) for f in self.fam]
        return torch.where(m, mean[0], mean[1]), torch.where(m, std[0], std[1])

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def act(self, obs, **kwargs):
        mean, std = self._actor(obs)
        self.distribution = Normal(mean, std)
        return self.distribution.sample()

    def act_inference(self, obs):
        return self._actor(obs)[0]

    def evaluate(self, obs, **kwargs):
        x = self.fam[0].get_critic_obs(obs)
        v = [f.critic(f.critic_obs_normalizer(x)) for f in self.fam]
        return torch.where(self._flick(x), v[0], v[1])

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        pass  # frozen: the family checkpoints' statistics

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True

    @torch.no_grad()
    def load_families(self, flick_path: str, wrist_path: str):
        """initialize both families from single-family rsl_rl checkpoints; observation columns beyond a checkpoint's
        input size get zero weights (normalizer mean 0, std 1)."""
        for f, path in zip(self.fam, (flick_path, wrist_path)):
            sd = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
            own = f.state_dict()
            for k, v in sd.items():
                t = own[k]
                if v.shape == t.shape:
                    t.copy_(v)
                elif k.endswith(".0.weight"):  # first layer: extra input columns at the end
                    t.zero_()
                    t[:, : v.shape[1]] = v
                elif k.endswith(("_mean", "_var", "_std")):
                    t.copy_(torch.zeros_like(t) if k.endswith("_mean") else torch.ones_like(t))
                    t[..., : v.shape[-1]] = v
                else:
                    raise ValueError(f"{path}: {k} {tuple(v.shape)} vs {tuple(t.shape)}")
            print(f"[mix] family {'flick' if f is self.fam[0] else 'wrist'} <- {path}", flush=True)
