"""Dreamer-V3 style World Model for model-based negotiation.

Learns a compact latent world model of the negotiation dynamics.
Agents imagine rollouts inside the world model to plan ahead
before committing to real proposals — crucial for high-stakes deals.
"""
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from asne.agents.base import BaseNegotiator
from asne.core.config import AgentConfig
from asne.core.types import AgentRole, Proposal


class RSSM(nn.Module):
    """Recurrent State-Space Model (RSSM) — Dreamer-style.
    Separates deterministic (h) and stochastic (z) latent states.
    """
    def __init__(self, obs_dim: int, action_dim: int, h_dim: int = 256, z_dim: int = 64) -> None:
        super().__init__()
        self.h_dim, self.z_dim = h_dim, z_dim
        # Prior: predict z from (h)
        self.prior_net = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ELU(), nn.Linear(h_dim, z_dim * 2))
        # Posterior: refine z from (h, obs)
        self.post_net = nn.Sequential(nn.Linear(h_dim + obs_dim, h_dim), nn.ELU(), nn.Linear(h_dim, z_dim * 2))
        # Transition (GRU)
        self.gru = nn.GRUCell(z_dim + action_dim, h_dim)
        # Decoder
        self.decoder = nn.Sequential(nn.Linear(h_dim + z_dim, 256), nn.ELU(), nn.Linear(256, obs_dim))
        # Reward predictor
        self.reward_pred = nn.Sequential(nn.Linear(h_dim + z_dim, 128), nn.ELU(), nn.Linear(128, 1))

    def _sample_z(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = params.chunk(2, dim=-1)
        std = F.softplus(log_std) + 1e-4
        z = mu + std * torch.randn_like(std)
        kl = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2))
        return z, kl.sum(-1).mean()

    def forward(self, obs: torch.Tensor, action: torch.Tensor, h: torch.Tensor) -> dict[str, torch.Tensor]:
        post_params = self.post_net(torch.cat([h, obs], dim=-1))
        z_post, kl = self._sample_z(post_params)
        h_next = self.gru(torch.cat([z_post, action], dim=-1), h)
        prior_params = self.prior_net(h_next)
        z_prior, _ = self._sample_z(prior_params)
        recon = self.decoder(torch.cat([h_next, z_post], dim=-1))
        reward = self.reward_pred(torch.cat([h_next, z_post], dim=-1))
        return {"h": h_next, "z": z_post, "z_prior": z_prior, "recon": recon, "reward": reward, "kl": kl}

    def imagine(self, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        h_next = self.gru(torch.cat([z, action], dim=-1), h)
        prior_params = self.prior_net(h_next)
        z_next, _ = self._sample_z(prior_params)
        reward = self.reward_pred(torch.cat([h_next, z_next], dim=-1))
        return {"h": h_next, "z": z_next, "reward": reward}


class WorldModelAgent(BaseNegotiator):
    """Dreamer-V3 inspired agent: learn + imagine + act."""
    def __init__(self, obs_dim: int, action_dim: int, imagine_horizon: int = 15, config: AgentConfig | None = None, role: AgentRole = AgentRole.BUYER) -> None:
        super().__init__(role=role)
        self.config = config or AgentConfig()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.imagine_horizon = imagine_horizon
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rssm = RSSM(obs_dim, action_dim).to(self.device)
        self.actor = nn.Sequential(nn.Linear(self.rssm.h_dim + self.rssm.z_dim, 256), nn.ELU(), nn.Linear(256, action_dim)).to(self.device)
        self.optimizer = torch.optim.AdamW(list(self.rssm.parameters()) + list(self.actor.parameters()), lr=3e-4)
        self._h = torch.zeros(1, self.rssm.h_dim, device=self.device)
        self._z = torch.zeros(1, self.rssm.z_dim, device=self.device)

    def _encode_obs(self, state: dict[str, Any]) -> torch.Tensor:
        features = []
        for key in sorted(state.keys()):
            val = state[key]
            if isinstance(val, (int, float)): features.append(float(val))
            elif isinstance(val, list): features.extend([float(v) for v in val[:5]])
        features = (features[:self.obs_dim] + [0.0] * self.obs_dim)[:self.obs_dim]
        return torch.FloatTensor(features).unsqueeze(0).to(self.device)

    def _imagine_best_action(self) -> int:
        best_action, best_return = 0, float("-inf")
        for action_idx in range(self.action_dim):
            action_onehot = F.one_hot(torch.tensor([action_idx]), self.action_dim).float().to(self.device)
            h, z = self._h.clone(), self._z.clone()
            total_reward = 0.0
            discount = 1.0
            for _ in range(self.imagine_horizon):
                out = self.rssm.imagine(h, z, action_onehot)
                h, z = out["h"], out["z"]
                total_reward += discount * out["reward"].item()
                discount *= 0.995
                action_onehot = F.one_hot(torch.randint(0, self.action_dim, (1,)), self.action_dim).float().to(self.device)
            if total_reward > best_return:
                best_return, best_action = total_reward, action_idx
        return best_action

    def propose(self, state: dict[str, Any]) -> Proposal:
        self._round += 1
        obs = self._encode_obs(state)
        dummy_action = torch.zeros(1, self.action_dim, device=self.device)
        with torch.no_grad():
            out = self.rssm.forward(obs, dummy_action, self._h)
            self._h, self._z = out["h"].detach(), out["z"].detach()
            best_action = self._imagine_best_action()
        base_value = state.get("deal_value", 100000)
        adjustment = (best_action - self.action_dim // 2) / self.action_dim
        proposed_value = base_value * (1 + adjustment * 0.2)
        return Proposal(agent_id=self.agent_id, round_num=self._round, terms={"value": proposed_value, "dreamer_action": best_action, "imagined_horizon": self.imagine_horizon}, value=proposed_value, metadata={"model": "DreamerV3"})

    def respond(self, proposal: Proposal, state: dict[str, Any]) -> bool:
        obs = self._encode_obs(state)
        dummy_action = torch.zeros(1, self.action_dim, device=self.device)
        with torch.no_grad():
            out = self.rssm.forward(obs, dummy_action, self._h)
            predicted_reward = out["reward"].item()
        return proposal.value >= predicted_reward * 0.8

    def learn(self, reward: float, next_state: dict[str, Any]) -> None:
        pass  # Full world model training handled by DreamerTrainer
