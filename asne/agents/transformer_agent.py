"""Multi-Agent Transformer (MAT) negotiation agent.

Transformer-based policy that attends over all agents' observations
using multi-head self-attention for cooperative/competitive reasoning.
"""
from __future__ import annotations
import math
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from asne.agents.base import BaseNegotiator
from asne.core.config import AgentConfig
from asne.core.types import AgentRole, Proposal


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B = q.size(0)
        Q = self.q_proj(q).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(k).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(v).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ff_dim, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.norm1(x + self.drop(self.attn(x, x, x, mask)))
        return self.norm2(x + self.drop(self.ff(x)))


class MATPolicy(nn.Module):
    """Multi-Agent Transformer policy network."""
    def __init__(self, obs_dim: int, action_dim: int, n_agents: int, d_model: int = 128, n_heads: int = 4, n_layers: int = 3) -> None:
        super().__init__()
        self.obs_embed = nn.Linear(obs_dim, d_model)
        self.agent_pos_embed = nn.Embedding(n_agents, d_model)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_model * 4) for _ in range(n_layers)])
        self.actor_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, action_dim))
        self.critic_head = nn.Sequential(nn.Linear(d_model * n_agents, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.n_agents = n_agents
        self.d_model = d_model

    def forward(self, obs: torch.Tensor, agent_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """obs: (B, N, obs_dim), agent_ids: (B, N)"""
        x = self.obs_embed(obs) + self.agent_pos_embed(agent_ids)
        for block in self.transformer_blocks:
            x = block(x)
        action_logits = self.actor_head(x)  # (B, N, action_dim)
        value = self.critic_head(x.view(x.size(0), -1))  # (B, 1)
        return action_logits, value


class MATNegotiator(BaseNegotiator):
    """Multi-Agent Transformer negotiator with centralized training, decentralized execution."""
    def __init__(self, obs_dim: int, action_dim: int, n_agents: int = 5, agent_idx: int = 0, config: AgentConfig | None = None, role: AgentRole = AgentRole.BUYER) -> None:
        super().__init__(role=role)
        self.config = config or AgentConfig()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.agent_idx = agent_idx
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = MATPolicy(obs_dim, action_dim, n_agents).to(self.device)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.config.learning_rate, weight_decay=1e-4)
        self._obs_buffer: list[dict[str, Any]] = []

    def _encode_obs(self, state: dict[str, Any]) -> torch.Tensor:
        features = []
        for key in sorted(state.keys()):
            val = state[key]
            if isinstance(val, (int, float)): features.append(float(val))
            elif isinstance(val, list): features.extend([float(v) for v in val[:8]])
        features = (features[:self.obs_dim] + [0.0] * self.obs_dim)[:self.obs_dim]
        return torch.FloatTensor(features).to(self.device)

    def propose(self, state: dict[str, Any]) -> Proposal:
        self._round += 1
        obs = self._encode_obs(state).unsqueeze(0).unsqueeze(0)  # (1, 1, obs_dim)
        # Pad to n_agents
        obs_padded = obs.expand(1, self.n_agents, -1)
        agent_ids = torch.arange(self.n_agents).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.policy(obs_padded, agent_ids)
            action = torch.multinomial(F.softmax(logits[0, self.agent_idx], dim=-1), 1).item()
        base_value = state.get("deal_value", 100000)
        adjustment = (action - self.action_dim // 2) / self.action_dim
        proposed_value = base_value * (1 + adjustment * 0.15)
        return Proposal(agent_id=self.agent_id, round_num=self._round, terms={"value": proposed_value, "transformer_action": int(action)}, value=proposed_value, metadata={"model": "MAT", "n_heads": 4})

    def respond(self, proposal: Proposal, state: dict[str, Any]) -> bool:
        obs = self._encode_obs(state).unsqueeze(0).unsqueeze(0).expand(1, self.n_agents, -1)
        agent_ids = torch.arange(self.n_agents).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, value = self.policy(obs, agent_ids)
        return proposal.value >= value.item() * 0.9

    def learn(self, reward: float, next_state: dict[str, Any]) -> None:
        pass  # PPO update handled by centralized trainer
