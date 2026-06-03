"""MAML (Model-Agnostic Meta-Learning) negotiator.

Learns a meta-policy that can adapt to new deal types / counterparties
in just 1-5 gradient steps. Enables rapid domain adaptation without
full retraining — critical for cross-industry B2B deployment.
"""
from __future__ import annotations
from copy import deepcopy
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from asne.agents.base import BaseNegotiator
from asne.core.config import AgentConfig
from asne.core.types import AgentRole, Proposal


class MetaPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, action_dim)
        )
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net(x), self.value_net(x)


class MAMLNegotiator(BaseNegotiator):
    """Meta-learner that adapts to new negotiation tasks in a few gradient steps."""
    def __init__(self, state_dim: int, action_dim: int, inner_lr: float = 0.01, inner_steps: int = 5, config: AgentConfig | None = None, role: AgentRole = AgentRole.BUYER) -> None:
        super().__init__(role=role)
        self.config = config or AgentConfig()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.meta_policy = MetaPolicy(state_dim, action_dim).to(self.device)
        self.meta_optimizer = torch.optim.Adam(self.meta_policy.parameters(), lr=self.config.learning_rate)
        self._task_policy: nn.Module | None = None
        self._support_set: list[tuple] = []

    def _state_to_tensor(self, state: dict[str, Any]) -> torch.Tensor:
        features = []
        for key in sorted(state.keys()):
            val = state[key]
            if isinstance(val, (int, float)): features.append(float(val))
            elif isinstance(val, list): features.extend([float(v) for v in val[:5]])
        features = (features[:self.state_dim] + [0.0] * self.state_dim)[:self.state_dim]
        return torch.FloatTensor(features).unsqueeze(0).to(self.device)

    def adapt(self, support_set: list[tuple]) -> None:
        """Inner loop: adapt meta-policy to a new task."""
        task_policy = deepcopy(self.meta_policy)
        opt = torch.optim.SGD(task_policy.parameters(), lr=self.inner_lr)
        for step in range(self.inner_steps):
            total_loss = torch.tensor(0.0, device=self.device)
            for state, action, reward in support_set:
                x = self._state_to_tensor(state)
                logits, value = task_policy(x)
                action_tensor = torch.tensor([action], device=self.device)
                reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device)
                actor_loss = F.cross_entropy(logits, action_tensor)
                critic_loss = F.mse_loss(value.squeeze(), reward_tensor)
                total_loss = total_loss + actor_loss + 0.5 * critic_loss
            opt.zero_grad()
            total_loss.backward()
            opt.step()
        self._task_policy = task_policy

    def propose(self, state: dict[str, Any]) -> Proposal:
        self._round += 1
        policy = self._task_policy if self._task_policy else self.meta_policy
        x = self._state_to_tensor(state)
        with torch.no_grad():
            logits, _ = policy(x)
            action = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
        base_value = state.get("deal_value", 100000)
        adjustment = (int(action) - self.action_dim // 2) / self.action_dim
        proposed_value = base_value * (1 + adjustment * 0.18)
        return Proposal(agent_id=self.agent_id, round_num=self._round, terms={"value": proposed_value, "maml_action": int(action), "adapted": self._task_policy is not None}, value=proposed_value, metadata={"model": "MAML", "inner_steps": self.inner_steps})

    def respond(self, proposal: Proposal, state: dict[str, Any]) -> bool:
        policy = self._task_policy if self._task_policy else self.meta_policy
        x = self._state_to_tensor(state)
        with torch.no_grad():
            _, value = policy(x)
        return proposal.value >= value.item() * 0.85

    def learn(self, reward: float, next_state: dict[str, Any]) -> None:
        pass  # Meta-update via outer loop
