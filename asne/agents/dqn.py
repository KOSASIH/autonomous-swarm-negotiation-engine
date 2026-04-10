"""Deep Q-Network negotiator agent."""
from __future__ import annotations
import random
from collections import deque
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from asne.agents.base import BaseNegotiator
from asne.core.config import AgentConfig
from asne.core.types import AgentRole, Proposal

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class DQNNegotiator(BaseNegotiator):
    def __init__(self, state_dim: int, action_dim: int, config: AgentConfig | None = None, role: AgentRole = AgentRole.BUYER) -> None:
        super().__init__(role=role)
        self.config = config or AgentConfig()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(state_dim, action_dim, self.config.hidden_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, self.config.hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.learning_rate)
        self.memory: deque = deque(maxlen=self.config.memory_size)
        self.steps = 0
        self._last_state: dict[str, Any] | None = None
        self._last_action: int | None = None

    def _get_epsilon(self) -> float:
        return self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * max(0, 1 - self.steps / self.config.epsilon_decay)

    def _state_to_tensor(self, state: dict[str, Any]) -> torch.Tensor:
        features = []
        for key in sorted(state.keys()):
            val = state[key]
            if isinstance(val, (int, float)):
                features.append(float(val))
            elif isinstance(val, (list, np.ndarray)):
                features.extend([float(v) for v in val])
        features = features[:self.state_dim]
        features.extend([0.0] * (self.state_dim - len(features)))
        return torch.FloatTensor(features).unsqueeze(0).to(self.device)

    def select_action(self, state: dict[str, Any]) -> int:
        if random.random() < self._get_epsilon():
            return random.randrange(self.action_dim)
        with torch.no_grad():
            return int(self.q_network(self._state_to_tensor(state)).argmax(dim=1).item())

    def propose(self, state: dict[str, Any]) -> Proposal:
        self._round += 1
        action = self.select_action(state)
        self._last_state = state
        self._last_action = action
        base_value = state.get("deal_value", 100000)
        adjustment = (action - self.action_dim // 2) / self.action_dim
        proposed_value = base_value * (1 + adjustment * 0.2)
        return Proposal(agent_id=self.agent_id, round_num=self._round, terms={"value": proposed_value, "action_idx": action}, value=proposed_value)

    def respond(self, proposal: Proposal, state: dict[str, Any]) -> bool:
        with torch.no_grad():
            accept_threshold = self.q_network(self._state_to_tensor(state)).mean().item()
        return proposal.value >= accept_threshold

    def learn(self, reward: float, next_state: dict[str, Any]) -> None:
        if self._last_state is None or self._last_action is None:
            return
        self.memory.append((self._last_state, self._last_action, reward, next_state, False))
        self.steps += 1
        if len(self.memory) < self.config.batch_size:
            return
        batch = random.sample(list(self.memory), self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        state_batch = torch.cat([self._state_to_tensor(s) for s in states])
        action_batch = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_state_batch = torch.cat([self._state_to_tensor(s) for s in next_states])
        done_batch = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        current_q = self.q_network(state_batch).gather(1, action_batch)
        with torch.no_grad():
            max_next_q = self.target_network(next_state_batch).max(1, keepdim=True)[0]
            target_q = reward_batch + self.config.gamma * max_next_q * (1 - done_batch)
        loss = nn.functional.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        if self.steps % 100 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
