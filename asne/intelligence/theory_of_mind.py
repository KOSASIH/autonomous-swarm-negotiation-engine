"""Theory of Mind (ToM) module for negotiation.

Models the mental states, beliefs, intentions, and preferences
of counterparties — enabling predictive reasoning about their
future actions. Critical for adversarial and cooperative dynamics.

Implements nested belief reasoning:
  Level 0: What does the counterparty want?
  Level 1: What does the counterparty think I want?
  Level 2: What does the counterparty think I think they want?
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BeliefState:
    """Mental model of a negotiation counterparty."""
    agent_id: str
    estimated_budget: float
    estimated_reservation_price: float
    risk_tolerance: float  # 0 = risk-averse, 1 = risk-seeking
    urgency: float  # 0 = no urgency, 1 = must close now
    preferred_issues: list[str] = field(default_factory=list)
    deception_probability: float = 0.0
    cooperation_level: float = 0.5
    update_count: int = 0


class ToMNetwork(nn.Module):
    """Neural ToM: predict counterparty's next action from their observable history."""
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128) -> None:
        super().__init__()
        # Belief encoder: compress counterparty observations into belief vector
        self.belief_encoder = nn.LSTM(obs_dim, hidden, batch_first=True, num_layers=2)
        # Mental state predictor
        self.mental_state_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 16))
        # Action predictor: what will they do next?
        self.action_predictor = nn.Sequential(nn.Linear(hidden + 16, hidden), nn.ReLU(), nn.Linear(hidden, action_dim))
        # Deception detector
        self.deception_head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, obs_seq: torch.Tensor) -> dict[str, torch.Tensor]:
        h, _ = self.belief_encoder(obs_seq)
        last_h = h[:, -1, :]
        mental_state = self.mental_state_head(last_h)
        action_logits = self.action_predictor(torch.cat([last_h, mental_state], dim=-1))
        deception_prob = self.deception_head(last_h)
        return {"mental_state": mental_state, "action_logits": action_logits, "deception_prob": deception_prob}


class TheoryOfMindModule:
    """Full Theory of Mind system for multi-party negotiations."""
    def __init__(self, obs_dim: int = 16, action_dim: int = 11) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tom_net = ToMNetwork(obs_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.tom_net.parameters(), lr=1e-3)
        self._belief_states: dict[str, BeliefState] = {}
        self._observation_history: dict[str, list[list[float]]] = {}

    def update_belief(self, agent_id: str, observation: dict[str, Any]) -> BeliefState:
        """Update belief model based on new counterparty observation."""
        if agent_id not in self._belief_states:
            self._belief_states[agent_id] = BeliefState(
                agent_id=agent_id,
                estimated_budget=float(observation.get("deal_value", 100000)),
                estimated_reservation_price=float(observation.get("deal_value", 100000)) * 0.8,
                risk_tolerance=0.5, urgency=float(observation.get("urgency", 0.5))
            )
        belief = self._belief_states[agent_id]
        # Bayesian update
        if "deal_value" in observation:
            alpha = 0.1  # learning rate for belief update
            belief.estimated_budget = (1 - alpha) * belief.estimated_budget + alpha * float(observation["deal_value"])
        belief.update_count += 1
        obs_vec = [float(observation.get(k, 0)) for k in sorted(observation.keys()) if isinstance(observation.get(k), (int, float))]
        obs_vec = (obs_vec[:self.obs_dim] + [0.0] * self.obs_dim)[:self.obs_dim]
        if agent_id not in self._observation_history:
            self._observation_history[agent_id] = []
        self._observation_history[agent_id].append(obs_vec)
        return belief

    def predict_action(self, agent_id: str) -> dict[str, Any]:
        """Predict counterparty's next action."""
        history = self._observation_history.get(agent_id, [[0.0] * self.obs_dim])
        obs_seq = torch.FloatTensor([history[-min(10, len(history)):] + [[0.0] * self.obs_dim] * max(0, 10 - len(history))]).to(self.device)
        with torch.no_grad():
            out = self.tom_net(obs_seq)
        probs = F.softmax(out["action_logits"][0], dim=-1).cpu().numpy()
        predicted_action = int(np.argmax(probs))
        deception_prob = float(out["deception_prob"].item())
        return {
            "predicted_action": predicted_action,
            "action_probabilities": probs.tolist(),
            "deception_probability": round(deception_prob, 4),
            "is_deceptive": deception_prob > 0.6,
            "belief": self._belief_states.get(agent_id),
        }

    def estimate_zopa(self, my_reservation: float, agent_id: str) -> dict[str, Any]:
        """Estimate Zone of Possible Agreement (ZOPA)."""
        belief = self._belief_states.get(agent_id)
        if not belief:
            return {"zopa_exists": False, "zopa_width": 0.0}
        zopa_low = max(my_reservation, belief.estimated_reservation_price)
        zopa_high = min(belief.estimated_budget, belief.estimated_budget * 1.2)
        zopa_exists = zopa_low < zopa_high
        return {
            "zopa_exists": zopa_exists,
            "zopa_low": round(zopa_low, 2),
            "zopa_high": round(zopa_high, 2),
            "zopa_width": round(max(0, zopa_high - zopa_low), 2),
            "recommended_bid": round((zopa_low + zopa_high) / 2, 2) if zopa_exists else my_reservation,
        }
