"""Counterfactual Reasoning Engine.

Answers 'what-if' questions about negotiation outcomes:
  - 'What if I had proposed $50k instead of $60k?'
  - 'What if the market conditions were different?'
  - 'What if we had more rounds available?'

Uses twin-network counterfactual inference:
1. Abduction: infer latent noise from observed outcome
2. Action: apply hypothetical intervention
3. Prediction: compute counterfactual outcome
"""
from __future__ import annotations
from typing import Any
import numpy as np
import torch
import torch.nn as nn


class CounterfactualNetwork(nn.Module):
    """Neural network for counterfactual prediction."""
    def __init__(self, state_dim: int, action_dim: int, outcome_dim: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(state_dim + action_dim, 256), nn.ELU(), nn.Linear(256, 128))
        self.noise_encoder = nn.Sequential(nn.Linear(128 + outcome_dim, 64), nn.ELU())
        self.cf_decoder = nn.Sequential(nn.Linear(64 + state_dim + action_dim, 256), nn.ELU(), nn.Linear(256, outcome_dim))

    def forward(self, state: torch.Tensor, action: torch.Tensor, observed_outcome: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=-1)
        h = self.encoder(sa)
        noise = self.noise_encoder(torch.cat([h, observed_outcome], dim=-1))
        return noise

    def predict_counterfactual(self, state: torch.Tensor, cf_action: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sa_cf = torch.cat([state, cf_action], dim=-1)
        return self.cf_decoder(torch.cat([noise, sa_cf], dim=-1))


class CounterfactualEngine:
    """Full counterfactual reasoning for negotiation analysis."""
    def __init__(self, state_dim: int = 16, action_dim: int = 11) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.outcome_dim = 4  # [deal_value, acceptance, fairness, rounds_saved]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cf_net = CounterfactualNetwork(state_dim, action_dim, self.outcome_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.cf_net.parameters(), lr=1e-3)
        self._replay: list[tuple] = []

    def _encode(self, x: dict[str, Any] | list | None, dim: int) -> torch.Tensor:
        if isinstance(x, dict):
            vals = [float(v) for v in x.values() if isinstance(v, (int, float))]
        elif isinstance(x, list):
            vals = [float(v) for v in x]
        else:
            vals = []
        vals = (vals[:dim] + [0.0] * dim)[:dim]
        return torch.FloatTensor(vals).unsqueeze(0).to(self.device)

    def query(
        self,
        factual_state: dict[str, Any],
        factual_action: int,
        factual_outcome: dict[str, float],
        counterfactual_action: int,
    ) -> dict[str, Any]:
        """Answer: what would outcome have been with counterfactual_action?"""
        state_t = self._encode(factual_state, self.state_dim)
        action_f = torch.zeros(1, self.action_dim, device=self.device)
        action_f[0, factual_action] = 1.0
        action_cf = torch.zeros(1, self.action_dim, device=self.device)
        action_cf[0, counterfactual_action] = 1.0
        outcome_vals = list(factual_outcome.values())[:self.outcome_dim]
        outcome_vals = (outcome_vals + [0.0] * self.outcome_dim)[:self.outcome_dim]
        outcome_t = torch.FloatTensor(outcome_vals).unsqueeze(0).to(self.device)
        with torch.no_grad():
            noise = self.cf_net(state_t, action_f, outcome_t)
            cf_outcome = self.cf_net.predict_counterfactual(state_t, action_cf, noise)
        cf_outcome_np = cf_outcome.cpu().numpy()[0]
        return {
            "factual_action": factual_action,
            "counterfactual_action": counterfactual_action,
            "factual_outcome": factual_outcome,
            "counterfactual_outcome": {"deal_value": float(cf_outcome_np[0]), "acceptance": float(np.clip(cf_outcome_np[1], 0, 1)), "fairness": float(np.clip(cf_outcome_np[2], 0, 1)), "rounds_saved": float(cf_outcome_np[3])},
            "outcome_delta": round(float(cf_outcome_np[0]) - factual_outcome.get("deal_value", 0), 2),
        }

    def find_regret_minimizing_action(
        self, state: dict[str, Any], outcome: dict[str, float], current_action: int
    ) -> dict[str, Any]:
        """Find action with minimum regret via counterfactual queries."""
        best_action, best_value = current_action, outcome.get("deal_value", 0)
        all_cf = []
        for cf_action in range(self.action_dim):
            if cf_action == current_action: continue
            cf = self.query(state, current_action, outcome, cf_action)
            all_cf.append(cf)
            if cf["counterfactual_outcome"]["deal_value"] > best_value:
                best_value = cf["counterfactual_outcome"]["deal_value"]
                best_action = cf_action
        return {"regret_minimizing_action": best_action, "expected_value": round(best_value, 2), "alternatives_evaluated": len(all_cf)}
