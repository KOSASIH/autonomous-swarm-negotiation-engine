"""B2B Deal negotiation environment."""
from __future__ import annotations
import random
from typing import Any
import numpy as np

class DealEnvironment:
    """Multi-issue B2B deal negotiation environment."""
    def __init__(self, deal_type: str = "procurement", parties: int = 2, constraints: dict[str, Any] | None = None, n_issues: int = 5) -> None:
        self.deal_type = deal_type
        self.n_parties = parties
        self.n_issues = n_issues
        self.constraints = constraints or {}
        self._budget = self.constraints.get("budget", 1_000_000)
        self._round = 0
        self._state: dict[str, Any] = {}

    def reset(self) -> dict[str, Any]:
        self._round = 0
        self._state = {
            "deal_type": self.deal_type, "deal_value": float(self._budget), "round": 0,
            "n_parties": self.n_parties, "issues": np.random.uniform(0, 1, self.n_issues).tolist(),
            "buyer_reservation": random.uniform(0.6, 0.8) * self._budget,
            "seller_reservation": random.uniform(0.8, 1.0) * self._budget,
            "market_conditions": random.uniform(-0.2, 0.2), "urgency": random.uniform(0, 1),
        }
        return self._state

    def step(self, actions: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict]:
        self._round += 1
        self._state["round"] = self._round
        if "proposed_value" in actions:
            self._state["deal_value"] = (self._state["deal_value"] + actions["proposed_value"]) / 2
        self._state["issues"] = [max(0, min(1, v + random.uniform(-0.05, 0.05))) for v in self._state["issues"]]
        deal_progress = 1.0 - abs(self._state["deal_value"] - self._budget) / self._budget
        done = self._round >= self.constraints.get("max_rounds", 100)
        return self._state, deal_progress, done, {"round": self._round, "deal_progress": deal_progress}
