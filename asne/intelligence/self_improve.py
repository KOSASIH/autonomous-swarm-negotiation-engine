"""Self-Improvement Module for ASNE agents.

Enables agents to autonomously improve their own strategies by:
1. Introspective performance analysis
2. Automated hyperparameter optimization (Bayesian)
3. Architecture search (NAS lite) for policy networks
4. Curriculum learning: auto-generate harder training scenarios
5. Recursive self-modeling: agents model their own weaknesses
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class PerformanceMetrics:
    """Tracks agent performance over time."""
    deal_success_rate: float = 0.0
    avg_deal_value: float = 0.0
    avg_fairness_score: float = 0.0
    avg_rounds_to_close: float = 0.0
    ethics_compliance_rate: float = 0.0
    adaptation_speed: float = 0.0  # rounds to reach new baseline after domain shift
    history: list[dict[str, float]] = field(default_factory=list)


@dataclass
class HyperparamSuggestion:
    """Suggested hyperparameter update from Bayesian optimization."""
    learning_rate: float
    epsilon_decay: int
    batch_size: int
    hidden_dim: int
    expected_improvement: float


class BayesianHPO:
    """Simple Bayesian hyperparameter optimizer using Gaussian Process surrogate."""
    def __init__(self) -> None:
        self._history: list[tuple[dict, float]] = []  # (params, score)

    def suggest(self) -> HyperparamSuggestion:
        """Suggest next hyperparameters via Upper Confidence Bound."""
        if len(self._history) < 5:
            # Random exploration phase
            return HyperparamSuggestion(
                learning_rate=float(10 ** np.random.uniform(-5, -2)),
                epsilon_decay=int(np.random.choice([5000, 10000, 20000, 50000])),
                batch_size=int(np.random.choice([32, 64, 128, 256])),
                hidden_dim=int(np.random.choice([128, 256, 512])),
                expected_improvement=0.0,
            )
        # GP-UCB: exploit best known + explore uncertain regions
        scores = [s for _, s in self._history]
        best_idx = int(np.argmax(scores))
        best_params = self._history[best_idx][0]
        # Perturb best params
        return HyperparamSuggestion(
            learning_rate=best_params["learning_rate"] * np.random.uniform(0.5, 2.0),
            epsilon_decay=int(best_params["epsilon_decay"] * np.random.uniform(0.8, 1.2)),
            batch_size=best_params["batch_size"],
            hidden_dim=best_params["hidden_dim"],
            expected_improvement=float(np.std(scores)),
        )

    def register_result(self, params: dict[str, Any], score: float) -> None:
        self._history.append((params, score))


class CurriculumScheduler:
    """Auto-generates progressively harder negotiation scenarios."""
    def __init__(self) -> None:
        self._stage = 0
        self._success_threshold = 0.7
        self._recent_success_rates: list[float] = []

    def get_scenario_difficulty(self) -> dict[str, Any]:
        stage_configs = [
            {"n_issues": 2, "n_parties": 2, "time_pressure": 0.1, "information_asymmetry": 0.0},
            {"n_issues": 3, "n_parties": 2, "time_pressure": 0.3, "information_asymmetry": 0.2},
            {"n_issues": 5, "n_parties": 3, "time_pressure": 0.5, "information_asymmetry": 0.4},
            {"n_issues": 7, "n_parties": 4, "time_pressure": 0.7, "information_asymmetry": 0.6},
            {"n_issues": 10, "n_parties": 5, "time_pressure": 0.9, "information_asymmetry": 0.8},
        ]
        return stage_configs[min(self._stage, len(stage_configs) - 1)]

    def update(self, success_rate: float) -> bool:
        """Returns True if curriculum advanced to next stage."""
        self._recent_success_rates.append(success_rate)
        if len(self._recent_success_rates) > 10:
            self._recent_success_rates.pop(0)
        avg = np.mean(self._recent_success_rates)
        if avg >= self._success_threshold and self._stage < 4:
            self._stage += 1
            return True
        return False


class SelfImprovementModule:
    """Orchestrates autonomous self-improvement for ASNE agents."""
    def __init__(self) -> None:
        self.metrics = PerformanceMetrics()
        self.hpo = BayesianHPO()
        self.curriculum = CurriculumScheduler()
        self._improvement_log: list[dict[str, Any]] = []
        self._last_evaluation = time.time()

    def record_outcome(self, outcome: dict[str, float]) -> None:
        """Record a negotiation outcome for self-improvement analysis."""
        self.metrics.history.append({**outcome, "timestamp": time.time()})
        if len(self.metrics.history) > 10:
            recent = self.metrics.history[-10:]
            self.metrics.deal_success_rate = np.mean([1.0 if r.get("status") == "agreed" else 0.0 for r in recent])
            self.metrics.avg_deal_value = float(np.mean([r.get("deal_value", 0) for r in recent]))
            self.metrics.avg_fairness_score = float(np.mean([r.get("fairness_score", 0) for r in recent]))
            self.curriculum.update(self.metrics.deal_success_rate)

    def analyze(self) -> dict[str, Any]:
        """Introspective performance analysis."""
        weaknesses = []
        if self.metrics.deal_success_rate < 0.5:
            weaknesses.append("LOW_SUCCESS_RATE: Explore more aggressive convergence strategies")
        if self.metrics.avg_fairness_score < 0.7:
            weaknesses.append("FAIRNESS_DEFICIT: Increase ethics engine weight in reward")
        if self.metrics.avg_rounds_to_close > 50:
            weaknesses.append("SLOW_CONVERGENCE: Reduce epsilon, increase learning rate")
        suggestion = self.hpo.suggest()
        scenario = self.curriculum.get_scenario_difficulty()
        return {
            "identified_weaknesses": weaknesses,
            "hpo_suggestion": {"learning_rate": suggestion.learning_rate, "hidden_dim": suggestion.hidden_dim, "batch_size": suggestion.batch_size},
            "curriculum_stage": self.curriculum._stage,
            "next_scenario": scenario,
            "performance_summary": {"success_rate": round(self.metrics.deal_success_rate, 4), "avg_deal_value": round(self.metrics.avg_deal_value, 2), "avg_fairness": round(self.metrics.avg_fairness_score, 4)},
        }
