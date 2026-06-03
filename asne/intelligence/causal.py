"""Causal Reasoning Engine for negotiation.

Goes beyond correlation-based ML to model CAUSE and EFFECT
in negotiation dynamics. Uses Structural Causal Models (SCMs)
and do-calculus to reason about interventions:
  - What WOULD happen if I proposed X?
  - What CAUSED the counterparty to reject?
  - How to intervene to achieve outcome Y?
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class CausalNode:
    """Node in a Structural Causal Model."""
    name: str
    parents: list[str] = field(default_factory=list)
    mechanism: str = "linear"  # linear | nonlinear | threshold
    noise_std: float = 0.1
    _data: list[float] = field(default_factory=list)


class StructuralCausalModel:
    """Structural Causal Model for negotiation dynamics."""
    def __init__(self) -> None:
        self.nodes: dict[str, CausalNode] = {}
        self._adjacency: dict[str, list[str]] = {}

    def add_node(self, node: CausalNode) -> None:
        self.nodes[node.name] = node
        self._adjacency[node.name] = node.parents

    def compute(self, assignments: dict[str, float], do_interventions: dict[str, float] | None = None) -> dict[str, float]:
        """Forward pass through SCM with optional do-calculus interventions."""
        do_interventions = do_interventions or {}
        results = dict(assignments)
        topo_order = self._topological_sort()
        for node_name in topo_order:
            if node_name in do_interventions:
                results[node_name] = do_interventions[node_name]
                continue
            node = self.nodes.get(node_name)
            if not node or not node.parents:
                if node_name not in results:
                    results[node_name] = 0.0
                continue
            parent_values = [results.get(p, 0.0) for p in node.parents]
            if node.mechanism == "linear":
                weights = np.ones(len(parent_values)) / len(parent_values)
                value = float(np.dot(weights, parent_values))
            elif node.mechanism == "threshold":
                value = 1.0 if np.mean(parent_values) > 0.5 else 0.0
            else:  # nonlinear
                value = float(np.tanh(np.mean(parent_values)))
            value += np.random.normal(0, node.noise_std)
            results[node_name] = value
        return results

    def _topological_sort(self) -> list[str]:
        visited, order = set(), []
        def dfs(node: str) -> None:
            if node in visited: return
            visited.add(node)
            for parent in self._adjacency.get(node, []):
                dfs(parent)
            order.append(node)
        for n in self.nodes: dfs(n)
        return order


class CausalReasoningEngine:
    """Causal reasoning for negotiation decision-making.

    Builds and queries an SCM of the negotiation process to enable:
    - Counterfactual queries (what if?)
    - Intervention planning (what should I do?)
    - Root cause analysis (why did this fail?)
    """
    def __init__(self) -> None:
        self.scm = StructuralCausalModel()
        self._build_negotiation_scm()
        self._history: list[dict[str, float]] = []

    def _build_negotiation_scm(self) -> None:
        """Construct the default negotiation causal graph."""
        nodes = [
            CausalNode("market_conditions", []),
            CausalNode("urgency", []),
            CausalNode("budget", []),
            CausalNode("proposed_value", ["market_conditions", "budget"]),
            CausalNode("counterparty_utility", ["proposed_value", "urgency"], mechanism="nonlinear"),
            CausalNode("acceptance_probability", ["counterparty_utility", "market_conditions"], mechanism="threshold"),
            CausalNode("deal_quality", ["proposed_value", "counterparty_utility"]),
        ]
        for node in nodes:
            self.scm.add_node(node)

    def estimate_causal_effect(
        self,
        treatment: str,
        outcome: str,
        treatment_value: float,
        control_value: float,
        context: dict[str, float],
    ) -> dict[str, float]:
        """Estimate ATE (Average Treatment Effect) via do-calculus."""
        y_treated = self.scm.compute(context, do_interventions={treatment: treatment_value})
        y_control = self.scm.compute(context, do_interventions={treatment: control_value})
        ate = y_treated.get(outcome, 0) - y_control.get(outcome, 0)
        return {
            "ate": round(ate, 6),
            "treated_outcome": round(y_treated.get(outcome, 0), 6),
            "control_outcome": round(y_control.get(outcome, 0), 6),
            "treatment": treatment,
            "outcome": outcome,
        }

    def find_optimal_intervention(
        self,
        target_node: str,
        target_value: float,
        controllable_nodes: list[str],
        context: dict[str, float],
        n_samples: int = 50,
    ) -> dict[str, Any]:
        """Find the intervention on controllable nodes that best achieves target."""
        best_intervention, best_distance = {}, float("inf")
        for _ in range(n_samples):
            intervention = {node: float(np.random.uniform(0, 1)) for node in controllable_nodes}
            result = self.scm.compute(context, do_interventions=intervention)
            distance = abs(result.get(target_node, 0) - target_value)
            if distance < best_distance:
                best_distance = distance
                best_intervention = dict(intervention)
        return {"optimal_intervention": best_intervention, "expected_distance": round(best_distance, 6)}

    def diagnose_failure(self, negotiation_trace: list[dict[str, float]]) -> list[str]:
        """Root cause analysis for failed negotiations."""
        if not negotiation_trace:
            return ["Insufficient negotiation data for diagnosis"]
        causes = []
        avg_acceptance = np.mean([t.get("acceptance_probability", 0.5) for t in negotiation_trace])
        avg_utility = np.mean([t.get("counterparty_utility", 0.5) for t in negotiation_trace])
        if avg_acceptance < 0.3:
            causes.append("LOW_ACCEPTANCE: Proposed values consistently below counterparty threshold")
        if avg_utility < 0.4:
            causes.append("LOW_UTILITY: Counterparty utility persistently low — revisit issue weighting")
        if len(negotiation_trace) > 50:
            causes.append("CONVERGENCE_FAILURE: Too many rounds — agents not converging, consider mediator")
        return causes if causes else ["NO_CLEAR_CAUSE: Stochastic failure, retry recommended"]
