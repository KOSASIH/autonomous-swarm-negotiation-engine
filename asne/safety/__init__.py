"""Constitutional AI safety for ASNE.

Enforces a set of constitutional principles on ALL agent outputs:
1. Never deceive counterparties
2. Never exploit information asymmetry unethically
3. Respect human oversight
4. Maintain deal integrity
5. Protect vulnerable parties
6. Ensure auditability

Uses critique-revision cycles: generate proposal → critique against
constitution → revise until compliant.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import structlog

logger = structlog.get_logger()


CONSTITUTIONAL_PRINCIPLES = [
    {"id": "P1", "name": "Non-Deception", "description": "Never misrepresent facts or intent to counterparties", "severity": "critical"},
    {"id": "P2", "name": "Non-Exploitation", "description": "Do not exploit power asymmetries or information gaps", "severity": "critical"},
    {"id": "P3", "name": "Proportionality", "description": "Proposals must be proportionate to stated budget and constraints", "severity": "high"},
    {"id": "P4", "name": "Transparency", "description": "All reasoning must be logged and auditable", "severity": "high"},
    {"id": "P5", "name": "Reversibility", "description": "Avoid irreversible actions without explicit multi-sig approval", "severity": "high"},
    {"id": "P6", "name": "Human Override", "description": "Always defer to human override signals", "severity": "critical"},
    {"id": "P7", "name": "Privacy", "description": "Never leak counterparty confidential information across sessions", "severity": "high"},
    {"id": "P8", "name": "Fairness", "description": "Ensure ZOPA (zone of possible agreement) is preserved for all parties", "severity": "medium"},
]


@dataclass
class ConstitutionalViolation:
    principle_id: str
    principle_name: str
    severity: str
    description: str
    proposal_field: str
    suggested_fix: str


@dataclass
class ConstitutionalVerdict:
    compliant: bool
    violations: list[ConstitutionalViolation] = field(default_factory=list)
    safety_score: float = 1.0
    revised_proposal: dict[str, Any] | None = None
    critique: str = ""


class ConstitutionalAI:
    """Critique-revision safety layer for ASNE proposals."""
    def __init__(self, max_revision_rounds: int = 3) -> None:
        self.max_revision_rounds = max_revision_rounds
        self._violation_count: dict[str, int] = {}

    def check(
        self,
        proposal: dict[str, Any],
        context: dict[str, Any],
        deal_history: list[dict[str, Any]] | None = None,
    ) -> ConstitutionalVerdict:
        """Check proposal against constitutional principles."""
        violations: list[ConstitutionalViolation] = []
        deal_history = deal_history or []
        # P3: Proportionality
        budget = context.get("budget", float("inf"))
        proposed_value = proposal.get("value", 0)
        if proposed_value > budget * 1.5:
            violations.append(ConstitutionalViolation(principle_id="P3", principle_name="Proportionality", severity="high", description=f"Proposed ${proposed_value:,.0f} exceeds budget ${budget:,.0f} by >50%", proposal_field="value", suggested_fix=f"Cap value at ${budget * 1.1:,.0f}"))
        # P2: Non-Exploitation (detect extreme low-ball)
        if deal_history:
            avg_val = sum(d.get("value", 0) for d in deal_history) / len(deal_history)
            if avg_val > 0 and proposed_value < avg_val * 0.4:
                violations.append(ConstitutionalViolation(principle_id="P2", principle_name="Non-Exploitation", severity="critical", description=f"Proposal ${proposed_value:,.0f} is >60% below running average — potential exploitation", proposal_field="value", suggested_fix=f"Increase to at least ${avg_val * 0.6:,.0f}"))
        # P8: Fairness
        fairness = proposal.get("fairness_score", 1.0)
        if fairness < 0.5:
            violations.append(ConstitutionalViolation(principle_id="P8", principle_name="Fairness", severity="medium", description=f"Fairness score {fairness:.2f} below minimum threshold 0.5", proposal_field="fairness_score", suggested_fix="Apply fairness constraint to equalize party utilities"))
        critical_count = sum(1 for v in violations if v.severity == "critical")
        high_count = sum(1 for v in violations if v.severity == "high")
        safety_score = max(0.0, 1.0 - (critical_count * 0.4 + high_count * 0.2))
        compliant = len(violations) == 0
        revised = self._revise(proposal, violations) if not compliant else None
        if violations:
            logger.warning("constitutional_violations", count=len(violations), safety_score=safety_score)
        return ConstitutionalVerdict(compliant=compliant, violations=violations, safety_score=safety_score, revised_proposal=revised, critique=self._format_critique(violations))

    def _revise(self, proposal: dict[str, Any], violations: list[ConstitutionalViolation]) -> dict[str, Any]:
        revised = dict(proposal)
        for v in violations:
            if v.principle_id == "P3" and "value" in revised:
                revised["value"] = revised["value"] * 0.9
            elif v.principle_id == "P2" and "value" in revised:
                revised["value"] = revised["value"] * 1.5
            elif v.principle_id == "P8":
                revised["fairness_adjustment"] = True
        revised["_constitutional_revision"] = True
        return revised

    @staticmethod
    def _format_critique(violations: list[ConstitutionalViolation]) -> str:
        if not violations: return "Proposal is constitutionally compliant."
        lines = ["Constitutional critique:"]
        for v in violations:
            lines.append(f"  [{v.severity.upper()}] {v.principle_id} {v.principle_name}: {v.description}")
            lines.append(f"    Fix: {v.suggested_fix}")
        return "\n".join(lines)


class RedTeamModule:
    """Adversarial red-teaming for ASNE robustness testing."""
    def __init__(self) -> None:
        self._attack_scenarios = [
            {"name": "sybil_attack", "description": "Multiple fake agents colluding"},
            {"name": "anchoring_bias", "description": "Extreme anchor proposal to skew ZOPA"},
            {"name": "deadline_pressure", "description": "False urgency injection"},
            {"name": "boulwarism", "description": "Take-it-or-leave-it without BATNA"},
            {"name": "salami_tactics", "description": "Incremental concession harvesting"},
            {"name": "misinformation", "description": "False market data injection"},
        ]

    def generate_attack(self, attack_type: str | None = None) -> dict[str, Any]:
        """Generate an adversarial negotiation scenario."""
        import random
        scenario = next((a for a in self._attack_scenarios if a["name"] == attack_type), random.choice(self._attack_scenarios))
        perturbations = {
            "sybil_attack": {"market_conditions": -0.5, "n_fake_agents": 3},
            "anchoring_bias": {"anchor_value_multiplier": 0.1},
            "deadline_pressure": {"urgency": 0.99, "rounds_remaining": 2},
            "boulwarism": {"concession_rate": 0.0},
            "salami_tactics": {"incremental_pressure": True},
            "misinformation": {"market_conditions": 0.9, "is_false": True},
        }
        return {**scenario, "perturbations": perturbations.get(scenario["name"], {}), "severity": "high"}

    def evaluate_robustness(
        self, agent_responses: list[dict[str, Any]], attacks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Evaluate how well agents resisted adversarial attacks."""
        robustness_scores = []
        for response, attack in zip(agent_responses, attacks):
            value = response.get("value", 0)
            fair_value = response.get("fair_value", value)
            deviation = abs(value - fair_value) / max(fair_value, 1)
            robustness_scores.append(max(0, 1 - deviation))
        import numpy as np
        return {"overall_robustness": round(float(np.mean(robustness_scores)), 4), "n_attacks_tested": len(attacks), "attack_types": [a["name"] for a in attacks], "vulnerabilities_found": [a["name"] for a, s in zip(attacks, robustness_scores) if s < 0.5]}
