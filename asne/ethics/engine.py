"""Ethics engine for fair negotiations."""
from __future__ import annotations
import structlog
from typing import Any
from asne.core.config import EthicsConfig
from asne.core.types import EthicsReport, Proposal
from asne.ethics.transparency import TransparencyLog

logger = structlog.get_logger()

class EthicsEngine:
    def __init__(self, config: EthicsConfig | None = None, fairness_threshold: float | None = None) -> None:
        self.config = config or EthicsConfig()
        if fairness_threshold is not None: self.config.fairness_threshold = fairness_threshold
        self.transparency_log = TransparencyLog()
        self._proposal_history: list[Proposal] = []

    def evaluate_proposal(self, proposal: Proposal) -> bool:
        self._proposal_history.append(proposal)
        self.transparency_log.log_event("proposal_evaluated", {"agent_id": proposal.agent_id, "value": proposal.value, "round": proposal.round_num})
        if self._proposal_history:
            avg = sum(p.value for p in self._proposal_history) / len(self._proposal_history)
            if avg > 0 and abs(proposal.value - avg) / avg > 0.5:
                logger.warning("ethics_flag_extreme_value", agent_id=proposal.agent_id)
                return False
        return True

    def audit_negotiation(self, proposals: list[Proposal], parties: list[str]) -> EthicsReport:
        if not proposals: return EthicsReport(fairness_score=1.0, compliant=True)
        party_values: dict[str, list[float]] = {}
        for p in proposals: party_values.setdefault(p.agent_id, []).append(p.value)
        avg_by_party = {pid: sum(vals) / len(vals) for pid, vals in party_values.items()}
        all_avgs = list(avg_by_party.values())
        power_asymmetry = (max(all_avgs) - min(all_avgs)) / max(max(all_avgs), 1) if len(all_avgs) >= 2 else 0.0
        fairness_score = max(0, 1.0 - power_asymmetry)
        bias_flags = [f"Power asymmetry {power_asymmetry:.2f} exceeds threshold"] if power_asymmetry > self.config.max_power_asymmetry else []
        compliant = fairness_score >= self.config.fairness_threshold and len(bias_flags) == 0
        return EthicsReport(fairness_score=round(fairness_score, 4), bias_flags=bias_flags, power_asymmetry=round(power_asymmetry, 4), compliant=compliant)
