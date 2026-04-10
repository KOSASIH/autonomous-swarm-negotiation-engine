"""Swarm coordination for multi-agent negotiation."""
from __future__ import annotations
import structlog
from typing import Any
from asne.agents.base import BaseNegotiator
from asne.agents.dqn import DQNNegotiator
from asne.core.config import AgentConfig
from asne.core.types import AgentRole, Agreement, DealStatus, EthicsReport, NegotiationResult, Proposal

logger = structlog.get_logger()

class NegotiatorSwarm:
    def __init__(self, n_agents: int = 5, strategy: str = "self_play", config: AgentConfig | None = None, ethics_engine: Any = None) -> None:
        self.config = config or AgentConfig(n_agents=n_agents)
        self.strategy = strategy
        self.ethics_engine = ethics_engine
        self.agents: list[BaseNegotiator] = []
        self._initialize_agents()

    def _initialize_agents(self) -> None:
        for i in range(self.config.n_agents):
            role = AgentRole.BUYER if i % 2 == 0 else AgentRole.SELLER
            agent = DQNNegotiator(state_dim=32, action_dim=11, config=self.config, role=role)
            self.agents.append(agent)
            logger.info("agent_initialized", agent_id=agent.agent_id, role=role.value)

    def negotiate(self, environment: Any, max_rounds: int = 100) -> NegotiationResult:
        history: list[Proposal] = []
        state = environment.reset() if hasattr(environment, "reset") else {}
        for round_num in range(1, max_rounds + 1):
            round_proposals = []
            for agent in self.agents:
                proposal = agent.propose(state)
                round_proposals.append(proposal)
                history.append(proposal)
                if self.ethics_engine and hasattr(self.ethics_engine, "evaluate_proposal"):
                    if not self.ethics_engine.evaluate_proposal(proposal):
                        logger.warning("ethics_violation", agent_id=agent.agent_id, round=round_num)
            if self._check_consensus(round_proposals):
                agreement = self._form_agreement(round_proposals, round_num)
                ethics_report = self._generate_ethics_report(history)
                logger.info("deal_reached", rounds=round_num, value=agreement.total_value)
                return NegotiationResult(status=DealStatus.AGREED, agreement=agreement, ethics_report=ethics_report, rounds_played=round_num, total_proposals=len(history), history=history)
            for agent in self.agents:
                reward = self._calculate_reward(agent, round_proposals)
                agent.learn(reward, state)
        return NegotiationResult(status=DealStatus.TIMEOUT, rounds_played=max_rounds, total_proposals=len(history), history=history, ethics_report=self._generate_ethics_report(history))

    def _check_consensus(self, proposals: list[Proposal]) -> bool:
        if len(proposals) < 2: return False
        values = [p.value for p in proposals]
        return (max(values) - min(values)) / max(max(values), 1) < 0.05

    def _form_agreement(self, proposals: list[Proposal], round_num: int) -> Agreement:
        avg_value = sum(p.value for p in proposals) / len(proposals)
        return Agreement(deal_id=f"deal-{round_num}", parties=[p.agent_id for p in proposals], terms={"negotiated_value": avg_value}, total_value=avg_value, rounds_taken=round_num)

    def _calculate_reward(self, agent: BaseNegotiator, proposals: list[Proposal]) -> float:
        agent_proposal = next((p for p in proposals if p.agent_id == agent.agent_id), None)
        if not agent_proposal: return 0.0
        avg_value = sum(p.value for p in proposals) / len(proposals)
        return max(0, 1.0 - abs(agent_proposal.value - avg_value) / max(avg_value, 1))

    def _generate_ethics_report(self, history: list[Proposal]) -> EthicsReport:
        if not history: return EthicsReport(fairness_score=1.0)
        values = [p.value for p in history]
        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        normalized_var = min(variance / max(mean_val**2, 1), 1.0)
        fairness = 1.0 - normalized_var
        return EthicsReport(fairness_score=round(fairness, 4), power_asymmetry=round(normalized_var, 4), compliant=fairness >= 0.85)
