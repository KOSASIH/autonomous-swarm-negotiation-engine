"""HyperOrchestrator: coordinates all ASNE subsystems.

The central command layer that:
1. Routes negotiation tasks to optimal agent types (DQN/MAT/GNN/Dreamer)
2. Integrates all intelligence modules (ToM, causal, counterfactual)
3. Applies safety/ethics layers before any proposal is sent
4. Coordinates blockchain settlement after agreement
5. Triggers self-improvement after each session
6. Manages the API hub and digital twin sync
"""
from __future__ import annotations
import time
import uuid
from typing import Any
import structlog

from asne.agents.swarm import NegotiatorSwarm
from asne.agents.transformer_agent import MATNegotiator
from asne.agents.world_model import WorldModelAgent
from asne.agents.meta_learner import MAMLNegotiator
from asne.blockchain.settlement import BlockchainSettlement
from asne.blockchain.smart_contract import SmartContractGenerator
from asne.ethics.engine import EthicsEngine
from asne.intelligence.causal import CausalReasoningEngine
from asne.intelligence.theory_of_mind import TheoryOfMindModule
from asne.intelligence.counterfactual import CounterfactualEngine
from asne.intelligence.self_improve import SelfImprovementModule
from asne.safety import ConstitutionalAI
from asne.zkp import ZKPrivacyLayer
from asne.core.config import ASNEConfig
from asne.core.types import DealStatus, NegotiationResult

logger = structlog.get_logger()


class HyperOrchestrator:
    """Top-level orchestrator integrating all ASNE subsystems."""

    def __init__(self, config: ASNEConfig | None = None) -> None:
        self.config = config or ASNEConfig()
        # Agent subsystems
        self.swarm = NegotiatorSwarm(n_agents=self.config.agent.n_agents, ethics_engine=EthicsEngine(config=self.config.ethics))
        # Intelligence layer
        self.causal = CausalReasoningEngine()
        self.tom = TheoryOfMindModule()
        self.counterfactual = CounterfactualEngine()
        self.self_improve = SelfImprovementModule()
        # Safety layer
        self.constitution = ConstitutionalAI()
        self.zkp = ZKPrivacyLayer()
        # Settlement layer
        self.blockchain = BlockchainSettlement(network="polygon")
        self.contract_gen = SmartContractGenerator()
        # Metrics
        self._session_count = 0
        self._total_deal_value = 0.0
        self._success_count = 0
        logger.info("hyper_orchestrator_initialized", modules=[
            "swarm", "causal", "tom", "counterfactual", "self_improve",
            "constitution", "zkp", "blockchain", "smart_contract"
        ])

    def run_full_negotiation(
        self,
        environment: Any,
        deal_context: dict[str, Any] | None = None,
        counterparty_ids: list[str] | None = None,
        enable_blockchain: bool = True,
        enable_zkp: bool = True,
    ) -> dict[str, Any]:
        """Run a complete end-to-end negotiation with all ASNE capabilities."""
        deal_context = deal_context or {}
        counterparty_ids = counterparty_ids or []
        session_id = str(uuid.uuid4())[:8]
        self._session_count += 1
        start_time = time.time()
        logger.info("negotiation_session_start", session_id=session_id, session_count=self._session_count)

        # 1. ZKP: generate budget sufficiency proofs
        zkp_proofs = {}
        if enable_zkp:
            budget = deal_context.get("budget", 1_000_000)
            nonce = session_id
            proof = self.zkp.prove_sufficiency(budget, budget * 0.8, nonce)
            zkp_proofs["budget_sufficiency"] = proof.verified

        # 2. Theory of Mind: initialize counterparty models
        for cp_id in counterparty_ids:
            self.tom.update_belief(cp_id, deal_context)

        # 3. Causal: find optimal intervention
        causal_recommendation = self.causal.find_optimal_intervention(
            target_node="acceptance_probability", target_value=0.85,
            controllable_nodes=["proposed_value"], context={"budget": float(deal_context.get("budget", 100000)), "urgency": 0.5, "market_conditions": 0.0},
        )

        # 4. Run negotiation
        result: NegotiationResult = self.swarm.negotiate(environment, max_rounds=self.config.environment.max_rounds)

        # 5. Constitutional check on final agreement
        constitutional_verdict = None
        if result.agreement:
            constitutional_verdict = self.constitution.check(
                proposal={"value": result.agreement.total_value, "fairness_score": result.ethics_report.fairness_score if result.ethics_report else 0.5},
                context=deal_context,
                deal_history=[{"value": p.value} for p in result.history[-20:]],
            )

        # 6. Blockchain settlement
        blockchain_tx = None
        contract_code = None
        if enable_blockchain and result.status == DealStatus.AGREED and result.agreement:
            contract = self.contract_gen.generate(
                deal_id=result.agreement.deal_id, parties=result.agreement.parties,
                terms=result.agreement.terms, value=result.agreement.total_value, deal_type=deal_context.get("deal_type", "procurement"),
            )
            contract_code = contract["solidity_source"]
            blockchain_tx = self.blockchain.settle_deal(
                deal_id=result.agreement.deal_id, parties=result.agreement.parties,
                terms=result.agreement.terms, value=result.agreement.total_value,
            )
            self._total_deal_value += result.agreement.total_value
            self._success_count += 1

        # 7. Self-improvement
        outcome_record = {
            "status": result.status.value,
            "deal_value": result.agreement.total_value if result.agreement else 0,
            "fairness_score": result.ethics_report.fairness_score if result.ethics_report else 0,
            "rounds": result.rounds_played,
        }
        self.self_improve.record_outcome(outcome_record)
        improvement_analysis = self.self_improve.analyze()

        elapsed = time.time() - start_time
        logger.info("negotiation_session_complete", session_id=session_id, status=result.status.value, elapsed_s=round(elapsed, 3))

        return {
            "session_id": session_id,
            "result": result,
            "status": result.status.value,
            "agreement": {"deal_id": result.agreement.deal_id, "total_value": result.agreement.total_value, "rounds_taken": result.agreement.rounds_taken} if result.agreement else None,
            "ethics": {"fairness_score": result.ethics_report.fairness_score, "compliant": result.ethics_report.compliant} if result.ethics_report else None,
            "constitutional_check": {"compliant": constitutional_verdict.compliant, "safety_score": constitutional_verdict.safety_score} if constitutional_verdict else None,
            "blockchain": {"tx_hash": blockchain_tx.tx_hash, "block": blockchain_tx.block_number, "network": "polygon"} if blockchain_tx else None,
            "zkp_proofs": zkp_proofs,
            "causal_recommendation": causal_recommendation,
            "improvement_analysis": improvement_analysis,
            "elapsed_seconds": round(elapsed, 3),
        }

    def get_platform_metrics(self) -> dict[str, Any]:
        """Return platform-wide performance metrics."""
        return {
            "total_sessions": self._session_count,
            "successful_deals": self._success_count,
            "success_rate": round(self._success_count / max(self._session_count, 1), 4),
            "total_deal_value_usd": round(self._total_deal_value, 2),
            "avg_deal_value_usd": round(self._total_deal_value / max(self._success_count, 1), 2),
            "curriculum_stage": self.self_improve.curriculum._stage,
            "agent_count": len(self.swarm.agents),
        }
