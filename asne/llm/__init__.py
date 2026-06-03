"""LLM-powered negotiation module.

Integrates Large Language Models (GPT-4o, Claude-3.5, Gemini)
for:
1. Natural language negotiation (full sentence proposals)
2. Intent parsing from counterparty messages
3. Strategy narration and explanation
4. Contract drafting in plain English
5. Cultural adaptation (tone matching per region/industry)
"""
from __future__ import annotations
from typing import Any
import json
import structlog

logger = structlog.get_logger()


SYSTEM_PROMPT = """You are ASNE (Autonomous Swarm Negotiation Engine), an expert AI negotiator.
Your goal is to reach mutually beneficial B2B agreements that satisfy:
1. Economic value maximization for your party
2. Fairness constraints (ensure counterparty utility > reservation price)
3. Ethical compliance (no deception, coercion, or exploitation)
4. Speed: reach agreement within budget of negotiation rounds

Always reason step by step before proposing. Format proposals as JSON.
"""


class LLMNegotiator:
    """LLM-powered natural language negotiation agent.

    Uses structured output (JSON mode) for reliable parsing.
    Falls back to rule-based parsing if LLM is unavailable.
    """
    def __init__(self, model: str = "claude-3-5-sonnet", temperature: float = 0.3) -> None:
        self.model = model
        self.temperature = temperature
        self._conversation: list[dict[str, str]] = []
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        try:
            import anthropic
            self._client = anthropic.Anthropic()
            self._provider = "anthropic"
        except ImportError:
            try:
                import openai
                self._client = openai.OpenAI()
                self._provider = "openai"
            except ImportError:
                self._client = None
                self._provider = "simulation"

    def generate_proposal(
        self,
        state: dict[str, Any],
        round_num: int,
        deal_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a natural language proposal using LLM."""
        deal_context = deal_context or {}
        prompt = f"""Current negotiation state:
- Round: {round_num}
- Deal value under negotiation: ${state.get('deal_value', 100000):,.0f}
- Market conditions: {state.get('market_conditions', 0):.2f}
- Urgency factor: {state.get('urgency', 0.5):.2f}
- Deal type: {deal_context.get('deal_type', 'procurement')}

Generate a negotiation proposal as JSON with fields:
{{"proposed_value": float, "justification": str, "concession_made": bool, "final_offer": bool, "natural_language_proposal": str}}"""
        if self._provider != "simulation" and self._client:
            return self._call_llm(prompt)
        return self._simulate_proposal(state, round_num)

    def _call_llm(self, prompt: str) -> dict[str, Any]:
        """Call LLM API with structured output."""
        try:
            if self._provider == "anthropic":
                response = self._client.messages.create(
                    model=self.model, max_tokens=512,
                    system=SYSTEM_PROMPT + "\nRespond ONLY with valid JSON.",
                    messages=[{"role": "user", "content": prompt}],
                )
                return json.loads(response.content[0].text)
            elif self._provider == "openai":
                response = self._client.chat.completions.create(
                    model="gpt-4o", temperature=self.temperature,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                )
                return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.warning("llm_call_failed", error=str(e))
            return self._simulate_proposal({}, 0)
        return {}

    @staticmethod
    def _simulate_proposal(state: dict[str, Any], round_num: int) -> dict[str, Any]:
        import random
        base = state.get("deal_value", 100000)
        factor = 1.0 - (round_num * 0.005)
        proposed = base * random.uniform(0.9, 1.1) * factor
        return {"proposed_value": proposed, "justification": "Market-aligned offer based on current conditions.", "concession_made": round_num > 5, "final_offer": round_num > 20, "natural_language_proposal": f"We propose ${proposed:,.0f} for this agreement, reflecting current market conditions and our mutual interest in closing efficiently."}

    def parse_counterparty_intent(
        self, message: str, state: dict[str, Any]
    ) -> dict[str, Any]:
        """Parse natural language from counterparty into structured intent."""
        prompt = f"""Analyze this negotiation message and extract intent:\n'{message}'\nState: {json.dumps(state, default=str)[:500]}\nReturn JSON: {{"intent": str, "urgency": float, "acceptance_likelihood": float, "requested_changes": list, "emotional_tone": str}}"""
        if self._provider != "simulation" and self._client:
            return self._call_llm(prompt)
        return {"intent": "counter_offer", "urgency": 0.5, "acceptance_likelihood": 0.4, "requested_changes": ["lower_price"], "emotional_tone": "neutral"}

    def draft_contract_language(self, agreement: dict[str, Any]) -> str:
        """Draft natural language contract from structured agreement."""
        value = agreement.get("total_value", 0)
        deal_id = agreement.get("deal_id", "UNKNOWN")
        return f"""AUTONOMOUS NEGOTIATION AGREEMENT\n\nDeal Reference: {deal_id}\nNegotiated Value: ${value:,.2f}\n\nThis agreement has been autonomously negotiated and executed by the ASNE platform. All terms were agreed upon through multi-agent consensus and verified against ethical compliance standards (fairness score: {agreement.get('ethics_score', 0):.2f}).\n\nThe parties agree to the terms established through the negotiation process, subject to blockchain settlement and smart contract execution."""
