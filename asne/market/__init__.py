"""Real-time market intelligence for negotiation context.

Provides live market data feeds that inform agent strategies:
- Commodity/asset prices
- Supply chain disruption signals
- Competitor pricing intelligence
- Macroeconomic indicators
- Industry-specific deal benchmarks
"""
from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator
import structlog

logger = structlog.get_logger()


@dataclass
class MarketSignal:
    signal_id: str
    category: str  # price | supply_chain | macro | sentiment | competitor
    ticker: str
    value: float
    change_pct: float
    confidence: float
    timestamp: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


class MarketIntelligenceEngine:
    """Aggregates and synthesizes market signals for negotiation."""
    def __init__(self, update_interval_s: float = 60.0) -> None:
        self._signals: dict[str, list[MarketSignal]] = {}
        self._update_interval = update_interval_s
        self._running = False
        self._callbacks: list = []

    async def start_streaming(self) -> None:
        """Start streaming market data (connects to real APIs in production)."""
        self._running = True
        logger.info("market_intelligence_streaming_started")
        while self._running:
            await self._fetch_market_data()
            await asyncio.sleep(self._update_interval)

    def stop(self) -> None:
        self._running = False

    async def _fetch_market_data(self) -> None:
        """Fetch data from market APIs (simulated here)."""
        import random
        # Simulate real-time market signals
        signals = [
            MarketSignal(signal_id=f"sig_{int(time.time())}_1", category="price", ticker="COMMODITY_STEEL", value=random.uniform(800, 1200), change_pct=random.uniform(-5, 5), confidence=0.9, timestamp=time.time(), source="bloomberg_sim"),
            MarketSignal(signal_id=f"sig_{int(time.time())}_2", category="supply_chain", ticker="LOGISTICS_INDEX", value=random.uniform(90, 110), change_pct=random.uniform(-3, 3), confidence=0.85, timestamp=time.time(), source="freightos_sim"),
            MarketSignal(signal_id=f"sig_{int(time.time())}_3", category="macro", ticker="INFLATION_YOY", value=random.uniform(2, 8), change_pct=random.uniform(-0.5, 0.5), confidence=0.95, timestamp=time.time(), source="fed_sim"),
            MarketSignal(signal_id=f"sig_{int(time.time())}_4", category="sentiment", ticker="DEAL_SENTIMENT", value=random.uniform(0.4, 0.9), change_pct=random.uniform(-10, 10), confidence=0.7, timestamp=time.time(), source="nlp_sentiment"),
        ]
        for signal in signals:
            if signal.ticker not in self._signals:
                self._signals[signal.ticker] = []
            self._signals[signal.ticker].append(signal)
            if len(self._signals[signal.ticker]) > 100:
                self._signals[signal.ticker] = self._signals[signal.ticker][-100:]
        for cb in self._callbacks:
            await cb(signals)

    def get_negotiation_context(self, deal_type: str = "procurement") -> dict[str, float]:
        """Synthesize market signals into negotiation context."""
        context: dict[str, float] = {}
        for ticker, signals in self._signals.items():
            if signals:
                latest = signals[-1]
                context[ticker.lower()] = latest.value
                context[f"{ticker.lower()}_trend"] = latest.change_pct
        return context

    def get_fair_value_estimate(
        self, asset_type: str, base_value: float
    ) -> dict[str, float]:
        """Estimate fair market value using available signals."""
        macro_factor = 1.0
        sentiment_factor = 1.0
        if "INFLATION_YOY" in self._signals and self._signals["INFLATION_YOY"]:
            inflation = self._signals["INFLATION_YOY"][-1].value / 100
            macro_factor = 1 + inflation * 0.1
        if "DEAL_SENTIMENT" in self._signals and self._signals["DEAL_SENTIMENT"]:
            sentiment = self._signals["DEAL_SENTIMENT"][-1].value
            sentiment_factor = 0.9 + sentiment * 0.2
        fair_value = base_value * macro_factor * sentiment_factor
        return {"fair_value": round(fair_value, 2), "low_estimate": round(fair_value * 0.85, 2), "high_estimate": round(fair_value * 1.15, 2), "macro_factor": round(macro_factor, 4), "sentiment_factor": round(sentiment_factor, 4), "confidence": 0.75}

    def register_callback(self, callback) -> None:
        """Register async callback for new market signals."""
        self._callbacks.append(callback)

    def subscribe(self, tickers: list[str]) -> None:
        """Subscribe to specific market tickers."""
        for ticker in tickers:
            if ticker not in self._signals:
                self._signals[ticker] = []
        logger.info("market_subscriptions_added", tickers=tickers)


class EmergentSwarmProtocol:
    """Emergent communication protocol for agent swarms.

    Agents develop their own shared vocabulary and communication
    patterns through interaction, without explicit programming.
    Inspired by multi-agent emergent language research.
    """
    def __init__(self, vocab_size: int = 64, msg_len: int = 8) -> None:
        self.vocab_size = vocab_size
        self.msg_len = msg_len
        self._message_history: list[dict[str, Any]] = []
        self._symbol_meanings: dict[int, str] = {}

    def encode_intent(
        self, intent: dict[str, Any], sender_id: str
    ) -> list[int]:
        """Encode agent intent as emergent symbol sequence."""
        import hashlib
        payload = str(sorted(intent.items()))
        hash_bytes = hashlib.md5(payload.encode()).digest()
        symbols = [b % self.vocab_size for b in hash_bytes[:self.msg_len]]
        self._message_history.append({"sender": sender_id, "intent": intent, "symbols": symbols})
        return symbols

    def decode_message(
        self, symbols: list[int], sender_history: list[dict]
    ) -> dict[str, Any]:
        """Decode emergent symbols by correlating with sender history."""
        if not sender_history:
            return {"decoded": False, "intent": "unknown"}
        # Find most similar historical encoding
        best_match = None
        best_overlap = 0
        for record in sender_history:
            overlap = sum(1 for a, b in zip(symbols, record.get("symbols", [])) if a == b)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = record
        if best_match and best_overlap >= self.msg_len // 2:
            return {"decoded": True, "intent": best_match.get("intent", {}), "confidence": best_overlap / self.msg_len}
        return {"decoded": False, "intent": "unknown", "confidence": 0.0}
