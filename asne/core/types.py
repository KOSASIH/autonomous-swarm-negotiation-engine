"""Shared type definitions."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from datetime import datetime

class DealStatus(str, Enum):
    PENDING = "pending"
    NEGOTIATING = "negotiating"
    AGREED = "agreed"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

class AgentRole(str, Enum):
    BUYER = "buyer"
    SELLER = "seller"
    MEDIATOR = "mediator"
    OBSERVER = "observer"

@dataclass
class Proposal:
    agent_id: str
    round_num: int
    terms: dict[str, Any]
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class Agreement:
    deal_id: str
    parties: list[str]
    terms: dict[str, Any]
    total_value: float
    rounds_taken: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    ethics_score: float = 0.0
    transparency_hash: str = ""

@dataclass
class EthicsReport:
    fairness_score: float
    bias_flags: list[str] = field(default_factory=list)
    power_asymmetry: float = 0.0
    recommendations: list[str] = field(default_factory=list)
    compliant: bool = True

@dataclass
class NegotiationResult:
    status: DealStatus
    agreement: Agreement | None = None
    ethics_report: EthicsReport | None = None
    rounds_played: int = 0
    total_proposals: int = 0
    history: list[Proposal] = field(default_factory=list)
