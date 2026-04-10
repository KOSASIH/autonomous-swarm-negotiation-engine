"""Base negotiator agent."""
from __future__ import annotations
import uuid
from abc import ABC, abstractmethod
from typing import Any
from asne.core.types import AgentRole, Proposal

class BaseNegotiator(ABC):
    def __init__(self, role: AgentRole = AgentRole.BUYER, agent_id: str | None = None) -> None:
        self.agent_id = agent_id or str(uuid.uuid4())
        self.role = role
        self._round = 0

    @abstractmethod
    def propose(self, state: dict[str, Any]) -> Proposal: ...

    @abstractmethod
    def respond(self, proposal: Proposal, state: dict[str, Any]) -> bool: ...

    @abstractmethod
    def learn(self, reward: float, next_state: dict[str, Any]) -> None: ...

    def reset(self) -> None:
        self._round = 0
