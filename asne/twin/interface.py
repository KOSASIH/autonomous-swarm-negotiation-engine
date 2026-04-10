"""Digital Twin integration interface."""
from __future__ import annotations
from typing import Any
from dataclasses import dataclass, field
import structlog

logger = structlog.get_logger()

@dataclass
class TwinState:
    entity_id: str
    entity_type: str
    attributes: dict[str, Any] = field(default_factory=dict)
    capabilities: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    health_score: float = 1.0

class DigitalTwinInterface:
    """Interface for syncing with organizational digital twins."""
    def __init__(self) -> None:
        self._twins: dict[str, TwinState] = {}

    def register_twin(self, twin: TwinState) -> None:
        self._twins[twin.entity_id] = twin
        logger.info("twin_registered", entity_id=twin.entity_id, type=twin.entity_type)

    def get_twin(self, entity_id: str) -> TwinState | None:
        return self._twins.get(entity_id)

    def get_negotiation_context(self, entity_id: str) -> dict[str, Any]:
        twin = self._twins.get(entity_id)
        if not twin: return {}
        return {"entity_id": twin.entity_id, "capabilities": twin.capabilities, "constraints": twin.constraints, "health_score": twin.health_score, "budget_available": twin.attributes.get("budget", 0), "risk_tolerance": twin.attributes.get("risk_tolerance", 0.5)}

    def update_twin(self, entity_id: str, updates: dict[str, Any]) -> bool:
        twin = self._twins.get(entity_id)
        if not twin: return False
        twin.attributes.update(updates.get("attributes", {}))
        if "health_score" in updates: twin.health_score = updates["health_score"]
        if "constraints" in updates: twin.constraints.update(updates["constraints"])
        return True
