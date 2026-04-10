"""Immutable transparency logging."""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

@dataclass
class LogEntry:
    timestamp: str
    event_type: str
    data: dict[str, Any]
    hash: str
    previous_hash: str

class TransparencyLog:
    """Hash-chained immutable audit log."""
    def __init__(self) -> None:
        self._entries: list[LogEntry] = []
        self._genesis_hash = hashlib.sha256(b"ASNE_GENESIS").hexdigest()

    def log_event(self, event_type: str, data: dict[str, Any]) -> LogEntry:
        previous_hash = self._entries[-1].hash if self._entries else self._genesis_hash
        timestamp = datetime.utcnow().isoformat()
        payload = json.dumps({"timestamp": timestamp, "event_type": event_type, "data": data, "prev": previous_hash}, sort_keys=True)
        entry_hash = hashlib.sha256(payload.encode()).hexdigest()
        entry = LogEntry(timestamp=timestamp, event_type=event_type, data=data, hash=entry_hash, previous_hash=previous_hash)
        self._entries.append(entry)
        return entry

    def verify_integrity(self) -> bool:
        if not self._entries: return True
        if self._entries[0].previous_hash != self._genesis_hash: return False
        for i in range(1, len(self._entries)):
            if self._entries[i].previous_hash != self._entries[i - 1].hash: return False
        return True

    def get_entries(self, event_type: str | None = None, limit: int = 100) -> list[LogEntry]:
        entries = [e for e in self._entries if e.event_type == event_type] if event_type else self._entries
        return entries[-limit:]

    @property
    def size(self) -> int:
        return len(self._entries)
