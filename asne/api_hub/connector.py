"""Pluggable API connector framework."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import httpx
import structlog
from asne.core.config import APIHubConfig

logger = structlog.get_logger()

class BaseConnector(ABC):
    def __init__(self, name: str, config: APIHubConfig | None = None) -> None:
        self.name = name
        self.config = config or APIHubConfig()
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.config.timeout)
        return self._client

    @abstractmethod
    async def connect(self, credentials: dict[str, str]) -> bool: ...
    @abstractmethod
    async def fetch_data(self, query: dict[str, Any]) -> dict[str, Any]: ...
    @abstractmethod
    async def push_data(self, data: dict[str, Any]) -> bool: ...

    async def close(self) -> None:
        if self._client: await self._client.aclose(); self._client = None

class ConnectorRegistry:
    def __init__(self) -> None:
        self._connectors: dict[str, BaseConnector] = {}

    def register(self, connector: BaseConnector) -> None:
        self._connectors[connector.name] = connector
        logger.info("connector_registered", name=connector.name)

    def get(self, name: str) -> BaseConnector | None:
        return self._connectors.get(name)

    def list_connectors(self) -> list[str]:
        return list(self._connectors.keys())

    async def close_all(self) -> None:
        for c in self._connectors.values(): await c.close()
