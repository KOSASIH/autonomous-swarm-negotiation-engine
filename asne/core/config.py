"""Application configuration."""
from __future__ import annotations
from pydantic import BaseModel, Field

class AgentConfig(BaseModel):
    n_agents: int = Field(default=5, ge=1, le=100)
    learning_rate: float = Field(default=3e-4, gt=0)
    gamma: float = Field(default=0.99, ge=0, le=1)
    epsilon_start: float = Field(default=1.0)
    epsilon_end: float = Field(default=0.01)
    epsilon_decay: int = Field(default=10000)
    batch_size: int = Field(default=64, ge=1)
    memory_size: int = Field(default=100000, ge=1000)
    hidden_dim: int = Field(default=256, ge=32)
    self_play: bool = Field(default=True)

class EnvironmentConfig(BaseModel):
    max_rounds: int = Field(default=100, ge=1)
    deal_value_range: tuple[float, float] = Field(default=(1000.0, 1_000_000.0))
    n_issues: int = Field(default=5, ge=1)
    timeout_seconds: float = Field(default=300.0, gt=0)

class QuantumConfig(BaseModel):
    backend: str = Field(default="aer_simulator")
    n_qubits: int = Field(default=8, ge=2, le=32)
    shots: int = Field(default=1024, ge=100)
    optimization_level: int = Field(default=1, ge=0, le=3)
    use_pennylane: bool = Field(default=False)

class EthicsConfig(BaseModel):
    fairness_threshold: float = Field(default=0.85, ge=0, le=1)
    bias_detection: bool = Field(default=True)
    transparency_logging: bool = Field(default=True)
    max_power_asymmetry: float = Field(default=0.3, ge=0, le=1)

class APIHubConfig(BaseModel):
    redis_url: str = Field(default="redis://localhost:6379/0")
    rate_limit: int = Field(default=100, ge=1)
    timeout: float = Field(default=30.0, gt=0)
    retry_max: int = Field(default=3, ge=0)

class ASNEConfig(BaseModel):
    agent: AgentConfig = Field(default_factory=AgentConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    quantum: QuantumConfig = Field(default_factory=QuantumConfig)
    ethics: EthicsConfig = Field(default_factory=EthicsConfig)
    api_hub: APIHubConfig = Field(default_factory=APIHubConfig)
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
