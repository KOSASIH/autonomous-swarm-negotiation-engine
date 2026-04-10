"""Tests for core configuration."""
from asne.core.config import ASNEConfig, AgentConfig, QuantumConfig, EthicsConfig

def test_default_config():
    config = ASNEConfig()
    assert config.agent.n_agents == 5
    assert config.quantum.n_qubits == 8
    assert config.ethics.fairness_threshold == 0.85

def test_agent_config_custom():
    config = AgentConfig(n_agents=10, learning_rate=1e-3)
    assert config.n_agents == 10
    assert config.learning_rate == 1e-3

def test_quantum_config():
    config = QuantumConfig(use_pennylane=True, n_qubits=4)
    assert config.use_pennylane is True
    assert config.n_qubits == 4

def test_ethics_config():
    config = EthicsConfig(fairness_threshold=0.9)
    assert config.fairness_threshold == 0.9
