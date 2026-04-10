"""Quantum optimization for deal parameter space."""
from __future__ import annotations
import numpy as np
import structlog
from typing import Any
from asne.core.config import QuantumConfig

logger = structlog.get_logger()

class QuantumDealOptimizer:
    """Hybrid classical-quantum optimizer using QAOA/VQE."""
    def __init__(self, config: QuantumConfig | None = None) -> None:
        self.config = config or QuantumConfig()

    def optimize(self, objective: dict[str, Any], constraints: list[dict[str, Any]], n_iterations: int = 100) -> dict[str, Any]:
        logger.info("quantum_optimization_start", n_constraints=len(constraints), n_iterations=n_iterations)
        if self.config.use_pennylane:
            return self._pennylane_optimize(objective, constraints, n_iterations)
        return self._qiskit_optimize(objective, constraints, n_iterations)

    def _qiskit_optimize(self, objective: dict[str, Any], constraints: list[dict[str, Any]], n_iterations: int) -> dict[str, Any]:
        try:
            from qiskit.circuit import QuantumCircuit
            from qiskit.primitives import StatevectorSampler
            n_qubits = min(self.config.n_qubits, len(constraints) + 2)
            qc = QuantumCircuit(n_qubits)
            for i in range(n_qubits): qc.h(i)
            gamma, beta = np.random.uniform(0, 2 * np.pi), np.random.uniform(0, np.pi)
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1); qc.rz(gamma, i + 1); qc.cx(i, i + 1)
            for i in range(n_qubits): qc.rx(2 * beta, i)
            qc.measure_all()
            sampler = StatevectorSampler()
            result = sampler.run([qc], shots=self.config.shots).result()
            counts = result[0].data.meas.get_counts()
            optimal_bits = max(counts, key=counts.get)
            return {"optimal_params": self._decode_bitstring(optimal_bits, objective), "method": "qaoa_qiskit", "n_qubits": n_qubits}
        except ImportError:
            return self._classical_fallback(objective, constraints, n_iterations)

    def _pennylane_optimize(self, objective: dict[str, Any], constraints: list[dict[str, Any]], n_iterations: int) -> dict[str, Any]:
        try:
            import pennylane as qml
            n_qubits = min(self.config.n_qubits, 8)
            dev = qml.device("default.qubit", wires=n_qubits)
            @qml.qnode(dev)
            def circuit(params):
                for i in range(n_qubits): qml.RY(params[i], wires=i)
                for i in range(n_qubits - 1): qml.CNOT(wires=[i, i + 1])
                for i in range(n_qubits): qml.RZ(params[n_qubits + i], wires=i)
                return qml.expval(qml.PauliZ(0))
            params = np.random.uniform(0, 2 * np.pi, 2 * n_qubits)
            opt = qml.GradientDescentOptimizer(stepsize=0.1)
            for _ in range(min(n_iterations, 50)): params = opt.step(circuit, params)
            return {"optimal_params": self._params_to_deal(params, objective), "method": "vqe_pennylane", "n_qubits": n_qubits}
        except ImportError:
            return self._classical_fallback(objective, constraints, n_iterations)

    def _classical_fallback(self, objective: dict[str, Any], constraints: list[dict[str, Any]], n_iterations: int) -> dict[str, Any]:
        best_params, best_score = {}, float("-inf")
        for _ in range(n_iterations):
            candidate = {k: v * np.random.uniform(0.8, 1.2) for k, v in objective.items() if isinstance(v, (int, float))}
            score = sum(candidate.values())
            if score > best_score: best_score, best_params = score, candidate
        return {"optimal_params": best_params, "method": "classical_random_search", "iterations": n_iterations}

    @staticmethod
    def _decode_bitstring(bitstring: str, objective: dict[str, Any]) -> dict[str, float]:
        bits = [int(b) for b in bitstring]
        keys = [k for k, v in objective.items() if isinstance(v, (int, float))]
        return {key: objective[key] * (1 + (bits[i] * 2 - 1) * 0.1) for i, key in enumerate(keys) if i < len(bits)}

    @staticmethod
    def _params_to_deal(params: np.ndarray, objective: dict[str, Any]) -> dict[str, float]:
        keys = [k for k, v in objective.items() if isinstance(v, (int, float))]
        return {key: float(objective[key] * (0.5 + abs(np.sin(params[i])))) for i, key in enumerate(keys) if i < len(params)}
