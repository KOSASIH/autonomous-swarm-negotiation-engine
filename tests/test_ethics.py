"""Tests for ethics engine."""
from asne.ethics.engine import EthicsEngine
from asne.ethics.transparency import TransparencyLog
from asne.core.types import Proposal

def test_ethics_engine():
    engine = EthicsEngine(fairness_threshold=0.85)
    proposal = Proposal(agent_id="a1", round_num=1, terms={}, value=100000)
    assert engine.evaluate_proposal(proposal) is True

def test_transparency_log():
    log = TransparencyLog()
    log.log_event("test", {"key": "value"})
    log.log_event("test2", {"key2": "value2"})
    assert log.size == 2
    assert log.verify_integrity() is True

def test_transparency_integrity():
    log = TransparencyLog()
    for i in range(10):
        log.log_event(f"event_{i}", {"i": i})
    assert log.verify_integrity() is True
