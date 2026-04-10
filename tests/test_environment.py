"""Tests for deal environment."""
from asne.environment.deal_env import DealEnvironment

def test_environment_reset():
    env = DealEnvironment(deal_type="procurement", constraints={"budget": 500000})
    state = env.reset()
    assert state["deal_value"] == 500000.0
    assert state["round"] == 0
    assert len(state["issues"]) == 5

def test_environment_step():
    env = DealEnvironment(constraints={"budget": 100000})
    env.reset()
    state, reward, done, info = env.step({"proposed_value": 90000})
    assert state["round"] == 1
    assert not done
