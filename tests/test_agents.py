"""Tests for negotiation agents."""
from asne.agents.dqn import DQNNegotiator
from asne.agents.swarm import NegotiatorSwarm
from asne.core.config import AgentConfig
from asne.core.types import AgentRole

def test_dqn_agent_creation():
    agent = DQNNegotiator(state_dim=16, action_dim=5)
    assert agent.role == AgentRole.BUYER
    assert agent.steps == 0

def test_dqn_propose():
    agent = DQNNegotiator(state_dim=16, action_dim=5)
    state = {"deal_value": 100000, "round": 1}
    proposal = agent.propose(state)
    assert proposal.agent_id == agent.agent_id
    assert proposal.value > 0

def test_swarm_creation():
    config = AgentConfig(n_agents=3)
    swarm = NegotiatorSwarm(n_agents=3, config=config)
    assert len(swarm.agents) == 3

def test_swarm_negotiate():
    from asne.environment.deal_env import DealEnvironment
    env = DealEnvironment(constraints={"budget": 100000})
    swarm = NegotiatorSwarm(n_agents=3)
    result = swarm.negotiate(env, max_rounds=10)
    assert result.rounds_played > 0
    assert result.status is not None
