"""Graph Neural Network negotiation agent.

Models negotiations as dynamic graphs where nodes=agents/issues
and edges=negotiation relationships. Uses message passing
for relational reasoning across multi-party deals.
"""
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from asne.agents.base import BaseNegotiator
from asne.core.config import AgentConfig
from asne.core.types import AgentRole, Proposal


class GraphConvLayer(nn.Module):
    """Graph convolutional layer with edge features."""
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 8) -> None:
        super().__init__()
        self.node_transform = nn.Linear(in_dim + edge_dim, out_dim)
        self.edge_transform = nn.Linear(in_dim * 2, edge_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()

    def forward(self, node_features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, N, D = node_features.shape
        src = node_features.unsqueeze(2).expand(-1, -1, N, -1)
        tgt = node_features.unsqueeze(1).expand(-1, N, -1, -1)
        edge_feats = self.edge_transform(torch.cat([src, tgt], dim=-1))  # (B, N, N, edge_dim)
        adj_expanded = adj.unsqueeze(-1)  # (B, N, N, 1)
        aggregated = (edge_feats * adj_expanded).sum(dim=2)  # (B, N, edge_dim)
        combined = torch.cat([node_features, aggregated], dim=-1)  # (B, N, D+edge_dim)
        return self.norm(self.act(self.node_transform(combined)))


class NegotiationGNN(nn.Module):
    """Graph Neural Network for multi-party negotiation modeling."""
    def __init__(self, node_dim: int, action_dim: int, n_layers: int = 4) -> None:
        super().__init__()
        hidden = 128
        self.input_proj = nn.Linear(node_dim, hidden)
        self.gnn_layers = nn.ModuleList([GraphConvLayer(hidden, hidden) for _ in range(n_layers)])
        self.actor = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, action_dim))
        self.critic = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, node_features: torch.Tensor, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(node_features)
        for layer in self.gnn_layers:
            x = x + layer(x, adj)  # residual
        logits = self.actor(x)
        value = self.critic(x.mean(dim=1))
        return logits, value


class GNNNegotiator(BaseNegotiator):
    """GNN-based agent that reasons over negotiation relationship graphs."""
    def __init__(self, node_dim: int, action_dim: int, n_parties: int = 5, config: AgentConfig | None = None, role: AgentRole = AgentRole.BUYER) -> None:
        super().__init__(role=role)
        self.config = config or AgentConfig()
        self.node_dim = node_dim
        self.action_dim = action_dim
        self.n_parties = n_parties
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gnn = NegotiationGNN(node_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.gnn.parameters(), lr=self.config.learning_rate)

    def _build_graph(self, state: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        N = self.n_parties
        node_feats = torch.zeros(1, N, self.node_dim, device=self.device)
        adj = torch.ones(1, N, N, device=self.device)
        adj[:, range(N), range(N)] = 0  # no self-loops
        deal_val = float(state.get("deal_value", 100000))
        node_feats[0, :, 0] = deal_val / 1e6
        if "issues" in state:
            issues = state["issues"][:self.node_dim - 1]
            node_feats[0, :, 1:1 + len(issues)] = torch.tensor(issues, device=self.device).unsqueeze(0).expand(N, -1)
        return node_feats, adj

    def propose(self, state: dict[str, Any]) -> Proposal:
        self._round += 1
        nodes, adj = self._build_graph(state)
        with torch.no_grad():
            logits, _ = self.gnn(nodes, adj)
            probs = F.softmax(logits[0, 0], dim=-1)
            action = torch.multinomial(probs, 1).item()
        base_value = state.get("deal_value", 100000)
        adjustment = (int(action) - self.action_dim // 2) / self.action_dim
        proposed_value = base_value * (1 + adjustment * 0.12)
        return Proposal(agent_id=self.agent_id, round_num=self._round, terms={"value": proposed_value, "gnn_action": int(action)}, value=proposed_value, metadata={"model": "GNN", "n_parties": self.n_parties})

    def respond(self, proposal: Proposal, state: dict[str, Any]) -> bool:
        nodes, adj = self._build_graph(state)
        with torch.no_grad():
            _, value = self.gnn(nodes, adj)
        return proposal.value >= value.item() * 1000

    def learn(self, reward: float, next_state: dict[str, Any]) -> None:
        pass  # Batch training handled by centralized trainer
