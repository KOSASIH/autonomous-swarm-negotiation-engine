"""FastAPI server for ASNE."""
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from asne.agents.swarm import NegotiatorSwarm
from asne.environment.deal_env import DealEnvironment
from asne.ethics.engine import EthicsEngine

class NegotiationRequest(BaseModel):
    deal_type: str = Field(default="procurement")
    parties: int = Field(default=2, ge=2, le=10)
    budget: float = Field(default=1_000_000, gt=0)
    n_agents: int = Field(default=5, ge=1, le=50)
    max_rounds: int = Field(default=100, ge=1, le=1000)
    fairness_threshold: float = Field(default=0.85, ge=0, le=1)

class NegotiationResponse(BaseModel):
    status: str
    deal_value: float | None = None
    rounds_played: int
    fairness_score: float | None = None
    compliant: bool | None = None

app = FastAPI(title="ASNE - Autonomous Swarm Negotiation Engine", version="0.1.0")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "0.1.0"}

@app.post("/negotiate", response_model=NegotiationResponse)
async def run_negotiation(request: NegotiationRequest):
    try:
        ethics = EthicsEngine(fairness_threshold=request.fairness_threshold)
        env = DealEnvironment(deal_type=request.deal_type, parties=request.parties, constraints={"budget": request.budget})
        swarm = NegotiatorSwarm(n_agents=request.n_agents, ethics_engine=ethics)
        result = swarm.negotiate(env, max_rounds=request.max_rounds)
        return NegotiationResponse(
            status=result.status.value,
            deal_value=result.agreement.total_value if result.agreement else None,
            rounds_played=result.rounds_played,
            fairness_score=result.ethics_report.fairness_score if result.ethics_report else None,
            compliant=result.ethics_report.compliant if result.ethics_report else None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    return {"active_negotiations": 0, "total_deals": 0}
