"""FastAPI handicapper endpoint — v11 reg model, full spread/total/moneyline output (M4 / Stage 2).

Replaces the old tournament win-probability endpoint. Loads the KenPom-independent v11 reg model and
the cached as-of ratings once at startup, then scores any men's matchup on demand. No KenPom key
needed (v11 is self-computed).

    uvicorn cbb.serve.app:app --reload
    curl -s localhost:8000/predict -H 'content-type: application/json' \\
      -d '{"team_a":"Duke","team_b":"North Carolina","date":"2027-02-08","venue":"home_b"}' | jq
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from cbb.serve.handicap import handicap_matchup, load_engine

log = logging.getLogger(__name__)
_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["engine"] = load_engine()
    log.info("handicap engine loaded (v11 reg model + as-of ratings)")
    yield
    _state.clear()


app = FastAPI(title="CBB Handicapper", version="2.0", lifespan=lifespan)


class MatchupRequest(BaseModel):
    team_a: str = Field(..., description="Team name or Kaggle TeamID")
    team_b: str = Field(..., description="Team name or Kaggle TeamID")
    date: str = Field(..., description="Game date, YYYY-MM-DD")
    venue: str = Field("home_a", description="home_a (a hosts) | home_b (b hosts) | neutral")


@app.get("/health")
def health():
    eng = _state.get("engine")
    return {"status": "ok" if eng else "loading",
            "model": "cbb-reg-model v11" if eng else None,
            "live_games": int((eng.mreg.Season == eng.mreg.Season.max()).sum()) if eng else 0}


@app.post("/predict")
def predict(req: MatchupRequest) -> dict:
    """Return the full handicap for a matchup: predicted score, spread, total, win prob, moneyline."""
    eng = _state.get("engine")
    if eng is None:
        raise HTTPException(503, "engine still loading")
    try:
        return handicap_matchup(eng, req.team_a, req.team_b, req.date, req.venue)
    except KeyError as e:
        raise HTTPException(404, detail=str(e))
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
