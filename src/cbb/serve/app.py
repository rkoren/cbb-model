"""FastAPI win-probability prediction endpoint.

Expects a production XGBoost booster + logistic calibrator loaded from MLflow,
and the KenPom client for live feature lookup during the regular season.

Run locally:
    uvicorn cbb.serve.app:app --reload
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from cbb.kenpom import KenPomClient
from cbb.train.model import ModelConfig, predict

log = logging.getLogger(__name__)

# ── State loaded at startup ────────────────────────────────────────────────────
_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    run_id = os.environ["MLFLOW_RUN_ID"]
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    # Download XGBoost model artifact
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="xgb_model/booster.ubj")
    booster = xgb.Booster()
    booster.load_model(local_path)

    # Load sklearn calibrator
    calibrator = mlflow.sklearn.load_model(f"runs:/{run_id}/calibrator")

    # Temperature params stored as MLflow run metrics
    metrics = run.data.metrics
    temp_params = {
        "T_M_close": metrics["T_M_close"],
        "T_M_blowout": metrics["T_M_blowout"],
        "T_W_close": metrics["T_W_close"],
        "T_W_blowout": metrics["T_W_blowout"],
    }
    vegas_alpha = metrics.get("vegas_alpha", 1.0)

    # Feature list stored as a run tag
    features = run.data.tags["features"].split(",")

    _state.update({
        "booster": booster,
        "calibrator": calibrator,
        "temp_params": temp_params,
        "vegas_alpha": vegas_alpha,
        "features": features,
        "kenpom": KenPomClient(),  # reads KENPOM_API_KEY from env
    })

    log.info("Model loaded from run %s (%d features)", run_id, len(features))
    yield
    _state.clear()


app = FastAPI(title="CBB Win Probability", lifespan=lifespan)


# ── Request / Response ─────────────────────────────────────────────────────────

class MatchupRequest(BaseModel):
    season: int = Field(..., description="Ending year of season (e.g. 2027)")
    team_a_id: int = Field(..., description="KenPom TeamID for team A")
    team_b_id: int = Field(..., description="KenPom TeamID for team B")
    men_women: int = Field(0, description="0=men, 1=women")
    # Optional — enriches prediction when known (e.g. during tournament)
    team_a_seed: int | None = Field(None, description="NCAA tournament seed for team A")
    team_b_seed: int | None = Field(None, description="NCAA tournament seed for team B")
    vegas_prob: float | None = Field(None, description="Market win probability for team A (0-1)")
    # Provide raw features directly if KenPom lookup should be skipped
    features: dict[str, float] | None = Field(None, description="Pre-computed features (overrides KenPom lookup)")


class MatchupResponse(BaseModel):
    team_a_win_prob: float
    team_b_win_prob: float
    model_prob: float          # before Vegas blend
    vegas_prob: float | None


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "features_loaded": len(_state.get("features", []))}


@app.post("/predict", response_model=MatchupResponse)
def predict_matchup(req: MatchupRequest):
    booster = _state["booster"]
    calibrator = _state["calibrator"]
    temp_params = _state["temp_params"]
    vegas_alpha = _state["vegas_alpha"]
    features = _state["features"]
    kenpom: KenPomClient = _state["kenpom"]

    if req.features:
        feat_dict = req.features
    else:
        feat_dict = _fetch_features(kenpom, req)

    seed_gap = None
    if req.team_a_seed is not None and req.team_b_seed is not None:
        seed_gap = abs(req.team_a_seed - req.team_b_seed)
        feat_dict["A_Seed"] = req.team_a_seed
        feat_dict["B_Seed"] = req.team_b_seed
        feat_dict["d_Seed"] = req.team_a_seed - req.team_b_seed

    feat_dict["men_women"] = req.men_women

    # Fill missing features with 0 (same as training fillna)
    row = pd.DataFrame([{f: feat_dict.get(f, 0.0) for f in features}])

    model_prob = predict(
        feature_row=row,
        booster=booster,
        calibrator=calibrator,
        temp_params=temp_params,
        men_women=req.men_women,
        seed_gap=seed_gap,
        vegas_prob=None,   # get model prob before blend
        vegas_alpha=1.0,
    )

    final_prob = predict(
        feature_row=row,
        booster=booster,
        calibrator=calibrator,
        temp_params=temp_params,
        men_women=req.men_women,
        seed_gap=seed_gap,
        vegas_prob=req.vegas_prob,
        vegas_alpha=vegas_alpha,
    )

    return MatchupResponse(
        team_a_win_prob=round(final_prob, 4),
        team_b_win_prob=round(1 - final_prob, 4),
        model_prob=round(model_prob, 4),
        vegas_prob=req.vegas_prob,
    )


# ── KenPom live feature lookup ─────────────────────────────────────────────────

def _fetch_features(kenpom: KenPomClient, req: MatchupRequest) -> dict[str, float]:
    """Pull current-season ratings for both teams and build differential features.

    Uses today's ratings (not archive) for in-season predictions. For pre-tournament
    predictions, callers should provide features directly using the archive endpoint
    at Selection Sunday date to avoid leakage.
    """
    try:
        ra = kenpom.ratings(year=req.season, team_id=req.team_a_id)
        rb = kenpom.ratings(year=req.season, team_id=req.team_b_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"KenPom API error: {e}")

    if ra.empty or rb.empty:
        raise HTTPException(status_code=404, detail="Team not found in KenPom for this season")

    a = ra.iloc[0]
    b = rb.iloc[0]

    kp_fields = {
        "AdjEM": ("AdjEM", "AdjEM"),
        "AdjOE": ("AdjOE", "AdjOE"),
        "AdjDE": ("AdjDE", "AdjDE"),
        "AdjTempo": ("AdjTempo", "AdjTempo"),
    }

    feats: dict[str, float] = {}
    for feat, (col_a, col_b) in kp_fields.items():
        va = float(a.get(col_a, np.nan))
        vb = float(b.get(col_b, np.nan))
        feats[f"A_{feat}"] = va
        feats[f"B_{feat}"] = vb
        feats[f"d_{feat}"] = va - vb

    return feats
