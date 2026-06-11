"""Serving contract for kitchen-platform.

Loaded by ``kitchen serve local`` (FastAPI) and the Lambda handler (Mangum).
The ``predict()`` function is the only required export.  ``RequestModel``,
``ResponseModel``, and ``FEATURES`` are optional but give the FastAPI app a
typed schema and auto-generated OpenAPI docs.

Loading strategy
----------------
On first call, ``predict()`` loads the ``CBBModel`` from the MLflow registry
(``KITCHEN_MODEL_NAME`` @ ``KITCHEN_MODEL_VERSION`` or the ``champion`` alias).
Subsequent calls reuse the cached model — no re-loading between requests.

If ``CBB_MODEL_DIR`` is set, ``predictor.py`` loads the local ``cbb_model.pkl``
from that directory instead of hitting the registry.  This is useful for local
dev without a running MLflow server.  (Note: ``KITCHEN_PREDICTOR_DIR`` is
reserved by ``kitchen serve local`` to locate this file — do not reuse it.)

Environment variables
---------------------
``MLFLOW_TRACKING_URI``    — tracking store (default: sqlite:///mlruns.db)
``MLFLOW_MODEL_NAME``      — registered model name (default: cbb-tournament-model)
``KITCHEN_MODEL_VERSION``  — version tag or alias (default: champion)
``CBB_MODEL_DIR``          — if set, load from local pickle instead of registry
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

log = logging.getLogger(__name__)

# ── Lazy model load ────────────────────────────────────────────────────────────

_model: Any | None = None


def _load_model() -> Any:
    """Load CBBModel from local pickle or MLflow registry."""
    local_dir = os.environ.get("CBB_MODEL_DIR")
    if local_dir:
        path = Path(local_dir) / "cbb_model.pkl"
        log.info("Loading CBBModel from local file: %s", path)
        with open(path, "rb") as f:
            return pickle.load(f)

    import mlflow.sklearn
    from kitchen.tracking import configure_from_env

    configure_from_env()
    model_name = os.environ.get("MLFLOW_MODEL_NAME", "cbb-tournament-model")
    version = os.environ.get("KITCHEN_MODEL_VERSION", "champion")
    uri = f"models:/{model_name}@{version}"
    log.info("Loading CBBModel from MLflow registry: %s", uri)
    return mlflow.sklearn.load_model(uri)


def _get_model() -> Any:
    global _model
    if _model is None:
        _model = _load_model()
    return _model


# ── Typed schema (optional — improves OpenAPI docs) ───────────────────────────

class RequestModel(BaseModel):
    """Win-probability request: all features required for a single matchup."""

    Season: int
    men_women: int              # 0 = men's, 1 = women's
    A_TeamID: int
    B_TeamID: int
    A_Seed: float
    B_Seed: float
    d_Seed: float
    d_AdjEM: float = 0.0
    d_AdjOE: float = 0.0
    d_AdjDE: float = 0.0
    d_AdjTempo: float = 0.0
    d_Rating: float = 0.0
    d_Elo: float = 0.0
    d_Quality: float = 0.0
    d_Form: float = 0.0
    d_MasseyRank: float = 0.0
    A_MasseyRank: float = 0.0
    B_MasseyRank: float = 0.0
    d_avg_PointDiff: float = 0.0
    d_avg_Score: float = 0.0
    d_eFGpct: float = 0.0
    d_TOpct: float = 0.0
    d_ORpct: float = 0.0
    d_FTrate: float = 0.0
    d_TSpct: float = 0.0
    d_opp_eFGpct: float = 0.0
    d_opp_TOpct: float = 0.0
    d_opp_ORpct: float = 0.0
    d_opp_FTrate: float = 0.0
    d_opp_TSpct: float = 0.0
    d_FG3pct: float = 0.0
    d_FG2pct: float = 0.0
    d_FTpct: float = 0.0
    d_opp_FG3pct: float = 0.0
    d_opp_FG2pct: float = 0.0
    d_opp_FTpct: float = 0.0
    d_3PArate: float = 0.0
    d_opp_3PArate: float = 0.0
    d_ASTrate: float = 0.0
    d_opp_ASTrate: float = 0.0
    d_Blkpct: float = 0.0
    d_Stlpct: float = 0.0
    d_NonStlTOpct: float = 0.0
    d_opp_NonStlTOpct: float = 0.0
    d_pct_pts_3: float = 0.0
    d_pct_pts_2: float = 0.0
    d_pct_pts_ft: float = 0.0
    d_opp_pct_pts_3: float = 0.0
    d_opp_pct_pts_2: float = 0.0
    d_opp_pct_pts_ft: float = 0.0
    d_quality_wtd_margin: float = 0.0


class ResponseModel(BaseModel):
    """Win probability response."""

    win_probability: float      # P(A_TeamID wins)
    A_TeamID: int
    B_TeamID: int


# Surfaced on GET /metadata so callers know which model is serving.
MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "cbb-tournament-model")
MODEL_VERSION = os.environ.get("KITCHEN_MODEL_VERSION", "champion")

# Feature list used for model.predict_batch — must match CBBModel.features
FEATURES = [
    "men_women",
    "d_AdjEM", "d_AdjOE", "d_AdjDE", "d_AdjTempo",
    "d_Rating",
    "d_Elo", "d_Quality",
    "A_Seed", "B_Seed", "d_Seed",
    "d_Form",
    "d_MasseyRank", "A_MasseyRank", "B_MasseyRank",
    "d_avg_PointDiff", "d_avg_Score",
    "d_eFGpct", "d_TOpct", "d_ORpct", "d_FTrate", "d_TSpct",
    "d_opp_eFGpct", "d_opp_TOpct", "d_opp_ORpct", "d_opp_FTrate", "d_opp_TSpct",
    "d_FG3pct", "d_FG2pct", "d_FTpct",
    "d_opp_FG3pct", "d_opp_FG2pct", "d_opp_FTpct",
    "d_3PArate", "d_opp_3PArate",
    "d_ASTrate", "d_opp_ASTrate",
    "d_Blkpct", "d_Stlpct",
    "d_NonStlTOpct", "d_opp_NonStlTOpct",
    "d_pct_pts_3", "d_pct_pts_2", "d_pct_pts_ft",
    "d_opp_pct_pts_3", "d_opp_pct_pts_2", "d_opp_pct_pts_ft",
    "d_quality_wtd_margin",
]


# ── Entry point ────────────────────────────────────────────────────────────────

def predict(payload: dict) -> dict:
    """Return win probability for a single matchup.

    Args:
        payload: Dict conforming to ``RequestModel`` — all feature columns for
                 one (A_TeamID, B_TeamID) pair.

    Returns:
        Dict with ``win_probability`` (float in [0, 1]), ``A_TeamID``, ``B_TeamID``.
    """
    model = _get_model()
    row = pd.DataFrame([payload])
    probs: np.ndarray = model.predict_batch(row)
    return {
        "win_probability": float(np.clip(probs[0], 0.0, 1.0)),
        "A_TeamID": int(payload.get("A_TeamID", 0)),
        "B_TeamID": int(payload.get("B_TeamID", 0)),
    }
