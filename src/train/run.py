"""CBB model training — kitchen-platform adapter.

Called by ``kitchen run train`` (via ``kitchen.flows.train_flow``) as::

    from src.train.run import train
    train(params, DataStore(), tracker)

The function runs the full LOTO training pipeline and logs all artifacts to
MLflow. The trained ``CBBModel`` (booster + calibrator + temp params + Vegas
alpha) is saved both locally (``data/processed/``) and to the MLflow run.
"""

from __future__ import annotations

import json
import logging
import pickle

import mlflow
import mlflow.sklearn

from kitchen.store import DataStore
from kitchen.tracking import Tracker

from cbb.train.model import (
    CBBModel,
    ModelConfig,
    train_loto,
    train_production,
)

log = logging.getLogger(__name__)

# Feature columns passed to XGBoost — filtered at runtime to those present in
# the matchup DataFrame.  Keeping this list here (not in params.yaml alone)
# makes the module self-documenting and avoids silent omissions if params.yaml
# is edited without updating the feature list.
FEATURE_CANDIDATES: list[str] = [
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


def _build_model_config(params: dict) -> ModelConfig:
    """Translate params.yaml model section into a ModelConfig."""
    mp = params.get("model", {})
    xgb_overrides = {
        k: v for k, v in mp.items()
        if k in {"eta", "subsample", "colsample_bynode", "max_depth",
                 "num_parallel_tree", "min_child_weight", "max_bin"}
    }
    config = ModelConfig(model_variant=mp.get("variant", "baseline"))
    if xgb_overrides:
        config.xgb_params.update(xgb_overrides)
    if "num_rounds" in mp:
        config.num_rounds = int(mp["num_rounds"])
    if "seed_gap_men" in mp:
        config.men_blowout_gap = int(mp["seed_gap_men"])
    if "seed_gap_women" in mp:
        config.women_blowout_gap = int(mp["seed_gap_women"])
    if "calibrator_C" in mp:
        config.logistic_C = float(mp["calibrator_C"])
    return config


def train(params: dict, store: DataStore, tracker: Tracker) -> CBBModel:
    """Run the full CBB training pipeline and return the production CBBModel.

    Steps
    -----
    1. Load ``matchups.parquet`` built by ``kitchen run features``.
    2. Determine XGBoost feature list (candidates filtered to columns present).
    3. Run LOTO cross-validation (``train_loto``) — logs metrics to MLflow.
    4. Train production model on all seasons (``train_production``).
    5. Wrap into ``CBBModel`` and save locally + log to MLflow.

    Args:
        params: Parsed ``params.yaml`` dict.
        store: ``DataStore`` rooted at the project directory.
        tracker: Configured ``Tracker`` — its experiment is already set before
                 this function is called by ``kitchen.flows.train_flow``.

    Returns:
        The trained ``CBBModel`` (also saved to ``data/processed/cbb_model.pkl``
        and logged as an MLflow sklearn artifact).
    """
    feature_candidates: list[str] = params.get("feature_candidates", FEATURE_CANDIDATES)

    # Load processed matchup dataset
    proc = store.processed_dir
    matchups = store.load_parquet("matchups.parquet")
    features = [f for f in feature_candidates if f in matchups.columns]
    matchups[features] = matchups[features].fillna(0)
    log.info("Training on %d matchups  %d features", len(matchups), len(features))

    config = _build_model_config(params)

    # ── LOTO training — run without its own MLflow run so metrics land on the
    # outer tracker.run() context started by train_flow.py.
    loto = train_loto(
        matchups,
        features,
        config=config,
        mlflow_experiment=None,
    )
    log.info("LOTO Brier: %.6f", loto.overall_brier)

    # ── Log LOTO metrics to the active (outer) run ────────────────────────────
    mlflow.log_metric("loto_brier", loto.overall_brier)
    mlflow.log_metric("vegas_alpha", loto.vegas_alpha)
    for k, v in loto.temp_params.items():
        mlflow.log_metric(k, v)
    for season, brier in loto.brier_by_season.items():
        mlflow.log_metric(f"brier_{season}", brier)
    mlflow.set_tag("model_variant", config.model_variant)
    mlflow.log_params({
        "num_rounds": config.num_rounds,
        "num_features": len(features),
        "num_seasons": len(sorted(matchups["Season"].unique())),
        **{f"xgb_{k}": v for k, v in config.xgb_params.items()},
    })

    # ── Production model (all seasons) ────────────────────────────────────────
    booster, calibrator = train_production(matchups, features, loto.temp_params, config)
    model = loto.to_model(booster, calibrator)

    # ── Persist locally ───────────────────────────────────────────────────────
    proc.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(proc / "prod_booster.ubj"))
    with open(proc / "prod_calibrator.pkl", "wb") as f:
        pickle.dump(calibrator, f)
    with open(proc / "cbb_model.pkl", "wb") as f:
        pickle.dump(model, f)

    loto_meta = {
        "features": loto.features,
        "temp_params": loto.temp_params,
        "vegas_alpha": loto.vegas_alpha,
    }
    loto_meta_path = proc / "loto_meta.json"
    with open(loto_meta_path, "w", encoding="utf-8") as f:
        json.dump(loto_meta, f)

    # ── Log production artifacts to the active (outer) run ────────────────────
    mlflow.sklearn.log_model(model, "cbb_model")
    mlflow.sklearn.log_model(calibrator, "calibrator")
    mlflow.log_artifact(str(proc / "prod_booster.ubj"), "xgb_model")
    mlflow.log_artifact(str(proc / "rating_meta.json"), "run_meta")
    mlflow.log_artifact(str(proc / "scaler.pkl"), "run_meta")
    mlflow.log_artifact(str(loto_meta_path), "run_meta")

    return model
