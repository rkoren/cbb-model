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

from cbb.holdout import HOLDOUT_PARQUET, score_holdout
from cbb.train.model import (
    CBBModel,
    ModelConfig,
    log_sklearn_model,
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


def _train_reg_season(params: dict, store: DataStore) -> object:
    """Train the parallel regular-season game-level model (GM-001b, ``model.target=reg_season``).

    A *separate* model from the tournament champion: it loads ``reg_games.parquet`` (built by the
    features stage), runs LOTO-by-season CV + a 2026 holdout, and logs every metric with a
    ``_reg`` suffix so it never collides with the tournament model's same-named metrics in the
    shared experiment. Register its own champion with::

        kitchen run train --variant reg_season --auto-promote \\
            --promote-metric loto_brier_reg --lower-is-better --model-name cbb-reg-model
    """
    from cbb.train.reg_model import RegConfig, score_reg_holdout, train_reg_loto

    # The platform's `holdout:` config (CBB-017) is the *tournament* holdout (holdout_2026.parquet,
    # tournament matchup features) — inapplicable to a RegModel (reg-game features). train_flow scores
    # it from this same params dict after train() returns, so pop it here to disable that scoring for
    # reg_season runs; the reg model scores its own holdout below (holdout_*_reg). See CBB-025.
    params.pop("holdout", None)

    games = store.load_parquet("reg_games.parquet")
    mp = params.get("model", {})
    config = RegConfig()
    if "num_rounds" in mp:
        config.num_rounds = int(mp["num_rounds"])
    if "calibrator_C" in mp:
        config.logistic_C = float(mp["calibrator_C"])

    log.info("Reg-season training on %d game-rows (%d seasons)",
             len(games), games["Season"].nunique())
    result = train_reg_loto(games, config)
    log.info("Reg LOTO: brier %.6f  margin_MAE %.3f  total_MAE %.3f",
             result.metrics["loto_brier_reg"], result.metrics["loto_margin_mae_reg"],
             result.metrics["loto_total_mae_reg"])

    for k, v in result.metrics.items():
        mlflow.log_metric(k, v)
    for season, brier in result.brier_by_season.items():
        mlflow.log_metric(f"brier_reg_{season}", brier)
    mlflow.set_tag("model_variant", "reg_season")
    mlflow.log_params({
        "model_target": "reg_season",
        "num_rounds_reg": config.num_rounds,
        "num_margin_features": len(result.model.margin_features),
        "num_total_features": len(result.model.total_features),
    })

    hscore = score_reg_holdout(result.model, games)
    for k, v in hscore.items():
        mlflow.log_metric(k, v)
    if "holdout_brier_reg" in hscore:
        log.info("Reg holdout 2026: brier %.6f  margin_MAE %.3f  total_MAE %.3f over %d games",
                 hscore["holdout_brier_reg"], hscore["holdout_margin_mae_reg"],
                 hscore["holdout_total_mae_reg"], int(hscore["holdout_n_games_reg"]))
    if "holdout_brier_reg_w" in hscore:
        log.info("Reg holdout 2026 (women): brier %.6f  margin_MAE %.3f over %d games",
                 hscore["holdout_brier_reg_w"], hscore["holdout_margin_mae_reg_w"],
                 int(hscore["holdout_n_games_reg_w"]))

    proc = store.processed_dir
    proc.mkdir(parents=True, exist_ok=True)
    with open(proc / "reg_model.pkl", "wb") as f:
        pickle.dump(result.model, f)
    log_sklearn_model(result.model, "cbb_model")  # same artifact path → registry/evaluate load it
    return result.model


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
    # GM-001b: `model.target=reg_season` (set by the `reg_season` menu variant) trains the
    # parallel regular-season model instead of the tournament one. The platform is single-model
    # per project (one hard-coded `train` stage), so a target switch is the seam — see CBB-020.
    if params.get("model", {}).get("target", "tournament") == "reg_season":
        return _train_reg_season(params, store)

    feature_candidates: list[str] = params.get("feature_candidates", FEATURE_CANDIDATES)

    # Load processed matchup dataset
    proc = store.processed_dir
    matchups = store.load_parquet("matchups.parquet")
    # The `kenpom_rich` variant adds the d_kp_* features to feature_candidates via the
    # menu `variants:` overlay (CBB-016) — run it with `kitchen run train --variant kenpom_rich`.
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
    # Total head (SC-001) trains on the sum (level) features; margin head uses `features`.
    total_features = [c for c in matchups.columns if c.startswith("s_")]
    if total_features and "men_women" in matchups.columns:
        total_features = ["men_women"] + total_features
    booster, calibrator, total_booster = train_production(
        matchups, features, loto.temp_params, config, total_features=total_features
    )
    model = loto.to_model(booster, calibrator, total_booster, total_features=total_features)

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

    # ── 2026 holdout: trusted generalization metric, distinct from CV loto_brier ──
    # Scores this ≤2025-trained model on the real 2026 tournament (built by the features
    # stage when results exist). No-op until 2026 results are provided. Iterate on
    # loto_brier; check holdout_brier sparingly (it overfits by selection if you peek often).
    holdout_path = proc / HOLDOUT_PARQUET
    if holdout_path.exists():
        try:
            hscore = score_holdout(model, store.load_parquet(HOLDOUT_PARQUET), features)
            for k, v in hscore.items():
                mlflow.log_metric(k, v)
            extra = ""
            if "holdout_margin_mae" in hscore:
                extra = "  margin_MAE %.2f  total_MAE %.2f (%d scored)" % (
                    hscore["holdout_margin_mae"], hscore["holdout_total_mae"], hscore["holdout_scored_games"],
                )
            log.info(
                "Holdout 2026 Brier %.6f  ECE %.4f over %d games%s",
                hscore["holdout_brier"], hscore["holdout_ece"], hscore["holdout_n_games"], extra,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Holdout scoring skipped: %s", exc)
    else:
        log.info("No %s — holdout_brier not logged (add 2026 results to enable)", HOLDOUT_PARQUET)

    # ── Log production artifacts to the active (outer) run ────────────────────
    log_sklearn_model(model, "cbb_model")
    log_sklearn_model(calibrator, "calibrator")
    mlflow.log_artifact(str(proc / "prod_booster.ubj"), "xgb_model")
    mlflow.log_artifact(str(proc / "rating_meta.json"), "run_meta")
    mlflow.log_artifact(str(proc / "scaler.pkl"), "run_meta")
    mlflow.log_artifact(str(loto_meta_path), "run_meta")

    return model
