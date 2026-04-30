"""Challenger pipeline: baseline + tournament path features + derived interactions.

Adds three feature groups on top of the baseline (experiments/baseline.py):
  1. Tournament path features — round number, quality of opponents beaten, prior margins
  2. Derived features — tempo×seed interaction, net rebounding edge

Both experiments log to the same MLflow experiment (cbb-tournament) under
model_variant=challenger vs model_variant=baseline, making them directly
comparable in the MLflow UI. Use flows/promote.py to promote the winner.

Run locally (from repo root):
    python experiments/challenger.py

Schedule via Prefect UI:
    prefect deployment build experiments/challenger.py:challenger_pipeline -n prod-challenger
"""

import os
import sys
from datetime import date
from pathlib import Path

# Make sibling experiment files importable when run as a script
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from dotenv import load_dotenv
from prefect import flow, task, get_run_logger

from kitchen import tracking
from kitchen.experiment import ExperimentConfig

# Re-use all shared tasks from the baseline experiment — no duplication
from baseline import (
    DATA_PROC,
    DATA_RAW,
    EXPERIMENT,
    _run_submission,
    build_symmetric_games,
    compute_all_features,
    ingest_all_kenpom_seasons,
    load_kaggle_data,
    log_eval_summary,
    run_production_training,
    run_training,
)
from cbb.features import compute_path_features
from cbb.train.model import ModelConfig


@task
def generate_submission(
    season: int,
    data: dict,
    reg_sym,
    booster,
    calibrator,
    loto,
    rating_meta: dict,
    scaler,
) -> object:
    """Challenger submission: baseline features + derived interactions + zeroed path features."""
    log = get_run_logger()

    def _challenger_hook(pred_df):
        pred_df = pred_df.copy()
        pred_df["tempo_seed_interaction"] = pred_df["d_AdjTempo"] * pred_df["d_Seed"]
        pred_df["d_NetReboundEdge"] = pred_df["d_ORpct"] - pred_df["d_opp_ORpct"]
        # Path features are zero pre-tournament (no prior rounds played)
        for col in ["round_num", "A_path_quality", "B_path_quality", "d_path_quality",
                    "A_prior_margin", "B_prior_margin", "d_prior_margin"]:
            if col not in pred_df.columns:
                pred_df[col] = 0.0
        return pred_df

    sub = _run_submission(
        season, data, reg_sym, booster, calibrator, loto, rating_meta, scaler,
        pre_predict_hook=_challenger_hook,
    )
    log.info("Challenger submission saved (%d rows) → %s", len(sub), DATA_PROC / f"submission_{season}.csv")
    return sub

load_dotenv()


@flow(name="cbb-challenger-pipeline")
def challenger_pipeline(
    season: int | None = None,
    kenpom_snapshot_date: str | None = None,
    holdout_season: int = 2026,
    xgb_params: dict | None = None,
    num_rounds: int | None = None,
):
    """Challenger pipeline: baseline features + path features + derived interactions.

    Tournament path features add round-by-round context that the baseline lacks:
    a 15-seed who beat a 2-seed looks very different in round 2 than round 1.
    These features are computable pre-tournament and leak-free.

    Args:
        season: Season year (e.g. 2027). Defaults to current year.
        kenpom_snapshot_date: Selection Sunday date (YYYY-MM-DD) for KenPom snapshot.
        holdout_season: Season to highlight in eval summary (default: 2026).
        xgb_params: Override XGBoost params for experiment runs.
        num_rounds: Override boosting rounds for experiment runs.
    """
    log = get_run_logger()
    season = season or date.today().year
    log.info("Running CBB challenger pipeline for season=%d", season)

    tracking.configure_from_env()
    tracking.init_experiment(EXPERIMENT)

    config = ModelConfig(model_variant="challenger")
    if xgb_params:
        config.xgb_params.update(xgb_params)
    if num_rounds is not None:
        config.num_rounds = num_rounds

    exp_config = ExperimentConfig(
        name=f"cbb-challenger-{season}",
        params={**config.xgb_params, "num_rounds": config.num_rounds},
        description="Baseline + tournament path features + tempo×seed + net rebound edge",
    )

    data = load_kaggle_data(season=season)

    try:
        all_seasons = sorted(data["M_reg_raw"]["Season"].unique().tolist())
        ingest_all_kenpom_seasons(seasons=all_seasons, snapshot_date=kenpom_snapshot_date)
    except Exception as e:
        log.warning("KenPom ingestion skipped: %s", e)

    reg_sym, tourn_sym = build_symmetric_games(data)
    matchups, rating_meta, artifacts = compute_all_features(reg_sym, tourn_sym, data)

    # ── Challenger-specific: tournament path features ──────────────────────────
    # adj_eff is re-derived inside compute_all_features; we need it for path quality.
    # Re-compute it cheaply from the cached parquet if available, else from scratch.
    # Since compute_all_features already ran, efficiency is in the matchups' adj_eff
    # frame. We access it by loading from the processed parquet if available, otherwise
    # compute fresh (same result, just extra work).
    adj_eff_path = DATA_PROC / "adj_eff.parquet"
    if adj_eff_path.exists():
        adj_eff = pd.read_parquet(adj_eff_path)
    else:
        from cbb.features import compute_adj_efficiency
        adj_eff = compute_adj_efficiency(reg_sym)

    log.info("Computing tournament path features...")
    path_df = compute_path_features(tourn_sym, adj_eff)
    matchups = matchups.merge(
        path_df,
        on=["Season", "men_women", "A_TeamID", "B_TeamID"],
        how="left",
    )
    log.info(
        "Path features joined — round_num coverage: %.1f%%",
        100 * matchups["round_num"].notna().mean(),
    )

    # ── Challenger-specific: derived interaction features ──────────────────────
    matchups["tempo_seed_interaction"] = matchups["d_AdjTempo"] * matchups["d_Seed"]
    matchups["d_NetReboundEdge"] = matchups["d_ORpct"] - matchups["d_opp_ORpct"]

    # ── Feature list: baseline candidates + challenger additions ──────────────
    feature_candidates = [
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
        # ── Challenger additions ───────────────────────────────────────────────
        "round_num",
        "A_path_quality", "B_path_quality", "d_path_quality",
        "A_prior_margin", "B_prior_margin", "d_prior_margin",
        "tempo_seed_interaction",
        "d_NetReboundEdge",
    ]
    features = [f for f in feature_candidates if f in matchups.columns]
    matchups[features] = matchups[features].fillna(0)

    log.info(
        "Challenger using %d features (%d more than baseline)",
        len(features),
        len(features) - len([f for f in feature_candidates
                             if f in matchups.columns and f not in [
                                 "round_num", "A_path_quality", "B_path_quality",
                                 "d_path_quality", "A_prior_margin", "B_prior_margin",
                                 "d_prior_margin", "tempo_seed_interaction", "d_NetReboundEdge",
                             ]]),
    )

    loto = run_training(matchups, features, config, exp_config)
    log_eval_summary(loto, holdout_season)
    booster, calibrator = run_production_training(matchups, features, loto)
    generate_submission(season, data, reg_sym, booster, calibrator, loto, rating_meta, artifacts["scaler"])

    log.info("Challenger pipeline complete. LOTO Brier: %.6f", loto.overall_brier)


if __name__ == "__main__":
    challenger_pipeline()
