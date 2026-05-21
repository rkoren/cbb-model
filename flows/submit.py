"""Generate a Kaggle submission from a specific MLflow training run.

Loads model artifacts from a prior training run (by run_id) and generates a
fresh submission CSV without re-training. Submission is optionally validated,
logged to the original MLflow run, and uploaded to Kaggle.

Usage:
    python flows/submit.py --run-id <mlflow_run_id>
    python flows/submit.py --run-id <mlflow_run_id> --competition march-machine-learning-mania-2026
    python flows/submit.py --run-id <mlflow_run_id> --season 2027
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
import types
from datetime import date
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from prefect import flow, get_run_logger

# Make experiments/ importable so we can reuse the shared task/helper functions
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

from kitchen import tracking
from kitchen.submit import log_submission

from baseline import (
    DATA_PROC,
    DATA_RAW,
    EXPERIMENT,
    _run_submission,
    build_symmetric_games,
    load_kaggle_data,
)

load_dotenv()


def _challenger_hook(pred_df: pd.DataFrame) -> pd.DataFrame:
    pred_df = pred_df.copy()
    pred_df["tempo_seed_interaction"] = pred_df["d_AdjTempo"] * pred_df["d_Seed"]
    pred_df["d_NetReboundEdge"] = pred_df["d_ORpct"] - pred_df["d_opp_ORpct"]
    for col in [
        "round_num",
        "A_path_quality", "B_path_quality", "d_path_quality",
        "A_prior_margin", "B_prior_margin", "d_prior_margin",
    ]:
        if col not in pred_df.columns:
            pred_df[col] = 0.0
    return pred_df


@flow(name="cbb-submit")
def cbb_submit_flow(
    run_id: str,
    season: int | None = None,
    competition: str | None = None,
    message: str = "",
    fetch_lb_score: bool = False,
) -> None:
    """Generate a submission from a specific MLflow training run.

    Downloads model artifacts logged by the training pipeline (booster,
    calibrator, scaler, rating_meta, feature list) and generates predictions
    for all C(68,2) seeded matchup pairs. Optionally uploads to Kaggle and
    logs the public leaderboard score back to the original MLflow run.

    Args:
        run_id: MLflow run ID to load artifacts from (printed at end of training run).
        season: Season year for predictions (e.g. 2026). Defaults to current year.
        competition: Kaggle competition slug. When set, uploads the submission.
        message: Message shown on the Kaggle leaderboard.
        fetch_lb_score: Poll for the public LB score after uploading.
    """
    log = get_run_logger()
    season = season or date.today().year

    tracking.configure_from_env()
    tracking.init_experiment(EXPERIMENT)

    log.info("Loading artifacts from MLflow run %s", run_id)

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    variant = run.data.tags.get("model_variant", "baseline")
    log.info("Model variant: %s  season: %d", variant, season)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        run_meta_dir = Path(mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="run_meta", dst_path=tmp,
        ))
        booster_dir = Path(mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="xgb_model", dst_path=tmp,
        ))

        with open(run_meta_dir / "loto_meta.json") as f:
            loto_meta = json.load(f)
        with open(run_meta_dir / "rating_meta.json") as f:
            rating_meta = json.load(f)
        with open(run_meta_dir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        calibrator = mlflow.sklearn.load_model(f"runs:/{run_id}/calibrator")

        booster = xgb.Booster()
        booster.load_model(str(booster_dir / "prod_booster.ubj"))

        loto_ns = types.SimpleNamespace(
            features=loto_meta["features"],
            temp_params=loto_meta["temp_params"],
            vegas_alpha=loto_meta["vegas_alpha"],
        )
        log.info("Artifacts loaded — %d features", len(loto_ns.features))

        data = load_kaggle_data(season=season)
        reg_sym, _ = build_symmetric_games(data)

        hook = _challenger_hook if variant == "challenger" else None
        sub = _run_submission(
            season, data, reg_sym, booster, calibrator,
            loto_ns, rating_meta, scaler,
            pre_predict_hook=hook,
        )

    sub_path = DATA_PROC / f"submission_{season}.csv"
    sample_path = DATA_RAW / "SampleSubmissionStage1.csv"

    if not sample_path.exists():
        log.warning("SampleSubmissionStage1.csv not found — skipping validation and MLflow logging")
        log.info("Submission saved (%d rows) → %s", len(sub), sub_path)
        return

    sample = pd.read_csv(sample_path)
    with mlflow.start_run(run_id=run_id):
        result = log_submission(
            submission=sub,
            sample=sample,
            file_path=sub_path,
            id_col="ID",
            target_col="Pred",
            competition=competition,
            message=message,
            fetch_lb_score=fetch_lb_score,
        )

    if "lb_score" in result:
        log.info("Leaderboard score: %.6f", result["lb_score"])
    log.info("Submission saved (%d rows) → %s", len(sub), sub_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CBB submission from a prior training run")
    parser.add_argument("--run-id", required=True, help="MLflow run ID (printed at end of training)")
    parser.add_argument("--season", type=int, default=None, help="Season year (default: current year)")
    parser.add_argument("--competition", default=None, help="Kaggle slug — omit to skip upload")
    parser.add_argument("--message", default="", help="Leaderboard message")
    parser.add_argument("--fetch-lb-score", action="store_true", help="Poll for public LB score after upload")
    args = parser.parse_args()

    cbb_submit_flow(
        run_id=args.run_id,
        season=args.season,
        competition=args.competition,
        message=args.message,
        fetch_lb_score=args.fetch_lb_score,
    )
