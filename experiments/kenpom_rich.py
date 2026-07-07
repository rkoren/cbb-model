"""KenPom-rich pipeline: baseline + novel, leak-safe KenPom team features.

Adds one feature group on top of the baseline (experiments/baseline.py): KenPom team
attributes that the manually-computed season_avgs genuinely lacks and that don't leak the
tournament label —

  - height endpoint: kp_AvgHgt, kp_HgtEff (roster-based, leak-free; the minutes-weighted
    kp_Exp/kp_Bench/kp_Continuity were dropped in KP-006 as a backdoor tournament-label leak)
  - ratings extras already cached but unused: kp_SOS, kp_Luck, kp_APL_Off, kp_APL_Def

Deliberately excludes four-factors / misc-stats / pointdist: those endpoints take only
y=year (no as-of-date), so past-season pulls include tournament games (backdoor leakage),
and they largely duplicate season_avgs. They return in Phase 2 via leak-free as-of-date
sourcing (regular-season game-level handicapper).

Like challenger, this logs to the same MLflow experiment (cbb-tournament) under
model_variant=kenpom_rich, directly comparable to baseline and challenger in the MLflow UI.

Run locally (from repo root):
    python experiments/kenpom_rich.py

Note: requires KENPOM_API_KEY for the height fetch + team-name map. Without it the KenPom
join is skipped and the run falls back to baseline features (logged as a warning).
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
from kitchen.submit import log_submission

# Re-use all shared tasks from the baseline experiment — no duplication
from baseline import (
    DATA_PROC,
    DATA_RAW,
    EXPERIMENT,
    _store,
    _run_submission,
    build_symmetric_games,
    compute_all_features,
    ingest_all_kenpom_seasons,
    load_kaggle_data,
    log_eval_summary,
    run_production_training,
    run_training,
)
from cbb.kenpom import KenPomClient, build_kenpom_rich_features, join_kenpom_rich
from cbb.kenpom.features import build_team_name_map
from cbb.train.model import ModelConfig

load_dotenv()


@task
def ingest_kenpom_height(seasons: list[int]) -> None:
    """Fetch and cache the KenPom height endpoint per season, skipping cached ones.

    Mirrors baseline.ingest_all_kenpom_seasons' skip-when-cached behaviour so re-runs are
    cheap. Height is roster-shaped (leak-safe), so no archive snapshot date is needed.
    """
    log = get_run_logger()
    client = KenPomClient()
    for s in seasons:
        out = DATA_PROC / f"kenpom_height_{s}.parquet"
        if out.exists():
            log.info("KenPom height for %d already cached — skipping", s)
            continue
        try:
            df = client.height(year=s)
            df.to_parquet(out, index=False)
            log.info("Fetched KenPom height for %d (%d teams)", s, len(df))
        except Exception as e:
            log.warning("KenPom height fetch failed for season %d: %s", s, e)


@task
def build_rich_feature_frame(seasons: list[int], data: dict) -> pd.DataFrame:
    """Assemble the KenPom rich-feature frame (kp_*) across seasons, keyed to Kaggle TeamIDs.

    Reads the already-cached kenpom_ratings_{s}.parquet (for SOS/Luck/APL) and
    kenpom_height_{s}.parquet (for height/experience/bench/continuity). Returns an empty
    frame if no cached source exists for any season (e.g. no API key) — the caller then
    runs with baseline features only.
    """
    log = get_run_logger()
    frames: list[pd.DataFrame] = []
    for s in seasons:
        ratings_path = _store.processed_dir / f"kenpom_ratings_{s}.parquet"
        height_path = DATA_PROC / f"kenpom_height_{s}.parquet"
        if not ratings_path.exists() and not height_path.exists():
            continue
        try:
            team_map = build_team_name_map(data["M_teams"], KenPomClient().teams(year=s))
            ratings_df = pd.read_parquet(ratings_path) if ratings_path.exists() else None
            height_df = pd.read_parquet(height_path) if height_path.exists() else None
            frames.append(build_kenpom_rich_features(s, team_map, ratings_df, height_df))
        except Exception as e:
            log.warning("KenPom rich features failed for season %d: %s", s, e)

    if not frames:
        log.warning("No KenPom rich features available — falling back to baseline features")
        return pd.DataFrame(columns=["Season", "men_women", "TeamID"])

    kp = pd.concat(frames, ignore_index=True)
    log.info("KenPom rich frame: %d team-seasons", len(kp))
    return kp


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
    kp_feats: pd.DataFrame,
    competition: str | None = None,
    message: str = "",
    fetch_lb_score: bool = False,
) -> object:
    """KenPom-rich submission: baseline features + kp_ team features joined onto pred pairs."""
    log = get_run_logger()

    def _kenpom_hook(pred_df):
        joined, added = join_kenpom_rich(pred_df, kp_feats)
        # Women's / unmatched teams have no KenPom row — neutral-fill so parity holds
        for col in added:
            joined[col] = joined[col].fillna(0.0)
        return joined

    sub = _run_submission(
        season, data, reg_sym, booster, calibrator, loto, rating_meta, scaler,
        pre_predict_hook=_kenpom_hook,
    )
    sub_path = DATA_PROC / f"submission_{season}.csv"
    sample_path = DATA_RAW / "SampleSubmissionStage1.csv"

    if sample_path.exists():
        sample = pd.read_csv(sample_path)
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
    else:
        log.warning("SampleSubmissionStage1.csv not found — skipping validation and MLflow artifact logging")

    log.info("KenPom-rich submission saved (%d rows) → %s", len(sub), sub_path)
    return sub


@flow(name="cbb-kenpom-rich-pipeline")
def kenpom_rich_pipeline(
    season: int | None = None,
    kenpom_snapshot_date: str | None = None,
    holdout_season: int = 2026,
    xgb_params: dict | None = None,
    num_rounds: int | None = None,
    generate_sub: bool = False,
    competition: str | None = None,
    submission_message: str = "",
    fetch_lb_score: bool = False,
):
    """KenPom-rich pipeline: baseline features + novel leak-safe KenPom team features.

    Args:
        season: Season year (e.g. 2027). Defaults to current year.
        kenpom_snapshot_date: Selection Sunday date (YYYY-MM-DD) for the KenPom ratings snapshot.
        holdout_season: Season to highlight in eval summary (default: 2026).
        xgb_params: Override XGBoost params for experiment runs.
        num_rounds: Override boosting rounds for experiment runs.
        generate_sub: Generate a submission CSV after training (default False).
        competition: Kaggle competition slug. When set, uploads the submission.
        submission_message: Message shown on the Kaggle leaderboard.
        fetch_lb_score: Poll for the public LB score after uploading.
    """
    log = get_run_logger()
    season = season or date.today().year
    log.info("Running CBB KenPom-rich pipeline for season=%d", season)

    tracking.configure_from_env()
    tracking.init_experiment(EXPERIMENT)

    config = ModelConfig(model_variant="kenpom_rich")
    if xgb_params:
        config.xgb_params.update(xgb_params)
    if num_rounds is not None:
        config.num_rounds = num_rounds

    data = load_kaggle_data(season=season)

    all_seasons = sorted(data["M_reg_raw"]["Season"].unique().tolist())
    try:
        ingest_all_kenpom_seasons(seasons=all_seasons, snapshot_date=kenpom_snapshot_date)
        ingest_kenpom_height(seasons=all_seasons)
    except Exception as e:
        log.warning("KenPom ingestion skipped: %s", e)

    reg_sym, tourn_sym = build_symmetric_games(data)
    matchups, rating_meta, artifacts = compute_all_features(reg_sym, tourn_sym, data)

    # ── KenPom-rich: join novel team features onto the matchup dataset ─────────────
    kp_feats = build_rich_feature_frame(all_seasons, data)
    matchups, added = join_kenpom_rich(matchups, kp_feats)
    if added:
        log.info("KenPom rich features joined: %d cols (%s)", len(added), ", ".join(added))
    else:
        log.info("No KenPom rich features joined — running with baseline feature set")

    # ── Feature list: baseline candidates + kp_ differentials ─────────────────────
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
        # ── KenPom-rich additions (differentials only; A_/B_ available if wanted) ──
        # KP-006 dropped the minutes-weighted kp_Exp/kp_Bench/kp_Continuity (leaky).
        "d_kp_AvgHgt", "d_kp_HgtEff",
        "d_kp_SOS", "d_kp_Luck", "d_kp_APL_Off", "d_kp_APL_Def",
    ]
    features = [f for f in feature_candidates if f in matchups.columns]
    matchups[features] = matchups[features].fillna(0)

    n_kp = len([f for f in features if f.startswith("d_kp_")])
    log.info("KenPom-rich using %d features (%d KenPom additions)", len(features), n_kp)

    loto = run_training(matchups, features, config)
    log_eval_summary(loto, holdout_season)
    booster, calibrator = run_production_training(matchups, features, loto)
    if generate_sub:
        generate_submission(
            season, data, reg_sym, booster, calibrator, loto, rating_meta, artifacts["scaler"],
            kp_feats, competition=competition, message=submission_message, fetch_lb_score=fetch_lb_score,
        )

    log.info("KenPom-rich pipeline complete. LOTO Brier: %.6f  run_id=%s", loto.overall_brier, loto.loto_run_id)


if __name__ == "__main__":
    kenpom_rich_pipeline()
