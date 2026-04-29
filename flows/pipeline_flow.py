"""Prefect pipeline: ingest KenPom → build features → train → log to MLflow.

Run locally:
    python flows/pipeline_flow.py

Schedule via Prefect UI or CLI:
    prefect deployment build flows/pipeline_flow.py:cbb_pipeline -n prod
"""

import os
from datetime import date
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv
from prefect import flow, task, get_run_logger
from sklearn.preprocessing import StandardScaler

from cbb.kenpom import KenPomClient
from cbb.features import (
    compute_adj_efficiency,
    compute_season_averages,
    add_four_factors,
    compute_elo,
    compute_recent_form,
    compute_glm_quality,
    compute_massey_ranks,
    build_matchup_dataset,
)
from cbb.train.model import (
    ModelConfig,
    LoTOResult,
    optimize_rating_weights,
    train_loto,
    train_production,
)
from cbb.evaluate import from_season_dict

load_dotenv()

DATA_RAW = Path(os.environ.get("DATA_RAW_DIR", "data/raw"))
DATA_PROC = Path(os.environ.get("DATA_PROC_DIR", "data/processed"))
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "cbb-tournament")

# ── Feature columns used in training (keep in sync with model artifacts) ──────
Z_FEATURES = ["d_AdjEM", "d_Elo", "d_Quality", "d_Form"]


# ── Tasks ─────────────────────────────────────────────────────────────────────

@task(retries=3, retry_delay_seconds=30)
def ingest_kenpom_ratings(season: int, snapshot_date: str | None = None) -> pd.DataFrame:
    """Pull KenPom ratings for a season, optionally at a historical date.

    For tournament predictions, snapshot_date should be Selection Sunday (≈ YYYY-03-16)
    to avoid leakage from post-tournament rating updates.
    """
    log = get_run_logger()
    client = KenPomClient()

    if snapshot_date:
        log.info("Fetching KenPom archive for %s at %s", season, snapshot_date)
        df = client.ratings_archive(date=snapshot_date)
    else:
        log.info("Fetching KenPom live ratings for %s", season)
        df = client.ratings(year=season)

    out = DATA_PROC / f"kenpom_ratings_{season}.parquet"
    df.to_parquet(out, index=False)
    log.info("Saved %d team ratings to %s", len(df), out)
    return df


@task
def load_kaggle_data(season: int | None = None) -> dict[str, pd.DataFrame]:
    """Load Kaggle competition CSVs from data/raw/.

    Kaggle data is DVC-tracked; run `dvc pull` before this task if running fresh.
    """
    log = get_run_logger()

    def _load(name):
        path = DATA_RAW / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}. Run `dvc pull` first.")
        return pd.read_csv(path)

    data = {
        "M_reg_raw": _load("MRegularSeasonDetailedResults"),
        "W_reg_raw": _load("WRegularSeasonDetailedResults"),
        "M_tourn_raw": _load("MNCAATourneyDetailedResults"),
        "W_tourn_raw": _load("WNCAATourneyDetailedResults"),
        "M_seeds": _load("MNCAATourneySeeds"),
        "W_seeds": _load("WNCAATourneySeeds"),
        "M_teams": _load("MTeams"),
        "W_teams": _load("WTeams"),
        "massey": _load("MMasseyOrdinals"),
    }
    log.info("Kaggle data loaded. M_reg games: %d", len(data["M_reg_raw"]))
    return data


@task
def build_symmetric_games(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert W/L game logs to symmetric T1/T2 format, OT-adjusted.

    Returns (reg_sym, tourn_sym) — symmetric regular-season and tournament games.
    """
    log = get_run_logger()
    FT_FACTOR = 0.475
    BOX_COLS = ["Score", "FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA", "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]

    def _normalize(df: pd.DataFrame, mw: int, is_tourn: bool = False) -> pd.DataFrame:
        records = []
        for _, row in df.iterrows():
            adjot = (40 + 5 * row.get("NumOT", 0)) / 40
            base = {"Season": row["Season"], "DayNum": row["DayNum"], "men_women": mw, "is_tourn": int(is_tourn)}
            loc = row.get("WLoc", "N")

            for t1_is_winner in [True, False]:
                rec = dict(base)
                if t1_is_winner:
                    rec["T1_TeamID"] = row["WTeamID"]
                    rec["T2_TeamID"] = row["LTeamID"]
                    rec["Outcome"] = 1
                    rec["T1_home"] = 1 if loc == "H" else (-1 if loc == "A" else 0)
                    for c in BOX_COLS:
                        rec[f"T1_{c}"] = row[f"W{c}"] / adjot
                        rec[f"T2_{c}"] = row[f"L{c}"] / adjot
                else:
                    rec["T1_TeamID"] = row["LTeamID"]
                    rec["T2_TeamID"] = row["WTeamID"]
                    rec["Outcome"] = 0
                    rec["T1_home"] = -1 if loc == "H" else (1 if loc == "A" else 0)
                    for c in BOX_COLS:
                        rec[f"T1_{c}"] = row[f"L{c}"] / adjot
                        rec[f"T2_{c}"] = row[f"W{c}"] / adjot
                rec["PointDiff"] = rec["T1_Score"] - rec["T2_Score"]
                records.append(rec)
        return pd.DataFrame(records)

    parts_reg = [
        _normalize(data["M_reg_raw"], mw=0),
        _normalize(data["W_reg_raw"], mw=1),
    ]
    parts_tourn = [
        _normalize(data["M_tourn_raw"], mw=0, is_tourn=True),
        _normalize(data["W_tourn_raw"], mw=1, is_tourn=True),
    ]
    reg_sym = pd.concat(parts_reg, ignore_index=True)
    tourn_sym = pd.concat(parts_tourn, ignore_index=True)
    log.info("reg_sym: %d rows, tourn_sym: %d rows", len(reg_sym), len(tourn_sym))
    return reg_sym, tourn_sym


@task
def compute_all_features(
    reg_sym: pd.DataFrame,
    tourn_sym: pd.DataFrame,
    data: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict, dict]:
    """Run all feature computations and build the matchup dataset."""
    log = get_run_logger()

    adj_eff = compute_adj_efficiency(reg_sym)
    log.info("adj_eff: %d rows, mean iters %.1f", len(adj_eff), adj_eff["iters"].mean())

    season_avgs = add_four_factors(compute_season_averages(reg_sym))

    M_elo = compute_elo(data["M_reg_raw"], men_women_flag=0)
    W_elo = compute_elo(data["W_reg_raw"], men_women_flag=1)
    elo_df = pd.concat([M_elo, W_elo], ignore_index=True)

    form_df = compute_recent_form(reg_sym)
    quality_df = compute_glm_quality(reg_sym)
    massey_lookup = compute_massey_ranks(data["massey"])

    # Seed lookup: (Season, TeamID) → seed number
    def _parse_seed(s):
        digits = "".join(filter(str.isdigit, str(s)[1:3]))
        return int(digits) if digits else None

    seeds = pd.concat([data["M_seeds"], data["W_seeds"]], ignore_index=True)
    seeds["SeedNum"] = seeds["Seed"].apply(_parse_seed)
    seed_lookup = seeds.set_index(["Season", "TeamID"])["SeedNum"].to_dict()

    matchups = build_matchup_dataset(
        tourn_sym=tourn_sym,
        reg_sym=reg_sym,
        adj_eff=adj_eff,
        season_avgs=season_avgs,
        elo_df=elo_df,
        quality_df=quality_df,
        form_df=form_df,
        seed_lookup=seed_lookup,
        massey_lookup=massey_lookup,
    )

    # Optimized rating blend: d_Rating = weighted sum of z-scored differentials
    scaler = StandardScaler()
    z_vals = matchups[Z_FEATURES].fillna(0)
    z_scaled = scaler.fit_transform(z_vals)
    z_cols = [f"{c}_z" for c in Z_FEATURES]
    for i, c in enumerate(z_cols):
        matchups[c] = z_scaled[:, i]

    opt_weights = optimize_rating_weights(matchups, z_cols)
    matchups["d_Rating"] = (matchups[z_cols] * opt_weights).sum(axis=1)

    log.info("Matchup dataset: %s", matchups.shape)
    out = DATA_PROC / "matchups.parquet"
    matchups.to_parquet(out, index=False)
    log.info("Saved matchups to %s", out)

    rating_meta = {"opt_weights": opt_weights.tolist(), "z_cols": z_cols}
    return matchups, rating_meta, {"scaler": scaler}


@task
def run_training(
    matchups: pd.DataFrame,
    features: list[str],
    config: ModelConfig | None = None,
) -> LoTOResult:
    """LOTO training + calibration + temperature scaling."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    return train_loto(matchups, features, config=config, mlflow_experiment=EXPERIMENT)


@task
def log_eval_summary(loto: LoTOResult, holdout_season: int) -> None:
    """Log a human-readable eval summary; surfaces holdout season Brier prominently."""
    log = get_run_logger()
    result = from_season_dict(loto.brier_by_season, n_games=len(loto.matchups))
    holdout = result.season(holdout_season)
    log.info(
        "Eval — LOTO avg: %.6f  |  %d holdout: %s  |  n_games: %d",
        loto.overall_brier,
        holdout_season,
        f"{holdout:.6f}" if holdout is not None else "n/a (season not in data)",
        result.n_games,
    )
    for season in sorted(result.per_season):
        log.info("  brier_%d = %.6f", season, result.per_season[season])


@task
def run_production_training(
    matchups: pd.DataFrame,
    features: list[str],
    loto: LoTOResult,
) -> None:
    """Train and register the production model artifact."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    booster, calibrator = train_production(
        matchups,
        features,
        loto.temp_params,
        mlflow_experiment=EXPERIMENT,
    )
    log = get_run_logger()
    log.info(
        "Production model trained. Brier LOTO=%.6f  T_M_close=%.3f",
        loto.overall_brier,
        loto.temp_params["T_M_close"],
    )


# ── Flow ───────────────────────────────────────────────────────────────────────

@flow(name="cbb-pipeline")
def cbb_pipeline(
    season: int | None = None,
    kenpom_snapshot_date: str | None = None,
    holdout_season: int = 2026,
    xgb_params: dict | None = None,
    num_rounds: int | None = None,
):
    """End-to-end CBB pipeline: ingest → features → train → log.

    Args:
        season: Season year (e.g. 2027). Defaults to current year.
        kenpom_snapshot_date: Date for KenPom archive snapshot (YYYY-MM-DD).
                              Use Selection Sunday date for tournament predictions.
        holdout_season: Season to highlight in eval summary (default: 2026).
        xgb_params: Override XGBoost params for experiment runs.
        num_rounds: Override boosting rounds for experiment runs.
    """
    log = get_run_logger()
    season = season or date.today().year
    log.info("Running CBB pipeline for season=%d", season)

    config = ModelConfig()
    if xgb_params:
        config.xgb_params.update(xgb_params)
    if num_rounds is not None:
        config.num_rounds = num_rounds

    # KenPom ingestion is optional — training runs on Kaggle box scores alone.
    # Skipped gracefully if KENPOM_API_KEY is not yet configured.
    try:
        ingest_kenpom_ratings(season=season, snapshot_date=kenpom_snapshot_date)
    except Exception as e:
        log.warning("KenPom ingestion skipped: %s", e)

    data = load_kaggle_data(season=season)
    reg_sym, tourn_sym = build_symmetric_games(data)
    matchups, rating_meta, _ = compute_all_features(reg_sym, tourn_sym, data)

    # Build feature list (same logic as notebook FEATURES list, filtered to present cols)
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
    ]
    features = [f for f in feature_candidates if f in matchups.columns]
    matchups[features] = matchups[features].fillna(0)

    loto = run_training(matchups, features, config)
    log_eval_summary(loto, holdout_season)
    run_production_training(matchups, features, loto)

    log.info("Pipeline complete. LOTO Brier: %.6f", loto.overall_brier)


if __name__ == "__main__":
    cbb_pipeline()
