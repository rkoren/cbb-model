"""Baseline model: iterative efficiency + four factors + Elo + GLM quality + Massey.

This is the active production model. All feature engineering mirrors the
efficiency_approach.ipynb notebook. Compare against experiments/challenger.py
to evaluate new feature additions.

Run locally (from repo root):
    python experiments/baseline.py

Schedule via Prefect UI:
    prefect deployment build experiments/baseline.py:cbb_pipeline -n prod
"""

import json
import os
import pickle
from datetime import date
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from prefect import flow, task, get_run_logger
from sklearn.preprocessing import StandardScaler

from kitchen import tracking
from kitchen.store import DataStore
from kitchen.submit import check_feature_parity, log_submission

from cbb.data import build_seed_lookup, build_symmetric_games as _build_sym
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
    build_prediction_features,
)
from cbb.train.model import (
    ModelConfig,
    LoTOResult,
    log_sklearn_model,
    optimize_rating_weights,
    predict_batch,
    train_loto,
    train_production,
)
from cbb.evaluate import from_season_dict
from cbb.kenpom.features import build_team_name_map, load_kenpom_efficiency, merge_kenpom_efficiency

load_dotenv()

DATA_RAW = Path(os.environ.get("DATA_RAW_DIR", "data/raw"))
DATA_PROC = Path(os.environ.get("DATA_PROC_DIR", "data/processed"))
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "cbb-tournament")

# Shared DataStore instance (rooted at repo root) — used in all tasks that
# read/write data/processed/.  Tasks that need raw data still read from DATA_RAW
# directly so the DVC-managed path override via DATA_RAW_DIR is preserved.
_store = DataStore(root=".")

# ── Feature columns used in training (keep in sync with model artifacts) ──────
Z_FEATURES = ["d_AdjEM", "d_Elo", "d_Quality", "d_Form"]


# ── Tasks ─────────────────────────────────────────────────────────────────────

@task
def ingest_all_kenpom_seasons(seasons: list[int], snapshot_date: str | None = None) -> None:
    """Fetch and cache KenPom ratings for all seasons, skipping already-saved ones.

    Uses ratings_archive(date=snapshot_date) for the most recent season when
    snapshot_date is provided (avoids post-tournament leakage). All other seasons
    use ratings(year=s) — tournament is already over, no leakage risk.

    Run this once before training; parquets persist across runs.
    """
    log = get_run_logger()
    client = KenPomClient()

    for s in seasons:
        out = DATA_PROC / f"kenpom_ratings_{s}.parquet"
        if out.exists():
            log.info("KenPom ratings for %d already cached — skipping", s)
            continue
        try:
            if snapshot_date and s == max(seasons):
                df = client.ratings_archive(date=snapshot_date)
                log.info("Fetched KenPom archive for %d at %s (%d teams)", s, snapshot_date, len(df))
            else:
                df = client.ratings(year=s)
                log.info("Fetched KenPom ratings for %d (%d teams)", s, len(df))
            df.to_parquet(out, index=False)
        except Exception as e:
            log.warning("KenPom fetch failed for season %d: %s", s, e)


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

    Delegates to ``cbb.data.build_symmetric_games`` — logic lives there so
    both this Prefect flow and ``src/features/run.py`` share the same code.

    Returns (reg_sym, tourn_sym) — symmetric regular-season and tournament games.
    """
    log = get_run_logger()
    reg_sym, tourn_sym = _build_sym(data)
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

    # Merge official KenPom efficiency for men's games where cached parquets exist.
    # Women's rows are always preserved from manual computation.
    seasons_in_data = sorted(reg_sym["Season"].unique())
    kenpom_merged_count = 0
    for s in seasons_in_data:
        kenpom_path = _store.processed_dir / f"kenpom_ratings_{s}.parquet"
        if kenpom_path.exists():
            try:
                team_map = build_team_name_map(data["M_teams"], KenPomClient().teams(year=s))
                kenpom_eff = load_kenpom_efficiency(s, kenpom_path, team_map)
                adj_eff = merge_kenpom_efficiency(adj_eff, kenpom_eff)
                kenpom_merged_count += 1
            except Exception as e:
                log.warning("KenPom merge failed for season %d: %s", s, e)
    if kenpom_merged_count:
        log.info("KenPom efficiency merged for %d/%d seasons", kenpom_merged_count, len(seasons_in_data))
    # Save post-KenPom so adj_eff.parquet reflects the final merged values
    _store.save_parquet(adj_eff, "adj_eff.parquet")

    season_avgs = add_four_factors(compute_season_averages(reg_sym))
    _store.save_parquet(season_avgs, "season_avgs.parquet")

    M_elo = compute_elo(data["M_reg_raw"], men_women_flag=0)
    W_elo = compute_elo(data["W_reg_raw"], men_women_flag=1)
    elo_df = pd.concat([M_elo, W_elo], ignore_index=True)
    _store.save_parquet(elo_df, "elo_df.parquet")

    form_df = compute_recent_form(reg_sym)
    _store.save_parquet(form_df, "form_df.parquet")

    quality_df = compute_glm_quality(reg_sym)
    _store.save_parquet(quality_df, "quality_df.parquet")

    massey_lookup = compute_massey_ranks(data["massey"])
    _store.save_parquet(
        pd.DataFrame(
            [(s, t, v) for (s, t), v in massey_lookup.items()],
            columns=["Season", "TeamID", "MasseyRank"],
        ),
        "massey_lookup.parquet",
    )

    # Seed lookup: (Season, TeamID) → seed number
    seed_lookup = build_seed_lookup(data["M_seeds"], data["W_seeds"])
    _store.save_parquet(
        pd.DataFrame(
            [(s, t, v) for (s, t), v in seed_lookup.items()],
            columns=["Season", "TeamID", "SeedNum"],
        ).astype({"Season": int, "TeamID": int, "SeedNum": int}),
        "seed_lookup.parquet",
    )

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
    _store.save_parquet(matchups, "matchups.parquet")

    rating_meta = {"opt_weights": opt_weights.tolist(), "z_cols": z_cols}
    proc = _store.processed_dir
    with open(proc / "rating_meta.json", "w") as f:
        json.dump(rating_meta, f)
    with open(proc / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return matchups, rating_meta, {"scaler": scaler}


@task
def run_training(
    matchups: pd.DataFrame,
    features: list[str],
    config: ModelConfig | None = None,
) -> LoTOResult:
    """LOTO training + calibration + temperature scaling."""
    return train_loto(
        matchups,
        features,
        config=config,
        mlflow_experiment=EXPERIMENT,
    )


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
) -> tuple:
    """Train production model on all seasons, save locally, log to LOTO run."""
    import mlflow.sklearn
    booster, calibrator, total_booster = train_production(matchups, features, loto.temp_params)
    log = get_run_logger()
    log.info(
        "Production model trained. Brier LOTO=%.6f  T_M_close=%.3f",
        loto.overall_brier,
        loto.temp_params["T_M_close"],
    )
    # Save locally for submission generation (avoids MLflow round-trip)
    proc = _store.processed_dir
    booster.save_model(str(proc / "prod_booster.ubj"))
    with open(proc / "prod_calibrator.pkl", "wb") as f:
        pickle.dump(calibrator, f)
    # Bundle into CBBModel and save as a single pickle for kitchen run evaluate
    from cbb.train.model import CBBModel  # noqa: PLC0415
    cbb_model = loto.to_model(booster, calibrator, total_booster)
    with open(proc / "cbb_model.pkl", "wb") as f:
        pickle.dump(cbb_model, f)
    # Log model artifact back into the LOTO run so it's co-located with metrics
    if loto.loto_run_id:
        loto_meta = {"features": loto.features, "temp_params": loto.temp_params, "vegas_alpha": loto.vegas_alpha}
        loto_meta_path = proc / "loto_meta.json"
        with open(loto_meta_path, "w") as f:
            json.dump(loto_meta, f)
        with mlflow.start_run(run_id=loto.loto_run_id):
            log_sklearn_model(cbb_model, "cbb_model")
            log_sklearn_model(calibrator, "calibrator")
            mlflow.log_artifact(str(proc / "prod_booster.ubj"), "xgb_model")
            mlflow.log_artifact(str(proc / "rating_meta.json"), "run_meta")
            mlflow.log_artifact(str(proc / "scaler.pkl"), "run_meta")
            mlflow.log_artifact(str(loto_meta_path), "run_meta")
        log.info("Production artifacts logged to LOTO run %s", loto.loto_run_id)
    return booster, calibrator


def _run_submission(
    season: int,
    data: dict,
    reg_sym: pd.DataFrame,
    booster,
    calibrator,
    loto: LoTOResult,
    rating_meta: dict,
    scaler,
    pre_predict_hook=None,
) -> pd.DataFrame:
    """Build and save the Kaggle submission CSV for the given season.

    Generates all C(68, 2) seeded matchup pairs for both men's and women's
    tournaments (A_TeamID < B_TeamID, matching Kaggle ID format), computes
    features from saved parquets, runs prediction, and writes
    data/processed/submission_{season}.csv.

    Args:
        season: Target season year.
        data: Kaggle data dict from load_kaggle_data.
        reg_sym: Symmetric regular-season game log.
        booster: Production XGBoost booster.
        calibrator: Production logistic calibrator.
        loto: LoTOResult with temp_params, features, vegas_alpha.
        rating_meta: Dict with opt_weights and z_cols.
        scaler: Fitted StandardScaler for Z_FEATURES → z_cols transform.
        pre_predict_hook: Optional callable(pred_df) → pred_df applied before
                          prediction — used by challenger to inject derived features.

    Returns:
        Submission DataFrame with ID and Pred columns.
    """
    adj_eff = _store.load_parquet("adj_eff.parquet")
    season_avgs = _store.load_parquet("season_avgs.parquet")
    elo_df = _store.load_parquet("elo_df.parquet")
    quality_df = _store.load_parquet("quality_df.parquet")
    form_df = _store.load_parquet("form_df.parquet")
    massey_lookup = (
        _store.load_parquet("massey_lookup.parquet")
        .set_index(["Season", "TeamID"])["MasseyRank"]
        .to_dict()
    )
    seed_lookup = (
        _store.load_parquet("seed_lookup.parquet")
        .set_index(["Season", "TeamID"])["SeedNum"]
        .to_dict()
    )

    # All C(68, 2) seeded pairs; A_TeamID < B_TeamID matches Kaggle ID format
    pair_rows = []
    for mw, seed_key in [(0, "M_seeds"), (1, "W_seeds")]:
        season_seeds = data[seed_key][data[seed_key]["Season"] == season]
        team_ids = sorted(season_seeds["TeamID"].unique())
        for ta, tb in combinations(team_ids, 2):
            pair_rows.append({"Season": season, "men_women": mw, "A_TeamID": ta, "B_TeamID": tb})
    pairs = pd.DataFrame(pair_rows)

    pred_df = build_prediction_features(
        pairs, adj_eff, season_avgs, elo_df, quality_df, form_df,
        seed_lookup, massey_lookup, reg_sym,
    )

    # Compute d_Rating: apply same scaler + opt_weights from training
    z_cols = rating_meta["z_cols"]
    base_feats = [c.replace("_z", "") for c in z_cols]
    z_scaled = scaler.transform(pred_df[base_feats].fillna(0))
    for i, c in enumerate(z_cols):
        pred_df[c] = z_scaled[:, i]
    opt_weights = np.array(rating_meta["opt_weights"])
    pred_df["d_Rating"] = (pred_df[z_cols] * opt_weights).sum(axis=1)

    if pre_predict_hook is not None:
        pred_df = pre_predict_hook(pred_df)

    parity_errors = check_feature_parity(loto.features, pred_df)
    if parity_errors:
        raise RuntimeError(
            "Feature parity check failed — submission features don't match training:\n"
            + "\n".join(f"  {e}" for e in parity_errors)
        )

    probs = predict_batch(
        pred_df, loto.features, booster, calibrator, loto.temp_params, loto.vegas_alpha,
    )

    ids = [f"{r.Season}_{r.A_TeamID}_{r.B_TeamID}" for r in pairs.itertuples()]
    submission = pd.DataFrame({"ID": ids, "Pred": probs})
    out = _store.processed_dir / f"submission_{season}.csv"
    submission.to_csv(out, index=False)
    return submission


@task
def generate_submission(
    season: int,
    data: dict,
    reg_sym: pd.DataFrame,
    booster,
    calibrator,
    loto: LoTOResult,
    rating_meta: dict,
    scaler,
    competition: str | None = None,
    message: str = "",
    fetch_lb_score: bool = False,
) -> pd.DataFrame:
    """Generate, validate, and log the Kaggle submission for the target season.

    Args:
        competition: Kaggle competition slug. When set, uploads the submission.
        message: Submission message shown on the leaderboard.
        fetch_lb_score: Poll for the public LB score after uploading (requires competition).
    """
    log = get_run_logger()
    sub = _run_submission(season, data, reg_sym, booster, calibrator, loto, rating_meta, scaler)
    sub_path = _store.processed_dir / f"submission_{season}.csv"
    sample_path = DATA_RAW / "SampleSubmissionStage1.csv"

    if sample_path.exists():
        import pandas as _pd
        sample = _pd.read_csv(sample_path)
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

    log.info("Submission saved (%d rows) → %s", len(sub), sub_path)
    return sub


# ── Flow ───────────────────────────────────────────────────────────────────────

@flow(name="cbb-pipeline")
def cbb_pipeline(
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
    """End-to-end CBB pipeline: ingest → features → train → log → submit.

    Args:
        season: Season year (e.g. 2027). Defaults to current year.
        kenpom_snapshot_date: Date for KenPom archive snapshot (YYYY-MM-DD).
                              Use Selection Sunday date for tournament predictions.
        holdout_season: Season to highlight in eval summary (default: 2026).
        xgb_params: Override XGBoost params for experiment runs.
        num_rounds: Override boosting rounds for experiment runs.
        generate_sub: Generate a submission CSV after training (default False).
                      Use flows/submit.py to generate from a specific run later.
        competition: Kaggle competition slug. When set, uploads the submission.
        submission_message: Message shown on the Kaggle leaderboard.
        fetch_lb_score: Poll for the public LB score after uploading.
    """
    log = get_run_logger()
    season = season or date.today().year
    log.info("Running CBB pipeline for season=%d", season)

    tracking.configure_from_env()
    tracking.init_experiment(EXPERIMENT)

    config = ModelConfig(model_variant="baseline")
    if xgb_params:
        config.xgb_params.update(xgb_params)
    if num_rounds is not None:
        config.num_rounds = num_rounds

    data = load_kaggle_data(season=season)

    # Ingest KenPom ratings for all training seasons (skips already-cached seasons).
    # Training falls back to manual efficiency for any season without a cached parquet.
    try:
        all_seasons = sorted(data["M_reg_raw"]["Season"].unique().tolist())
        ingest_all_kenpom_seasons(seasons=all_seasons, snapshot_date=kenpom_snapshot_date)
    except Exception as e:
        log.warning("KenPom ingestion skipped: %s", e)
    reg_sym, tourn_sym = build_symmetric_games(data)
    matchups, rating_meta, artifacts = compute_all_features(reg_sym, tourn_sym, data)

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
    booster, calibrator = run_production_training(matchups, features, loto)
    if generate_sub:
        generate_submission(
            season, data, reg_sym, booster, calibrator, loto, rating_meta, artifacts["scaler"],
            competition=competition, message=submission_message, fetch_lb_score=fetch_lb_score,
        )

    log.info("Pipeline complete. LOTO Brier: %.6f  run_id=%s", loto.overall_brier, loto.loto_run_id)


if __name__ == "__main__":
    cbb_pipeline()
