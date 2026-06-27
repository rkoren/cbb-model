"""CBB feature engineering — kitchen-platform adapter.

Called by ``kitchen run features`` as::

    from src.features.run import build
    build(params, DataStore())

Also imported by ``kitchen run train`` (which runs features first via the
generic train_flow). All feature computation is delegated to the modules
in ``src/cbb/features/``; this file is purely the kitchen glue.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from kitchen.store import DataStore

from cbb.data import build_seed_lookup, build_symmetric_games
from cbb.features import (
    add_four_factors,
    build_matchup_dataset,
    compute_adj_efficiency,
    compute_elo,
    compute_glm_quality,
    compute_massey_ranks,
    compute_recent_form,
    compute_season_averages,
)
from cbb.kenpom.features import (
    build_team_name_map,
    load_kenpom_efficiency,
    merge_kenpom_efficiency,
)
from cbb.kenpom.rich_features import build_kenpom_rich_features, join_kenpom_rich
from cbb.holdout import (
    HOLDOUT_DIR,
    HOLDOUT_PARQUET,
    build_holdout_matchups,
    load_holdout_results,
)
from cbb.train.model import optimize_rating_weights

log = logging.getLogger(__name__)


def _load_all_csvs(params: dict, store: DataStore) -> dict[str, pd.DataFrame]:
    """Read the nine Kaggle CSVs listed in params.features.raw_files from store.raw_dir."""
    raw_files = params.get("features", {}).get("raw_files", [])
    data: dict[str, pd.DataFrame] = {}

    # Fixed key mapping: Kaggle CSV stem → dict key used by the rest of the pipeline
    key_map = {
        "MRegularSeasonDetailedResults": "M_reg_raw",
        "WRegularSeasonDetailedResults": "W_reg_raw",
        "MNCAATourneyDetailedResults": "M_tourn_raw",
        "WNCAATourneyDetailedResults": "W_tourn_raw",
        "MNCAATourneySeeds": "M_seeds",
        "WNCAATourneySeeds": "W_seeds",
        "MTeams": "M_teams",
        "WTeams": "W_teams",
        "MMasseyOrdinals": "massey",
    }
    for stem in raw_files:
        key = key_map.get(stem, stem)
        data[key] = store.load_csv(f"{stem}.csv")

    log.info("Loaded %d Kaggle CSVs from %s", len(data), store.raw_dir)
    return data


# KenPom inputs are fetched from the API (not regenerable from data/raw), so they live in a
# dedicated DVC-tracked dir — `dvc pull` restores them in CI. Populate with scripts/fetch_kenpom_*.
KENPOM_DIR = Path("data/kenpom")


def _apply_kenpom(
    adj_eff: pd.DataFrame,
    seasons: list[int],
    m_teams: pd.DataFrame,
    kenpom_dir: Path = KENPOM_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Best-effort KenPom integration: efficiency overlay + rich feature frame.

    For each season with a cached ``kenpom_ratings_{season}.parquet`` the KenPom→Kaggle team
    map is built once and reused for both:

      1. the men's efficiency overlay (AdjOE/AdjDE/AdjEM/AdjTempo), and
      2. the rich feature frame — ``kp_SOS/kp_Luck/kp_APL_Off/kp_APL_Def`` from the cached
         ratings parquet, plus ``kp_AvgHgt/kp_HgtEff/kp_Exp/kp_Bench/kp_Continuity`` when a
         cached ``kenpom_height_{season}.parquet`` exists (height is fetched/DVC-tracked
         separately, so it is simply absent until then).

    Women's rows are preserved from manual computation. Returns ``(adj_eff, kp_rich)`` where
    ``kp_rich`` is keyed (Season, men_women, TeamID) with ``kp_*`` columns — an empty frame
    when no KenPom data is available, which downstream joins handle as a no-op.
    """
    kp_frames: list[pd.DataFrame] = []
    try:
        from cbb.kenpom import KenPomClient  # noqa: PLC0415

        client = KenPomClient()
        merged = 0
        for s in seasons:
            ratings_path = kenpom_dir / f"kenpom_ratings_{s}.parquet"
            if not ratings_path.exists():
                continue
            try:
                team_map = build_team_name_map(m_teams, client.teams(year=s))
            except Exception as exc:  # noqa: BLE001
                log.warning("KenPom team map failed for season %d: %s", s, exc)
                continue

            try:
                kenpom_eff = load_kenpom_efficiency(s, ratings_path, team_map)
                adj_eff = merge_kenpom_efficiency(adj_eff, kenpom_eff)
                merged += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("KenPom efficiency merge failed for season %d: %s", s, exc)

            try:
                ratings_df = pd.read_parquet(ratings_path)
                height_path = kenpom_dir / f"kenpom_height_{s}.parquet"
                height_df = pd.read_parquet(height_path) if height_path.exists() else None
                kp_frames.append(
                    build_kenpom_rich_features(s, team_map, ratings_df, height_df)
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("KenPom rich features failed for season %d: %s", s, exc)
        if merged:
            log.info("KenPom efficiency merged for %d/%d seasons", merged, len(seasons))
    except Exception as exc:  # noqa: BLE001
        log.warning("KenPom integration skipped: %s", exc)

    kp_rich = (
        pd.concat(kp_frames, ignore_index=True)
        if kp_frames
        else pd.DataFrame(columns=["Season", "men_women", "TeamID"])
    )
    return adj_eff, kp_rich


def build(params: dict, store: DataStore) -> None:
    """Build all CBB features and write them to ``data/processed/``.

    Reads
    -----
    ``data/raw/`` — nine Kaggle CSVs (must exist; run ``dvc pull`` first).
    ``data/kenpom/kenpom_ratings_{season}.parquet`` — optional KenPom cache;
    run ``python -m cbb.kenpom.ingest`` or the Prefect ingest task to populate.

    Writes
    ------
    ``data/processed/adj_eff.parquet``
    ``data/processed/season_avgs.parquet``
    ``data/processed/elo_df.parquet``
    ``data/processed/form_df.parquet``
    ``data/processed/quality_df.parquet``
    ``data/processed/massey_lookup.parquet``
    ``data/processed/seed_lookup.parquet``
    ``data/processed/scaler.pkl``
    ``data/processed/rating_meta.json``
    ``data/processed/matchups.parquet``  ← consumed by ``kitchen run train``

    Args:
        params: Parsed ``params.yaml`` dict.
        store: ``DataStore`` rooted at the project directory.
    """
    proc = store.processed_dir
    proc.mkdir(parents=True, exist_ok=True)

    z_features: list[str] = params.get("features", {}).get(
        "z_features", ["d_AdjEM", "d_Elo", "d_Quality", "d_Form"]
    )

    # ── Load raw data ──────────────────────────────────────────────────────────
    data = _load_all_csvs(params, store)

    # ── Symmetric game representations ────────────────────────────────────────
    reg_sym, tourn_sym = build_symmetric_games(data)
    log.info("reg_sym: %d rows  tourn_sym: %d rows", len(reg_sym), len(tourn_sym))

    # ── Feature computation ────────────────────────────────────────────────────
    adj_eff = compute_adj_efficiency(reg_sym)
    log.info("adj_eff: %d rows  mean iters %.1f", len(adj_eff), adj_eff["iters"].mean())

    seasons = sorted(reg_sym["Season"].unique().tolist())
    adj_eff, kp_rich = _apply_kenpom(adj_eff, seasons, data["M_teams"])
    store.save_parquet(adj_eff, "adj_eff.parquet")

    season_avgs = add_four_factors(compute_season_averages(reg_sym))
    store.save_parquet(season_avgs, "season_avgs.parquet")

    M_elo = compute_elo(data["M_reg_raw"], men_women_flag=0)
    W_elo = compute_elo(data["W_reg_raw"], men_women_flag=1)
    elo_df = pd.concat([M_elo, W_elo], ignore_index=True)
    store.save_parquet(elo_df, "elo_df.parquet")

    form_df = compute_recent_form(reg_sym)
    store.save_parquet(form_df, "form_df.parquet")

    quality_df = compute_glm_quality(reg_sym)
    store.save_parquet(quality_df, "quality_df.parquet")

    massey_lookup = compute_massey_ranks(data["massey"])
    massey_df = pd.DataFrame(
        [(s, t, v) for (s, t), v in massey_lookup.items()],
        columns=["Season", "TeamID", "MasseyRank"],
    )
    store.save_parquet(massey_df, "massey_lookup.parquet")

    seed_lookup = build_seed_lookup(data["M_seeds"], data["W_seeds"])
    seed_df = (
        pd.DataFrame(
            [(s, t, v) for (s, t), v in seed_lookup.items()],
            columns=["Season", "TeamID", "SeedNum"],
        )
        .astype({"Season": int, "TeamID": int, "SeedNum": int})
    )
    store.save_parquet(seed_df, "seed_lookup.parquet")

    # ── Matchup dataset ────────────────────────────────────────────────────────
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

    # ── Rating blend: d_Rating = weighted sum of z-scored differentials ────────
    scaler = StandardScaler()
    z_vals = matchups[z_features].fillna(0)
    z_scaled = scaler.fit_transform(z_vals)
    z_cols = [f"{c}_z" for c in z_features]
    for i, col in enumerate(z_cols):
        matchups[col] = z_scaled[:, i]

    opt_weights = optimize_rating_weights(matchups, z_cols)
    matchups["d_Rating"] = (matchups[z_cols] * opt_weights).sum(axis=1)

    # ── KenPom rich features (additive) ────────────────────────────────────────
    # Adds A_/B_/d_kp_* columns. Harmless to the baseline model, which selects only
    # its `feature_candidates`; an experiment opts in by adding `d_kp_*` to that list.
    matchups, kp_added = join_kenpom_rich(matchups, kp_rich)
    if kp_added:
        kp_diffs = sorted(c for c in kp_added if c.startswith("d_"))
        log.info("KenPom rich features joined: %d cols (%s)", len(kp_added), ", ".join(kp_diffs))
    else:
        log.info("No KenPom rich features available — matchups has baseline columns only")

    rating_meta = {"opt_weights": opt_weights.tolist(), "z_cols": z_cols}
    with open(proc / "rating_meta.json", "w", encoding="utf-8") as f:
        json.dump(rating_meta, f)

    with open(proc / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    store.save_parquet(matchups, "matchups.parquet")
    log.info(
        "Features built → %s  (%d rows, %d cols)",
        proc / "matchups.parquet",
        len(matchups),
        len(matchups.columns),
    )

    # ── 2026 frozen holdout (gated: only once actual 2026 results are provided) ──
    # Built from the same in-memory frames + post-processing → feature parity with training.
    # Training never sees these games (they aren't in matchups.parquet), so it stays leak-free.
    holdout_results = load_holdout_results()
    if holdout_results is None:
        log.info("No 2026 holdout results in %s/ — skipping holdout (drop them in to enable)", HOLDOUT_DIR)
    else:
        try:
            holdout = build_holdout_matchups(
                holdout_results, adj_eff, season_avgs, elo_df, quality_df, form_df,
                seed_lookup, massey_lookup, reg_sym, scaler, opt_weights, z_features, kp_rich,
            )
            store.save_parquet(holdout, HOLDOUT_PARQUET)
            log.info("Holdout built → %s  (%d games)", proc / HOLDOUT_PARQUET, len(holdout))
        except Exception as exc:  # noqa: BLE001
            log.warning("Holdout build skipped: %s", exc)
