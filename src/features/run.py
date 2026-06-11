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


def _try_merge_kenpom(
    adj_eff: pd.DataFrame,
    seasons: list[int],
    m_teams: pd.DataFrame,
    processed_dir: Path,
) -> pd.DataFrame:
    """Best-effort: overlay official KenPom efficiency where cached parquets exist.

    Women's rows are always preserved from manual computation. Skips any season
    that lacks a cached kenpom_ratings_{season}.parquet file.
    """
    try:
        from cbb.kenpom import KenPomClient  # noqa: PLC0415

        client = KenPomClient()
        merged = 0
        for s in seasons:
            kenpom_path = processed_dir / f"kenpom_ratings_{s}.parquet"
            if not kenpom_path.exists():
                continue
            try:
                team_map = build_team_name_map(m_teams, client.teams(year=s))
                kenpom_eff = load_kenpom_efficiency(s, kenpom_path, team_map)
                adj_eff = merge_kenpom_efficiency(adj_eff, kenpom_eff)
                merged += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("KenPom merge failed for season %d: %s", s, exc)
        if merged:
            log.info("KenPom efficiency merged for %d/%d seasons", merged, len(seasons))
    except Exception as exc:  # noqa: BLE001
        log.warning("KenPom merge skipped: %s", exc)
    return adj_eff


def build(params: dict, store: DataStore) -> None:
    """Build all CBB features and write them to ``data/processed/``.

    Reads
    -----
    ``data/raw/`` — nine Kaggle CSVs (must exist; run ``dvc pull`` first).
    ``data/processed/kenpom_ratings_{season}.parquet`` — optional KenPom cache;
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
    adj_eff = _try_merge_kenpom(adj_eff, seasons, data["M_teams"], proc)
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
