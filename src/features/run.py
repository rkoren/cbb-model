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
from cbb.features.reg_games import build_reg_games
from cbb.kenpom.features import (
    build_team_name_map,
    load_selection_sunday_efficiency,
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
    team_spellings: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, dict]]:
    """Best-effort KenPom integration: efficiency overlay + rich feature frame.

    For each season with a cached ``kenpom_ratings_{season}.parquet`` the KenPom→Kaggle team
    map is built once and reused for both:

      1. the men's efficiency overlay (AdjOE/AdjDE/AdjEM/AdjTempo), and
      2. the rich feature frame — ``kp_SOS/kp_Luck/kp_APL_Off/kp_APL_Def`` from the cached
         ratings parquet, plus ``kp_AvgHgt/kp_HgtEff/kp_Exp/kp_Bench/kp_Continuity`` when a
         cached ``kenpom_height_{season}.parquet`` exists (height is fetched/DVC-tracked
         separately, so it is simply absent until then).

    Women's rows are preserved from manual computation. Returns ``(adj_eff, kp_rich, team_maps)``
    where ``kp_rich`` is keyed (Season, men_women, TeamID) with ``kp_*`` columns (empty when no
    KenPom data is available — downstream joins handle it as a no-op) and ``team_maps`` is
    ``{season: {KenPom name → Kaggle TeamID}}`` (reused for the GM-002 as-of-date archive join).
    """
    kp_frames: list[pd.DataFrame] = []
    team_maps: dict[int, dict] = {}
    try:
        from cbb.kenpom import KenPomClient  # noqa: PLC0415

        client = KenPomClient()
        merged = 0
        for s in seasons:
            ratings_path = kenpom_dir / f"kenpom_ratings_{s}.parquet"
            if not ratings_path.exists():
                continue
            try:
                team_map = build_team_name_map(m_teams, client.teams(year=s), team_spellings)
                team_maps[s] = team_map
            except Exception as exc:  # noqa: BLE001
                log.warning("KenPom team map failed for season %d: %s", s, exc)
                continue

            try:
                # KP-005: leak-free efficiency from the Selection-Sunday archive, NOT the cached
                # end-of-season `ratings` (which == AdjEMFinal → leaks the tournament). Empty for
                # pre-2012 seasons (no archive) → manual efficiency kept, also leak-free.
                archive_path = kenpom_dir / "archive" / f"kenpom_archive_{s}.parquet"
                kenpom_eff = load_selection_sunday_efficiency(s, archive_path, team_map)
                if len(kenpom_eff):
                    adj_eff = merge_kenpom_efficiency(adj_eff, kenpom_eff)
                    merged += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("KenPom efficiency merge failed for season %d: %s", s, exc)

            try:
                # KP-005: rich ratings-extras (SOS/Luck/APL) come only from the end-of-season
                # `ratings` file and have no leak-free as-of source (the archive lacks them) → drop
                # them (pass ratings_df=None). Height stays (roster-based, y=year, negligible leak).
                height_path = kenpom_dir / f"kenpom_height_{s}.parquet"
                height_df = pd.read_parquet(height_path) if height_path.exists() else None
                kp_frames.append(
                    build_kenpom_rich_features(s, team_map, ratings_df=None, height_df=height_df)
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
    return adj_eff, kp_rich, team_maps


ARCHIVE_DIR = KENPOM_DIR / "archive"
TORVIK_DIR = Path("data/torvik")


def _load_asof_inputs(
    seasons: list[int], team_maps: dict[int, dict], store: DataStore
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load the as-of-date inputs: KenPom archive (men, GM-002) + Torvik women (WM-003) + DayZero.

    Best-effort: any missing cache → an empty frame (→ that join is a no-op). Reuses the KenPom
    ``team_maps`` from :func:`_apply_kenpom`; builds women team maps from WTeams+WTeamSpellings
    against the cached Torvik names (no external call). M/W DayZero are identical, so one dict serves.
    """
    from cbb.features.torvik_asof import load_all_torvik_women  # noqa: PLC0415
    from cbb.kenpom.asof_features import load_all_archive_snapshots  # noqa: PLC0415
    from cbb.kenpom.features import build_team_name_map  # noqa: PLC0415

    dayzero: dict = {}
    try:
        ms = store.load_csv("MSeasons.csv")
        dayzero = dict(zip(ms["Season"], pd.to_datetime(ms["DayZero"])))
    except Exception as exc:  # noqa: BLE001
        log.warning("MSeasons (DayZero) unavailable — as-of join disabled: %s", exc)

    try:
        kp = load_all_archive_snapshots(seasons, ARCHIVE_DIR, team_maps)
    except Exception as exc:  # noqa: BLE001
        log.warning("As-of KenPom snapshots skipped: %s", exc)
        kp = pd.DataFrame()

    # BartTorvik women: build women team maps from the cached Torvik names, then load snapshots.
    tv = pd.DataFrame()
    try:
        w_teams = store.load_csv("WTeams.csv")
        w_spell = store.load_csv("WTeamSpellings.csv")
        w_maps: dict[int, dict] = {}
        for s in seasons:
            p = TORVIK_DIR / f"torvik_women_{s}.parquet"
            if p.exists():
                names = pd.read_parquet(p)[["team"]].drop_duplicates().rename(columns={"team": "TeamName"})
                w_maps[s] = build_team_name_map(w_teams, names, w_spell)
        tv = load_all_torvik_women(seasons, TORVIK_DIR, w_maps)
    except Exception as exc:  # noqa: BLE001
        log.warning("As-of Torvik-women snapshots skipped: %s", exc)

    for label, snaps in (("KenPom", kp), ("Torvik-women", tv)):
        if len(snaps):
            log.info("As-of %s: %d snapshot-rows across %d seasons",
                     label, len(snaps), snaps["Season"].nunique())
    return kp, tv, dayzero


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
    # MTeamSpellings (Kaggle's name-variant table) lifts KenPom→Kaggle mid-major recall ~84%→100%
    # (GM-004) — more teams get KenPom efficiency + as-of features. Best-effort: absent → fuzzy-only.
    try:
        team_spellings = store.load_csv("MTeamSpellings.csv")
    except Exception as exc:  # noqa: BLE001
        log.warning("MTeamSpellings.csv unavailable — name map falls back to fuzzy-only: %s", exc)
        team_spellings = None
    adj_eff, kp_rich, team_maps = _apply_kenpom(
        adj_eff, seasons, data["M_teams"], team_spellings=team_spellings
    )
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

    # ── Regular-season game-level dataset (GM-001, M3 "handicap any game") ───────
    # A *parallel* dataset to matchups.parquet: one row per reg-season game (symmetric A/B),
    # leak-free by construction (pre-game Elo + venue + prior-season priors — no full-season
    # leakage). Best-effort so a failure here never blocks the Kaggle features build. Season
    # 2026 is left in the frame but excluded from training as the trusted reg-season holdout.
    try:
        # As-of-date KenPom (GM-002): load cached weekly archive snapshots (men's, 2012+) and the
        # Season→DayZero map; best-effort — absent snapshots make the join a no-op (Elo+priors only).
        asof_snaps, torvik_women, dayzero = _load_asof_inputs(seasons, team_maps, store)
        reg_games = build_reg_games(data, adj_eff, asof_snapshots=asof_snaps,
                                    dayzero_by_season=dayzero, torvik_women_snapshots=torvik_women)
        store.save_parquet(reg_games, "reg_games.parquet")
        log.info(
            "Reg-season game dataset → %s  (%d rows, %d cols, seasons %d-%d)",
            proc / "reg_games.parquet", len(reg_games), len(reg_games.columns),
            int(reg_games["Season"].min()), int(reg_games["Season"].max()),
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Reg-season game dataset skipped: %s", exc)

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
