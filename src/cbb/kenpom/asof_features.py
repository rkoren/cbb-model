"""As-of-date KenPom ratings for the regular-season game model (GM-002).

The reg-season dataset (:mod:`cbb.features.reg_games`) is leak-free but carries only *prior-season*
KenPom efficiency — so early/mid-season games lean on stale priors. This module adds **within-season,
point-in-time** KenPom ratings from the ``archive`` endpoint: for each game, each team's
``AdjEM/AdjOE/AdjDE/AdjTempo`` **as of the most recent snapshot strictly before the game date**.

Leak-freeness rests on two things:
  * **Only the non-``Final`` archive columns.** The archive also returns ``*Final`` (end-of-season)
    plus end-of-season ``Seed``/``Event`` — all of which leak the future; the fetch script keeps
    only the four as-of adj columns + ``TeamName``/``ArchiveDate``.
  * **``allow_exact_matches=False`` in the as-of join.** KenPom snapshots are taken end-of-day,
    *after* that day's games are scored — so a same-day snapshot already contains the game's own
    result. Requiring the snapshot date to be *strictly before* the game date excludes it.

Coverage: the archive begins in **2012** and is **men's only**. Earlier men's seasons, all women's
games, and any game before its season's first snapshot get NaN → the reg model fills them and falls
back on pre-game Elo + prior-season priors. Additive to the ``*_prev`` priors, not a replacement.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Archive as-of column -> as-of feature name. Strictly the non-Final adj columns (see module docstring).
ASOF_COLS: dict[str, str] = {
    "AdjEM": "kp_AdjEM_asof",
    "AdjOE": "kp_AdjOE_asof",
    "AdjDE": "kp_AdjDE_asof",
    "AdjTempo": "kp_AdjTempo_asof",
}
ASOF_FEATURES: list[str] = list(ASOF_COLS.values())

# Earliest season with KenPom archive coverage (verified against the API).
ARCHIVE_FIRST_SEASON = 2012


def load_archive_snapshots(
    season: int, archive_dir: str | Path, team_map: dict[str, int]
) -> pd.DataFrame:
    """Load one season's cached archive snapshots → long as-of frame keyed to Kaggle TeamIDs.

    Reads ``<archive_dir>/kenpom_archive_{season}.parquet`` (stacked weekly snapshots, one
    ``ArchiveDate`` per snapshot), maps ``TeamName`` → Kaggle ``TeamID`` (per-season map, since
    names drift), keeps the four as-of adj columns, and returns
    ``Season, TeamID, ArchiveDate (datetime), kp_*_asof``. Empty frame if the file is absent.
    """
    path = Path(archive_dir) / f"kenpom_archive_{season}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["Season", "TeamID", "ArchiveDate", *ASOF_FEATURES])
    df = pd.read_parquet(path)
    present = {src: tgt for src, tgt in ASOF_COLS.items() if src in df.columns}
    df = df.rename(columns=present)
    df["TeamID"] = df["TeamName"].map(team_map)
    df = df.dropna(subset=["TeamID"]).copy()
    df["TeamID"] = df["TeamID"].astype(int)
    df["Season"] = season
    df["ArchiveDate"] = pd.to_datetime(df["ArchiveDate"])
    return df[["Season", "TeamID", "ArchiveDate", *[c for c in ASOF_FEATURES if c in df.columns]]]


def load_all_archive_snapshots(
    seasons: list[int], archive_dir: str | Path, team_maps: dict[int, dict[str, int]]
) -> pd.DataFrame:
    """Concatenate :func:`load_archive_snapshots` across seasons (skipping ones with no map/file)."""
    parts = [
        load_archive_snapshots(s, archive_dir, team_maps[s])
        for s in seasons
        if s in team_maps and (Path(archive_dir) / f"kenpom_archive_{s}.parquet").exists()
    ]
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=["Season", "TeamID", "ArchiveDate", *ASOF_FEATURES])
    return pd.concat(parts, ignore_index=True)


def game_dates(games: pd.DataFrame, dayzero_by_season: dict[int, "pd.Timestamp"]) -> pd.Series:
    """Kaggle ``DayNum`` → calendar date: ``DayZero[Season] + DayNum days`` (NaT if no DayZero)."""
    dz = games["Season"].map(dayzero_by_season)
    return pd.to_datetime(dz) + pd.to_timedelta(games["DayNum"], unit="D")


def _asof_side(games: pd.DataFrame, snapshots: pd.DataFrame, side: str, feats: list[str]) -> pd.DataFrame:
    """As-of features for one side (A or B): latest snapshot strictly before each game's date."""
    tid = f"{side}_TeamID"
    left = pd.DataFrame({
        "_ord": np.arange(len(games)),
        "Season": games["Season"].to_numpy(),
        tid: games[tid].to_numpy(),
        "game_date": games["game_date"].to_numpy(),
    }).sort_values("game_date")
    right = snapshots.rename(columns={"TeamID": tid}).sort_values("ArchiveDate")
    merged = pd.merge_asof(
        left, right,
        left_on="game_date", right_on="ArchiveDate",
        by=["Season", tid],
        direction="backward",
        allow_exact_matches=False,   # snapshots are end-of-day → exclude same-day (leak)
    )
    return merged.sort_values("_ord")[feats].reset_index(drop=True)


def add_asof_features(
    games: pd.DataFrame,
    snapshots: pd.DataFrame,
    dayzero_by_season: dict[int, "pd.Timestamp"],
) -> tuple[pd.DataFrame, list[str]]:
    """Attach ``A_/B_/d_/s_`` as-of features to the symmetric reg-season game frame.

    Source-generic (KenPom ``kp_*_asof`` men, BartTorvik ``tv_*_asof`` women, WM-003): any column
    in ``snapshots`` ending ``_asof`` is joined. For each side, merge-asof each team's most recent
    snapshot *strictly before* the game date, then form differentials (``d_*`` → margin head) and
    sums (``s_*`` → total head), mirroring the ``*_prev`` priors so the reg model's prefix-based
    feature derivation picks them up unchanged. No-op (``(games, [])``) when no snapshots.
    """
    feats = [c for c in snapshots.columns if c.endswith("_asof")]
    if not feats or snapshots.empty:
        return games, []

    games = games.copy()
    games["game_date"] = game_dates(games, dayzero_by_season)
    a = _asof_side(games, snapshots, "A", feats)
    b = _asof_side(games, snapshots, "B", feats)

    added: list[str] = []
    for c in feats:
        games[f"A_{c}"] = a[c].to_numpy()
        games[f"B_{c}"] = b[c].to_numpy()
        games[f"d_{c}"] = games[f"A_{c}"] - games[f"B_{c}"]
        # NaN propagates (don't fillna(0) per side): a half-sum (real + 0) is a misleading partial
        # signal for the total head; a clean NaN gets uniformly zero-filled by the trainer instead.
        games[f"s_{c}"] = games[f"A_{c}"] + games[f"B_{c}"]
        added += [f"A_{c}", f"B_{c}", f"d_{c}", f"s_{c}"]
    games = games.drop(columns=["game_date"])
    return games, added
