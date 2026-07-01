"""BartTorvik women's as-of ratings loader (WM-003) — the free women's KenPom-analog.

Women have no KenPom and no Kaggle Massey; BartTorvik (barttorvik.com/ncaaw) publishes free
opponent-adjusted AdjOE/AdjDE/AdjTempo with daily as-of snapshots (fetched by
``scripts/fetch_torvik_women.py`` → ``data/torvik/torvik_women_{season}.parquet``). This loads them
into the same long shape the KenPom archive uses, so they flow through the shared
:func:`cbb.kenpom.asof_features.add_asof_features` pipeline (→ ``d_tv_*_asof``/``s_tv_*_asof``).

As-of (timemachine) coverage is **women 2025–2026**; earlier seasons fall back on rolling box-score
(GM-003) + priors. Torvik ratings match KenPom for men at corr 0.996–0.998, so the women's numbers
are trustworthy.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# BartTorvik ratings → as-of feature names (AdjEM derived = AdjOE − AdjDE).
TV_FEATURES = ["tv_AdjEM_asof", "tv_AdjOE_asof", "tv_AdjDE_asof", "tv_AdjTempo_asof"]
TORVIK_FIRST_ASOF_SEASON = 2025  # daily timemachine snapshots start here (season-final JSON goes further)


def load_torvik_women_snapshots(
    season: int, torvik_dir: str | Path, team_map: dict[str, int]
) -> pd.DataFrame:
    """Load one season's cached Torvik women snapshots → long as-of frame keyed to Kaggle TeamIDs."""
    path = Path(torvik_dir) / f"torvik_women_{season}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["Season", "TeamID", "ArchiveDate", *TV_FEATURES])
    df = pd.read_parquet(path)
    df["TeamID"] = df["team"].map(team_map)
    df = df.dropna(subset=["TeamID"]).copy()
    df["TeamID"] = df["TeamID"].astype(int)
    df["Season"] = season
    df["ArchiveDate"] = pd.to_datetime(df["ArchiveDate"])
    df["tv_AdjEM_asof"] = df["tv_AdjOE"] - df["tv_AdjDE"]
    df["tv_AdjOE_asof"] = df["tv_AdjOE"]
    df["tv_AdjDE_asof"] = df["tv_AdjDE"]
    df["tv_AdjTempo_asof"] = df["tv_AdjTempo"]
    return df[["Season", "TeamID", "ArchiveDate", *TV_FEATURES]]


def load_all_torvik_women(
    seasons: list[int], torvik_dir: str | Path, team_maps: dict[int, dict[str, int]]
) -> pd.DataFrame:
    """Concatenate :func:`load_torvik_women_snapshots` across seasons (skip ones with no map/file)."""
    parts = [
        load_torvik_women_snapshots(s, torvik_dir, team_maps[s])
        for s in seasons
        if s in team_maps and (Path(torvik_dir) / f"torvik_women_{s}.parquet").exists()
    ]
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=["Season", "TeamID", "ArchiveDate", *TV_FEATURES])
    return pd.concat(parts, ignore_index=True)
