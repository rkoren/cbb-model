"""Richer KenPom team features, beyond the AdjOE/AdjDE/AdjEM/AdjTempo efficiency override.

Phase 1 leads with the signal the manually-computed ``season_avgs`` genuinely lacks, and
that is *leak-safe* against a tournament-outcome label:

  - **Height endpoint** — ``AvgHgt, HgtEff`` (roster-based, leak-free). ``season_avgs`` has
    nothing like them. The endpoint's ``Exp/Bench/Continuity`` are *minutes/game-weighted*, so
    they absorb tournament games (a backdoor label leak) and are excluded (KP-006).
  - **Ratings extras** — ``SOS, Luck, APL_Off, APL_Def``. Already present in the cached
    ``kenpom_ratings_{season}.parquet`` (baseline only reads AdjOE/AdjDE/AdjEM/AdjTempo from
    it), so these cost zero extra API calls and are leak-consistent with baseline's existing
    use of those same ratings.

Deliberately EXCLUDED in Phase 1: four-factors / misc-stats / pointdist. Those endpoints
accept only ``y=year`` (no as-of-date), so a past-season pull is tournament-inclusive —
backdoor leakage against a tournament label — and they largely duplicate the leak-free
``season_avgs``. They return in Phase 2 via leak-free as-of-date sourcing.

KenPom is men's only (TeamID < 2000). Women's teams are absent from these frames; downstream
joins leave them NaN and the experiment fills them neutrally.

Pure transforms only — fetching/caching lives in the Prefect experiment, mirroring how
``cbb.kenpom.features`` stays pure while ``experiments/baseline.py`` owns ingestion.
"""

from __future__ import annotations

import logging

import pandas as pd

from .features import _team_name_col

log = logging.getLogger(__name__)

# Source KenPom column -> kp_ prefixed feature name. Only columns actually present in the
# fetched frame are used; missing ones are logged and skipped (KenPom schema can drift).
#
# KP-006: only the *roster-based* height columns are mapped. ``Exp``/``Bench``/``Continuity``
# are minutes/game-weighted on the `height` endpoint (`y=year`, end-of-season), so a team that
# played 6 tournament games carries a different value than a round-1 exit — a backdoor
# tournament-label leak. ``AvgHgt``/``HgtEff`` are roster-shaped and leak-free, so they stay.
HEIGHT_COLS: dict[str, str] = {
    "AvgHgt": "kp_AvgHgt",
    "HgtEff": "kp_HgtEff",
}

RATINGS_EXTRA_COLS: dict[str, str] = {
    "SOS": "kp_SOS",
    "Luck": "kp_Luck",
    "APL_Off": "kp_APL_Off",
    "APL_Def": "kp_APL_Def",
}

# Full set of kp_ feature names this module can emit (for downstream feature-list building).
KP_RICH_FEATURES: list[str] = list(HEIGHT_COLS.values()) + list(RATINGS_EXTRA_COLS.values())


def _select_renamed(df: pd.DataFrame, colmap: dict[str, str], source: str) -> pd.DataFrame:
    """Return df[name_col + present source cols] renamed to kp_ targets, logging absences."""
    name_col = _team_name_col(df)
    present = {src: tgt for src, tgt in colmap.items() if src in df.columns}
    missing = [src for src in colmap if src not in df.columns]
    if missing:
        log.warning("KenPom %s frame missing columns: %s", source, ", ".join(missing))
    out = df[[name_col] + list(present)].rename(columns={name_col: "TeamName", **present})
    return out


def build_kenpom_rich_features(
    season: int,
    team_name_map: dict[str, int],
    ratings_df: pd.DataFrame | None = None,
    height_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a tidy KenPom rich-feature frame keyed to Kaggle TeamIDs for one season.

    Args:
        season: Season year (sets the Season column).
        team_name_map: KenPom team name -> Kaggle TeamID, from
            ``cbb.kenpom.features.build_team_name_map``.
        ratings_df: Raw KenPom ratings frame (the cached ``kenpom_ratings_{season}.parquet``);
            ``SOS/Luck/APL_Off/APL_Def`` are extracted if present.
        height_df: Raw KenPom ``height`` endpoint frame; height/experience/bench/continuity
            are extracted if present.

    Returns:
        DataFrame with columns ``Season, men_women`` (always 0), ``TeamID`` and whichever
        ``kp_*`` features were available. Empty (with just the key columns) if no source
        frame yielded any feature column.
    """
    parts: list[pd.DataFrame] = []
    if ratings_df is not None and len(ratings_df):
        parts.append(_select_renamed(ratings_df, RATINGS_EXTRA_COLS, "ratings"))
    if height_df is not None and len(height_df):
        parts.append(_select_renamed(height_df, HEIGHT_COLS, "height"))

    if not parts:
        return pd.DataFrame(columns=["Season", "men_women", "TeamID"])

    merged = parts[0]
    for nxt in parts[1:]:
        merged = merged.merge(nxt, on="TeamName", how="outer")

    merged["TeamID"] = merged["TeamName"].map(team_name_map)
    merged = merged.dropna(subset=["TeamID"]).copy()
    merged["TeamID"] = merged["TeamID"].astype(int)
    merged["Season"] = season
    merged["men_women"] = 0

    kp_cols = [c for c in merged.columns if c.startswith("kp_")]
    result = merged[["Season", "men_women", "TeamID"] + kp_cols].reset_index(drop=True)
    log.info(
        "KenPom rich features for %d: %d teams, %d feature cols (%s)",
        season, len(result), len(kp_cols), ", ".join(kp_cols),
    )
    return result


def join_kenpom_rich(df: pd.DataFrame, kp_feats: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add ``A_<kp>``, ``B_<kp>`` and ``d_<kp>`` columns to an A/B matchup frame.

    Mirrors how ``build_matchup_dataset`` materialises differentials, and how
    ``experiments/challenger.py`` joins extra feature groups onto ``matchups`` post-hoc.

    Args:
        df: Frame with ``Season, men_women, A_TeamID, B_TeamID`` (matchups or prediction pairs).
        kp_feats: Output of ``build_kenpom_rich_features`` (one or more seasons concatenated).

    Returns:
        ``(joined_df, added_cols)`` where ``added_cols`` is every ``A_/B_/d_`` column added.
    """
    kp_cols = [c for c in kp_feats.columns if c.startswith("kp_")]
    if not kp_cols:
        return df, []

    keys = ["Season", "men_women", "TeamID"]
    a = kp_feats[keys + kp_cols].rename(
        columns={"TeamID": "A_TeamID", **{c: f"A_{c}" for c in kp_cols}}
    )
    b = kp_feats[keys + kp_cols].rename(
        columns={"TeamID": "B_TeamID", **{c: f"B_{c}" for c in kp_cols}}
    )
    df = df.merge(a, on=["Season", "men_women", "A_TeamID"], how="left")
    df = df.merge(b, on=["Season", "men_women", "B_TeamID"], how="left")

    added: list[str] = []
    for c in kp_cols:
        df[f"d_{c}"] = df[f"A_{c}"] - df[f"B_{c}"]
        added += [f"A_{c}", f"B_{c}", f"d_{c}"]
    return df, added
