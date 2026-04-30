"""KenPom efficiency integration: team name mapping and season efficiency loading.

KenPom only covers men's basketball. Women's efficiency is always computed
manually from Kaggle box scores and flows through unchanged.
"""

from __future__ import annotations

import difflib
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# KenPom API column names — verify against actual API response on first run.
# The client returns raw JSON as a DataFrame; column names reflect KenPom's API schema.
_KP_EFFICIENCY_COLS = ["AdjOE", "AdjDE", "AdjEM", "AdjTempo"]

# Known KenPom name → Kaggle TeamName mismatches.
_NAME_OVERRIDES: dict[str, str] = {
    "N.C. State": "NC State",
    "Saint Mary's (CA)": "St Mary's CA",
    "Saint Joseph's": "St Joseph's PA",
    "Detroit Mercy": "Detroit Mercy",
    "Penn": "Pennsylvania",
    "Loyola Chicago": "Loyola Chicago",
    "Louisiana": "Louisiana Lafayette",
    "UTSA": "UT San Antonio",
    "Little Rock": "Ark Little Rock",
    "SIU Edwardsville": "SIU Edwardsvlle",
    "USC Upstate": "South Carolina Up",
    "St. Francis (NY)": "St Francis NY",
    "St. Francis (PA)": "St Francis PA",
    "St. Thomas": "St Thomas MN",
    "UT Rio Grande Valley": "UT Rio Grande Vly",
}

_MATCH_THRESHOLD = 0.85


def build_team_name_map(
    mteams_df: pd.DataFrame,
    kenpom_teams_df: pd.DataFrame,
) -> dict[str, int]:
    """Build a mapping from KenPom team name to Kaggle TeamID.

    Applies manual overrides first, then exact match, then fuzzy match
    (SequenceMatcher threshold=0.85). Logs unmatched teams as warnings.

    Args:
        mteams_df: Kaggle MTeams.csv DataFrame — needs TeamID and TeamName columns.
        kenpom_teams_df: DataFrame from KenPomClient.teams() for any season.

    Returns:
        dict mapping KenPom team name → Kaggle TeamID.
    """
    kaggle_names: dict[str, int] = dict(zip(mteams_df["TeamName"], mteams_df["TeamID"]))
    kaggle_name_list = list(kaggle_names.keys())
    kp_name_col = _team_name_col(kenpom_teams_df)

    result: dict[str, int] = {}
    unmatched: list[str] = []

    for kp_name in kenpom_teams_df[kp_name_col].dropna().unique():
        canonical = _NAME_OVERRIDES.get(kp_name, kp_name)

        if canonical in kaggle_names:
            result[kp_name] = kaggle_names[canonical]
            continue

        matches = difflib.get_close_matches(canonical, kaggle_name_list, n=1, cutoff=_MATCH_THRESHOLD)
        if matches:
            result[kp_name] = kaggle_names[matches[0]]
        else:
            unmatched.append(kp_name)

    if unmatched:
        log.warning(
            "KenPom teams with no Kaggle match (%d): %s",
            len(unmatched),
            ", ".join(sorted(unmatched)),
        )

    log.info("Team name map built: %d matched, %d unmatched", len(result), len(unmatched))
    return result


def load_kenpom_efficiency(
    season: int,
    kenpom_parquet_path: Path,
    team_name_map: dict[str, int],
) -> pd.DataFrame:
    """Load a saved KenPom ratings parquet and return efficiency keyed to Kaggle TeamIDs.

    Args:
        season: Season year (e.g. 2026) — sets the Season column.
        kenpom_parquet_path: Path to kenpom_ratings_{season}.parquet.
        team_name_map: KenPom name → Kaggle TeamID from build_team_name_map().

    Returns:
        DataFrame with columns: Season, men_women (always 0), TeamID, and whichever
        of AdjOE/AdjDE/AdjEM/AdjTempo are present in the parquet.
    """
    df = pd.read_parquet(kenpom_parquet_path)
    kp_name_col = _team_name_col(df)

    eff_cols = [c for c in _KP_EFFICIENCY_COLS if c in df.columns]
    if not eff_cols:
        raise ValueError(
            f"KenPom parquet at {kenpom_parquet_path} has none of the expected efficiency "
            f"columns {_KP_EFFICIENCY_COLS}. Found: {df.columns.tolist()}"
        )

    df = df.copy()
    df["TeamID"] = df[kp_name_col].map(team_name_map)
    df = df.dropna(subset=["TeamID"])
    df["TeamID"] = df["TeamID"].astype(int)
    df["Season"] = season
    df["men_women"] = 0

    return df[["Season", "men_women", "TeamID"] + eff_cols].reset_index(drop=True)


def merge_kenpom_efficiency(
    adj_eff_df: pd.DataFrame,
    kenpom_eff_df: pd.DataFrame,
) -> pd.DataFrame:
    """Replace men's efficiency in adj_eff_df with official KenPom values.

    Women's rows (men_women == 1) and men's teams absent from KenPom data
    are preserved from the manually computed values unchanged.

    Args:
        adj_eff_df: Output of compute_adj_efficiency() — flat DataFrame with
                    Season, men_women, TeamID, AdjOE, AdjDE, AdjEM, AdjTempo.
        kenpom_eff_df: Output of load_kenpom_efficiency().

    Returns:
        Copy of adj_eff_df with men's efficiency overwritten where KenPom data exists.
    """
    eff_cols = [c for c in _KP_EFFICIENCY_COLS if c in adj_eff_df.columns and c in kenpom_eff_df.columns]

    merged = adj_eff_df.merge(
        kenpom_eff_df[["Season", "men_women", "TeamID"] + eff_cols].rename(
            columns={c: f"{c}_kp" for c in eff_cols}
        ),
        on=["Season", "men_women", "TeamID"],
        how="left",
    )

    for col in eff_cols:
        kp_col = f"{col}_kp"
        mask = merged[kp_col].notna()
        merged.loc[mask, col] = merged.loc[mask, kp_col]
        merged.drop(columns=[kp_col], inplace=True)

    n_replaced = kenpom_eff_df["TeamID"].nunique()
    log.info("KenPom efficiency merged: %d men's teams updated", n_replaced)
    return merged


def _team_name_col(df: pd.DataFrame) -> str:
    for candidate in ["TeamName", "team_name", "Team", "team", "name"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Cannot find team name column. Available: {df.columns.tolist()}")
