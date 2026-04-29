"""Season box-score averages and Four Factors for offense and defense."""

import numpy as np
import pandas as pd

_FT_FACTOR = 0.475

_BOX_COLS = ["Score", "FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA", "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]


def compute_season_averages(reg_sym: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-team-season box score means from symmetric game log.

    Args:
        reg_sym: Symmetric regular-season games. Expects T1_*/T2_* column prefixes
                 for all _BOX_COLS plus a PointDiff column.

    Returns:
        DataFrame indexed by (men_women, Season, TeamID) with avg_* and avg_opp_* columns.
    """
    t1_cols = [f"T1_{c}" for c in _BOX_COLS] + ["PointDiff"]
    t2_cols = [f"T2_{c}" for c in _BOX_COLS]

    own = (
        reg_sym.groupby(["men_women", "Season", "T1_TeamID"])[t1_cols]
        .mean()
        .rename(columns=lambda c: c.replace("T1_", "avg_"))
        .rename(columns={"PointDiff": "avg_PointDiff"})
        .reset_index()
        .rename(columns={"T1_TeamID": "TeamID"})
    )

    opp = (
        reg_sym.groupby(["men_women", "Season", "T1_TeamID"])[t2_cols]
        .mean()
        .rename(columns=lambda c: c.replace("T2_", "avg_opp_"))
        .reset_index()
        .rename(columns={"T1_TeamID": "TeamID"})
    )

    return own.merge(opp, on=["men_women", "Season", "TeamID"])


def add_four_factors(season_avgs: pd.DataFrame) -> pd.DataFrame:
    """Compute Four Factors and extended shooting/style stats from season averages.

    Adds columns in-place (returns a copy). All rate stats use clipped denominators
    to avoid divide-by-zero on teams with unusual game logs.

    Args:
        season_avgs: Output of compute_season_averages().

    Returns:
        season_avgs with additional rate columns appended.
    """
    s = season_avgs.copy()

    s["poss"] = s["avg_FGA"] - s["avg_OR"] + s["avg_TO"] + _FT_FACTOR * s["avg_FTA"]
    s["opp_poss"] = s["avg_opp_FGA"] - s["avg_opp_OR"] + s["avg_opp_TO"] + _FT_FACTOR * s["avg_opp_FTA"]

    # Four Factors — offense
    s["eFGpct"] = (s["avg_FGM"] + 0.5 * s["avg_FGM3"]) / s["avg_FGA"].clip(lower=1)
    s["TOpct"] = s["avg_TO"] / s["poss"].clip(lower=1)
    s["ORpct"] = s["avg_OR"] / (s["avg_OR"] + s["avg_opp_DR"]).clip(lower=1)
    s["FTrate"] = s["avg_FTA"] / s["avg_FGA"].clip(lower=1)
    s["TSpct"] = s["avg_Score"] / (2 * (s["avg_FGA"] + _FT_FACTOR * s["avg_FTA"])).clip(lower=1)

    # Four Factors — defense
    s["opp_eFGpct"] = (s["avg_opp_FGM"] + 0.5 * s["avg_opp_FGM3"]) / s["avg_opp_FGA"].clip(lower=1)
    s["opp_TOpct"] = s["avg_opp_TO"] / s["opp_poss"].clip(lower=1)
    s["opp_ORpct"] = s["avg_opp_OR"] / (s["avg_opp_OR"] + s["avg_DR"]).clip(lower=1)
    s["opp_FTrate"] = s["avg_opp_FTA"] / s["avg_opp_FGA"].clip(lower=1)
    s["opp_TSpct"] = s["avg_opp_Score"] / (2 * (s["avg_opp_FGA"] + _FT_FACTOR * s["avg_opp_FTA"])).clip(lower=1)

    # Shooting percentages
    s["FG3pct"] = s["avg_FGM3"] / s["avg_FGA3"].clip(lower=1)
    s["opp_FG3pct"] = s["avg_opp_FGM3"] / s["avg_opp_FGA3"].clip(lower=1)
    s["FG2pct"] = (s["avg_FGM"] - s["avg_FGM3"]) / (s["avg_FGA"] - s["avg_FGA3"]).clip(lower=1)
    s["opp_FG2pct"] = (s["avg_opp_FGM"] - s["avg_opp_FGM3"]) / (s["avg_opp_FGA"] - s["avg_opp_FGA3"]).clip(lower=1)
    s["FTpct"] = s["avg_FTM"] / s["avg_FTA"].clip(lower=1)
    s["opp_FTpct"] = s["avg_opp_FTM"] / s["avg_opp_FTA"].clip(lower=1)

    # Style
    s["3PArate"] = s["avg_FGA3"] / s["avg_FGA"].clip(lower=1)
    s["opp_3PArate"] = s["avg_opp_FGA3"] / s["avg_opp_FGA"].clip(lower=1)
    s["ASTrate"] = s["avg_Ast"] / s["avg_FGM"].clip(lower=1)
    s["opp_ASTrate"] = s["avg_opp_Ast"] / s["avg_opp_FGM"].clip(lower=1)

    # Defensive actions
    s["Blkpct"] = s["avg_Blk"] / (s["avg_opp_FGA"] - s["avg_opp_FGA3"]).clip(lower=1)
    s["Stlpct"] = s["avg_Stl"] / s["opp_poss"].clip(lower=1)

    # Non-steal turnovers: TOs not attributable to opponent steals
    s["NonStlTOpct"] = (s["avg_TO"] - s["avg_opp_Stl"]).clip(lower=0) / s["poss"].clip(lower=1)
    s["opp_NonStlTOpct"] = (s["avg_opp_TO"] - s["avg_Stl"]).clip(lower=0) / s["opp_poss"].clip(lower=1)

    # Point distribution
    s["pct_pts_3"] = 3 * s["avg_FGM3"] / s["avg_Score"].clip(lower=1)
    s["pct_pts_2"] = 2 * (s["avg_FGM"] - s["avg_FGM3"]) / s["avg_Score"].clip(lower=1)
    s["pct_pts_ft"] = s["avg_FTM"] / s["avg_Score"].clip(lower=1)
    s["opp_pct_pts_3"] = 3 * s["avg_opp_FGM3"] / s["avg_opp_Score"].clip(lower=1)
    s["opp_pct_pts_2"] = 2 * (s["avg_opp_FGM"] - s["avg_opp_FGM3"]) / s["avg_opp_Score"].clip(lower=1)
    s["opp_pct_pts_ft"] = s["avg_opp_FTM"] / s["avg_opp_Score"].clip(lower=1)

    return s
