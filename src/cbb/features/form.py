"""Recent-form win percentage from the tail end of the regular season."""

import pandas as pd


def compute_recent_form(
    reg_sym: pd.DataFrame,
    daynum_cutoff: int = 103,
) -> pd.DataFrame:
    """Win ratio over games played after daynum_cutoff.

    Captures sustained late-season performance rather than a single hot week.
    DayNum 103 is roughly 30 days before Selection Sunday; longer windows were
    tested and showed diminishing returns compared to full-season averages.

    Args:
        reg_sym: Symmetric regular-season game log. Requires: men_women, Season,
                 DayNum, T1_TeamID, Outcome (1 = T1 won).
        daynum_cutoff: Only games after this DayNum are included.

    Returns:
        DataFrame with columns: men_women, Season, TeamID, RecentWinPct, RecentGames.
    """
    recent = reg_sym[reg_sym["DayNum"] > daynum_cutoff].copy()
    recent["Win"] = (recent["Outcome"] == 1).astype(int)

    return (
        recent.groupby(["men_women", "Season", "T1_TeamID"])
        .agg(RecentWinPct=("Win", "mean"), RecentGames=("Win", "count"))
        .reset_index()
        .rename(columns={"T1_TeamID": "TeamID"})
    )
