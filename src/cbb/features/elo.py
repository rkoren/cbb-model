"""Per-season Elo ratings, reset each season."""

import pandas as pd


def _expected(elo_a: float, elo_b: float, width: float) -> float:
    return 1.0 / (1 + 10 ** ((elo_b - elo_a) / width))


def compute_elo(
    reg_raw: pd.DataFrame,
    men_women_flag: int,
    base_elo: float = 1000.0,
    k_factor: float = 100.0,
    width: float = 400.0,
) -> pd.DataFrame:
    """End-of-season Elo per (Season, TeamID), reset to base_elo each season.

    Args:
        reg_raw: Original W/L game log (not symmetric). Requires columns:
                 Season, DayNum, WTeamID, LTeamID.
        men_women_flag: 0 for men, 1 for women.

    Returns:
        DataFrame with columns: Season, men_women, TeamID, Elo.
    """
    records = []
    for season, grp in reg_raw.sort_values(["Season", "DayNum"]).groupby("Season"):
        elos: dict[int, float] = {}
        for _, row in grp.iterrows():
            w, l = int(row["WTeamID"]), int(row["LTeamID"])
            ew = elos.get(w, base_elo)
            el = elos.get(l, base_elo)
            delta = k_factor * (1 - _expected(ew, el, width))
            elos[w] = ew + delta
            elos[l] = el - delta
        for tid, elo_val in elos.items():
            records.append({"Season": season, "men_women": men_women_flag, "TeamID": tid, "Elo": elo_val})

    return pd.DataFrame(records)
