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


def compute_pregame_elo(
    reg_raw: pd.DataFrame,
    men_women_flag: int,
    base_elo: float = 1000.0,
    k_factor: float = 100.0,
    width: float = 400.0,
    carry: float = 0.75,
) -> pd.DataFrame:
    """Point-in-time (pre-game) Elo for every regular-season game — leak-free by construction.

    Walks games in (Season, DayNum) order maintaining a running rating per team and emits, for
    each game, both teams' Elo *as of just before that game* — so a game's features never depend
    on its own result or any later game. Unlike :func:`compute_elo` (which resets to ``base_elo``
    every season and returns only the end-of-season value), each new season seeds a team from its
    prior-season-end Elo regressed toward the mean — ``base_elo + carry * (prev_end - base_elo)``
    — so early-season games carry real signal instead of a cold ``base_elo``-for-everyone.

    This is the leak-free strength backbone of the regular-season game-level dataset (GM-001);
    the tournament path keeps using :func:`compute_elo` (end-of-season is fine post-tournament).

    Args:
        reg_raw: Original W/L game log. Requires columns Season, DayNum, WTeamID, LTeamID.
        men_women_flag: 0 for men, 1 for women.
        carry: Fraction of the prior-season-end deviation from the mean carried into the next
               season's opening rating (0 = full reset, 1 = no regression). Default 0.75.

    Returns:
        One row per raw game (aligned to reg_raw's game order): Season, men_women, DayNum,
        WTeamID, LTeamID, W_Elo_pre, L_Elo_pre.
    """
    prev_end: dict[int, float] = {}
    rows: list[tuple] = []
    for season, grp in reg_raw.sort_values(["Season", "DayNum"]).groupby("Season"):
        elos: dict[int, float] = {}
        for r in grp.itertuples(index=False):
            w, l = int(r.WTeamID), int(r.LTeamID)
            if w not in elos:
                elos[w] = base_elo + carry * (prev_end[w] - base_elo) if w in prev_end else base_elo
            if l not in elos:
                elos[l] = base_elo + carry * (prev_end[l] - base_elo) if l in prev_end else base_elo
            ew, el = elos[w], elos[l]
            rows.append((int(season), men_women_flag, int(r.DayNum), w, l, ew, el))
            delta = k_factor * (1 - _expected(ew, el, width))
            elos[w] = ew + delta
            elos[l] = el - delta
        prev_end.update(elos)

    return pd.DataFrame(
        rows,
        columns=["Season", "men_women", "DayNum", "WTeamID", "LTeamID", "W_Elo_pre", "L_Elo_pre"],
    )
