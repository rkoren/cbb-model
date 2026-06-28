"""Tests for the regular-season game-level dataset (GM-001): pre-game Elo + builder.

The two things that must hold are (1) **leak-freeness** — a game's pre-game Elo depends only on
strictly-earlier games — and (2) **A/B symmetry** — each game yields a winner-as-A row and a
loser-as-A row with consistently oriented Margin/venue.
"""

import pandas as pd
import pytest

from cbb.features.elo import compute_pregame_elo
from cbb.features.reg_games import build_reg_game_dataset


def _reg_raw():
    # 3 games in season 2020, ordered by DayNum. Team 1101 beats 1102, then 1101 beats 1103,
    # then 1102 beats 1103. WLoc varies so venue orientation can be checked.
    return pd.DataFrame({
        "Season": [2020, 2020, 2020],
        "DayNum": [10, 20, 30],
        "WTeamID": [1101, 1101, 1102],
        "LTeamID": [1102, 1103, 1103],
        "WScore": [80, 75, 70],
        "LScore": [60, 70, 68],
        "WLoc": ["H", "N", "A"],
    })


# ── compute_pregame_elo ─────────────────────────────────────────────────────────

def test_first_game_is_base_elo():
    pre = compute_pregame_elo(_reg_raw(), men_women_flag=0, base_elo=1000.0)
    g1 = pre.iloc[0]
    # No prior games and no prior season → both teams open at base_elo.
    assert g1["W_Elo_pre"] == pytest.approx(1000.0)
    assert g1["L_Elo_pre"] == pytest.approx(1000.0)


def test_pregame_elo_excludes_current_game():
    pre = compute_pregame_elo(_reg_raw(), men_women_flag=0, base_elo=1000.0)
    # 1101's pre-game Elo in game 2 must equal its post-game-1 rating (winner rose above base),
    # i.e. it reflects game 1 but NOT game 2 itself.
    g2 = pre[(pre["DayNum"] == 20)].iloc[0]
    assert g2["W_Elo_pre"] > 1000.0  # 1101 won game 1 → entered game 2 above base
    # 1103 has not played before game 2 → still at base.
    assert g2["L_Elo_pre"] == pytest.approx(1000.0)


def test_one_row_per_game():
    pre = compute_pregame_elo(_reg_raw(), men_women_flag=0)
    assert len(pre) == 3
    assert list(pre.columns) == ["Season", "men_women", "DayNum", "WTeamID", "LTeamID", "W_Elo_pre", "L_Elo_pre"]


def test_prior_season_carryover():
    # Two seasons; a team that ended 2019 strong should open 2020 above base (carryover),
    # not reset to base_elo.
    raw = pd.DataFrame({
        "Season": [2019, 2020],
        "DayNum": [10, 10],
        "WTeamID": [1101, 1101],
        "LTeamID": [1102, 1103],
        "WScore": [80, 75], "LScore": [60, 70], "WLoc": ["H", "H"],
    })
    pre = compute_pregame_elo(raw, men_women_flag=0, base_elo=1000.0, carry=0.75)
    g_2020 = pre[pre["Season"] == 2020].iloc[0]
    assert g_2020["W_Elo_pre"] > 1000.0  # 1101 carried its 2019 gain into 2020


def test_carry_zero_is_full_reset():
    raw = pd.DataFrame({
        "Season": [2019, 2020], "DayNum": [10, 10],
        "WTeamID": [1101, 1101], "LTeamID": [1102, 1103],
        "WScore": [80, 75], "LScore": [60, 70], "WLoc": ["H", "H"],
    })
    pre = compute_pregame_elo(raw, men_women_flag=0, base_elo=1000.0, carry=0.0)
    assert pre[pre["Season"] == 2020].iloc[0]["W_Elo_pre"] == pytest.approx(1000.0)


# ── build_reg_game_dataset ──────────────────────────────────────────────────────

def _adj_eff():
    # Prior-season (2019) adj_eff so 2020 games get *_prev priors via the Season-1 join.
    return pd.DataFrame({
        "Season": [2019, 2019, 2019],
        "men_women": [0, 0, 0],
        "TeamID": [1101, 1102, 1103],
        "AdjOE": [115.0, 105.0, 100.0],
        "AdjDE": [95.0, 100.0, 105.0],
        "AdjEM": [20.0, 5.0, -5.0],
        "AdjTempo": [68.0, 70.0, 66.0],
    })


def _build():
    raw = _reg_raw()
    # add a 2019 game so 2020 has a prior season for carryover (not strictly needed here)
    pre = compute_pregame_elo(raw, men_women_flag=0)
    return build_reg_game_dataset(raw, pre, _adj_eff(), men_women=0)


def test_two_rows_per_game():
    games = _build()
    assert len(games) == 2 * len(_reg_raw())  # symmetric


def test_outcome_and_margin_orientation():
    games = _build()
    # Game 1: 1101 beat 1102 80-60. Winner-as-A row: Outcome 1, Margin +20; loser-as-A: 0, -20.
    g1 = games[(games["DayNum"] == 10)]
    win_row = g1[g1["A_TeamID"] == 1101].iloc[0]
    los_row = g1[g1["A_TeamID"] == 1102].iloc[0]
    assert win_row["Outcome"] == 1 and win_row["Margin"] == pytest.approx(20.0)
    assert los_row["Outcome"] == 0 and los_row["Margin"] == pytest.approx(-20.0)
    assert win_row["Total"] == pytest.approx(140.0) and los_row["Total"] == pytest.approx(140.0)


def test_home_court_orientation():
    games = _build()
    g1 = games[(games["DayNum"] == 10)]  # WLoc="H": winner (1101) was home
    assert g1[g1["A_TeamID"] == 1101].iloc[0]["A_home"] == 1   # A=winner=home
    assert g1[g1["A_TeamID"] == 1102].iloc[0]["A_home"] == -1  # A=loser=away
    g2 = games[(games["DayNum"] == 20)]  # WLoc="N": neutral
    assert (g2["A_home"] == 0).all()
    g3 = games[(games["DayNum"] == 30)]  # WLoc="A": winner (1102) was away
    assert g3[g3["A_TeamID"] == 1102].iloc[0]["A_home"] == -1  # A=winner=away
    assert g3[g3["A_TeamID"] == 1103].iloc[0]["A_home"] == 1   # A=loser=home


def test_prior_season_priors_joined():
    games = _build()
    win_row = games[(games["DayNum"] == 10) & (games["A_TeamID"] == 1101)].iloc[0]
    # 1101 prior AdjEM 20, 1102 prior AdjEM 5 → d_AdjEM_prev = 15.
    assert win_row["d_AdjEM_prev"] == pytest.approx(15.0)
    # Total head sum: AdjTempo 68 + 70 = 138.
    assert win_row["s_AdjTempo_prev"] == pytest.approx(138.0)


def test_d_elo_pre_present_and_antisymmetric():
    games = _build()
    g1 = games[games["DayNum"] == 10]
    win = g1[g1["A_TeamID"] == 1101].iloc[0]["d_Elo_pre"]
    los = g1[g1["A_TeamID"] == 1102].iloc[0]["d_Elo_pre"]
    assert win == pytest.approx(-los)  # A/B swap negates the differential
