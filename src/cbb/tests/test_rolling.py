"""Tests for rolling as-of box-score features (GM-003).

Load-bearing: (1) leak-freeness — a game's rolling value uses only strictly-earlier games and
the first MIN_GAMES emit NaN; (2) the DayNum-doubleheader rule — same-day games are simultaneous
(both use the pre-day snapshot, neither sees the other).
"""

import numpy as np
import pandas as pd
import pytest

from cbb.features.rolling import _poss, compute_rolling_boxscore


def _game(daynum, w, l, ws, ls, *, wfga=60, wor=10, wto=12, wfta=20,
          lfga=58, lor=8, lto=14, lfta=16, ot=0, season=2020):
    return {"Season": season, "DayNum": daynum, "WTeamID": w, "LTeamID": l,
            "WScore": ws, "LScore": ls, "WFGA": wfga, "WOR": wor, "WTO": wto, "WFTA": wfta,
            "LFGA": lfga, "LOR": lor, "LTO": lto, "LFTA": lfta, "NumOT": ot}


def test_possessions_formula():
    assert _poss(60, 10, 12, 20) == pytest.approx(60 - 10 + 12 + 0.475 * 20)


def test_first_min_games_are_nan():
    # 1101 plays 3 games; with min_games=2 the first two emit NaN, the third emits a value.
    raw = pd.DataFrame([_game(1, 1101, 1201, 80, 60), _game(2, 1101, 1202, 75, 70),
                        _game(3, 1101, 1203, 70, 68)])
    out = compute_rolling_boxscore(raw, 0, min_games=2)
    w = out[out.WTeamID == 1101].sort_values("DayNum")
    assert np.isnan(w.iloc[0].W_bs_OE_asof)   # game 1: 0 prior games
    assert np.isnan(w.iloc[1].W_bs_OE_asof)   # game 2: 1 prior game (< min 2)
    assert np.isfinite(w.iloc[2].W_bs_OE_asof)  # game 3: 2 prior games → emits


def test_emit_excludes_current_game():
    # After two identical games, the 3rd game's as-of OE reflects only games 1–2, not game 3.
    raw = pd.DataFrame([_game(1, 1101, 1201, 80, 60), _game(2, 1101, 1202, 80, 60),
                        _game(3, 1101, 1203, 999, 0)])  # blowout that must NOT affect the emit
    out = compute_rolling_boxscore(raw, 0, min_games=2)
    row = out[(out.WTeamID == 1101) & (out.DayNum == 3)].iloc[0]
    poss = _poss(60, 10, 12, 20)
    assert row.W_bs_OE_asof == pytest.approx(100 * 80 / poss)   # from games 1–2 only
    assert row.W_bs_NetEff_asof == pytest.approx(row.W_bs_OE_asof - row.W_bs_DE_asof)


def test_daynum_doubleheader_uses_pre_day_snapshot():
    # 1101 builds 2 games of history (DayNum 1,2), then plays TWO games on DayNum 3.
    # Both DayNum-3 emits must use the same pre-day state — neither sees the other's result.
    raw = pd.DataFrame([
        _game(1, 1101, 1201, 80, 60), _game(2, 1101, 1202, 80, 60),
        _game(3, 1101, 1203, 100, 50),   # doubleheader game A
        _game(3, 1101, 1204, 40, 90),    # doubleheader game B (would swing OE if it leaked)
    ])
    out = compute_rolling_boxscore(raw, 0, min_games=2)
    d3 = out[(out.WTeamID == 1101) & (out.DayNum == 3)]
    assert len(d3) == 2
    assert d3.W_bs_OE_asof.nunique() == 1  # identical → both used the pre-DayNum-3 snapshot


def test_ot_normalizes_tempo():
    # Same box line, one game OT: OT game's possessions divide by (40+5)/40 → lower per-40 tempo.
    base = [_game(1, 1101, 1201, 80, 60), _game(2, 1101, 1202, 80, 60)]
    reg = compute_rolling_boxscore(pd.DataFrame(base + [_game(3, 1101, 1203, 70, 68)]), 0, min_games=2)
    otg = compute_rolling_boxscore(pd.DataFrame(
        [_game(1, 1101, 1201, 80, 60, ot=1), _game(2, 1101, 1202, 80, 60, ot=1),
         _game(3, 1101, 1203, 70, 68)]), 0, min_games=2)
    t_reg = reg[(reg.WTeamID == 1101) & (reg.DayNum == 3)].iloc[0].W_bs_Tempo_asof
    t_ot = otg[(otg.WTeamID == 1101) & (otg.DayNum == 3)].iloc[0].W_bs_Tempo_asof
    assert t_ot < t_reg   # OT possessions spread over 45 min → lower per-40 pace


def test_both_teams_emitted_per_game():
    raw = pd.DataFrame([_game(1, 1101, 1201, 80, 60)])
    out = compute_rolling_boxscore(raw, 1, min_games=2)
    assert set(["W_bs_Tempo_asof", "L_bs_OE_asof", "men_women"]).issubset(out.columns)
    assert out.iloc[0].men_women == 1
