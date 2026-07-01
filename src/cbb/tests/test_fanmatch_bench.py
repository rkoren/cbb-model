"""Tests for the FanMatch benchmark core (GM-004).

The load-bearing properties: home-perspective orientation (a home win → positive realized margin
matched to FanMatch's home-minus-visitor prediction), per-stage drop counting, and identical
scoring for both predictors.
"""

import numpy as np
import pandas as pd
import pytest

from cbb.benchmark.fanmatch_bench import (
    fanmatch_predictions,
    match_fanmatch_to_results,
    score_predictions,
)

DZ = pd.Timestamp("2025-11-01")  # DayNum 10 → 2025-11-11
TMAP = {"Duke": 1101, "Kansas": 1102, "Iowa": 1103}


def _fanmatch(rows):
    return pd.DataFrame(rows)


def _results(rows):
    return pd.DataFrame(rows)


def test_fanmatch_predictions_home_perspective():
    fm = fanmatch_predictions(_fanmatch([
        {"Home": "Duke", "Visitor": "Kansas", "HomePred": 80, "VisitorPred": 70, "HomeWP": 75},
    ]))
    assert fm["fm_margin"].iloc[0] == 10        # home − visitor
    assert fm["fm_total"].iloc[0] == 150
    assert fm["fm_home_wp"].iloc[0] == pytest.approx(0.75)  # 0–100 → 0–1


def test_match_home_win_orientation():
    fm = _fanmatch([{"DateOfGame": "2025-11-11", "Home": "Duke", "Visitor": "Kansas",
                     "HomePred": 80, "VisitorPred": 70, "HomeWP": 75}])
    res = _results([{"DayNum": 10, "WTeamID": 1101, "LTeamID": 1102,  # Duke (home) won
                     "WScore": 78, "LScore": 72, "WLoc": "H"}])
    matched, counts = match_fanmatch_to_results(fm, res, TMAP, DZ)
    row = matched.iloc[0]
    assert row["home_won"] == 1
    assert row["act_margin"] == 6      # home won by 6 → positive (matches fm_margin sign convention)
    assert row["act_total"] == 150
    assert not row["neutral"]


def test_match_home_loss_orientation():
    fm = _fanmatch([{"DateOfGame": "2025-11-11", "Home": "Duke", "Visitor": "Kansas",
                     "HomePred": 80, "VisitorPred": 70, "HomeWP": 75}])
    res = _results([{"DayNum": 10, "WTeamID": 1102, "LTeamID": 1101,  # Kansas (visitor) won
                     "WScore": 80, "LScore": 75, "WLoc": "A"}])
    matched, _ = match_fanmatch_to_results(fm, res, TMAP, DZ)
    row = matched.iloc[0]
    assert row["home_won"] == 0
    assert row["act_margin"] == -5     # home lost by 5 → negative (home perspective)


def test_match_neutral_flag():
    fm = _fanmatch([{"DateOfGame": "2025-11-11", "Home": "Duke", "Visitor": "Kansas",
                     "HomePred": 80, "VisitorPred": 70, "HomeWP": 75}])
    res = _results([{"DayNum": 10, "WTeamID": 1101, "LTeamID": 1102,
                     "WScore": 78, "LScore": 72, "WLoc": "N"}])
    matched, _ = match_fanmatch_to_results(fm, res, TMAP, DZ)
    assert bool(matched.iloc[0]["neutral"]) is True


def test_unmapped_name_dropped_with_count():
    fm = _fanmatch([
        {"DateOfGame": "2025-11-11", "Home": "Duke", "Visitor": "Kansas",
         "HomePred": 80, "VisitorPred": 70, "HomeWP": 75},
        {"DateOfGame": "2025-11-11", "Home": "Duke", "Visitor": "NobodyU",  # unmapped
         "HomePred": 80, "VisitorPred": 70, "HomeWP": 75},
    ])
    res = _results([{"DayNum": 10, "WTeamID": 1101, "LTeamID": 1102,
                     "WScore": 78, "LScore": 72, "WLoc": "H"}])
    matched, counts = match_fanmatch_to_results(fm, res, TMAP, DZ)
    assert counts["fanmatch"] == 2 and counts["names_mapped"] == 1
    assert counts["matched_to_actual"] == 1


def test_unplayed_game_dropped():
    # Mapped names but no matching actual result (e.g. cancelled) → dropped at the join.
    fm = _fanmatch([{"DateOfGame": "2025-11-11", "Home": "Duke", "Visitor": "Iowa",
                     "HomePred": 80, "VisitorPred": 70, "HomeWP": 75}])
    res = _results([{"DayNum": 10, "WTeamID": 1101, "LTeamID": 1102,  # Duke-Kansas, not Duke-Iowa
                     "WScore": 78, "LScore": 72, "WLoc": "H"}])
    matched, counts = match_fanmatch_to_results(fm, res, TMAP, DZ)
    assert counts["names_mapped"] == 1 and counts["matched_to_actual"] == 0


def test_score_predictions_arithmetic():
    matched = pd.DataFrame({
        "act_margin": [10.0, -4.0], "act_total": [150.0, 130.0], "home_won": [1, 0],
    })
    out = score_predictions(matched, margin=np.array([7.0, -2.0]),
                            total=np.array([148.0, 134.0]), home_wp=np.array([0.8, 0.3]))
    assert out["margin_mae"] == pytest.approx((3 + 2) / 2)
    assert out["total_mae"] == pytest.approx((2 + 4) / 2)
    assert out["brier"] == pytest.approx(((0.8 - 1) ** 2 + (0.3 - 0) ** 2) / 2)
    assert out["n"] == 2
