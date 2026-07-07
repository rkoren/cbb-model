"""Tests for the unified slate matcher (DASH-001b).

Load-bearing properties: the comparator re-orients correctly into our A/B convention (the
bug-prone core — get it wrong and every us-vs-KenPom delta is silently corrupted), the join keys
on the *unordered* pair + date, and both predictors score against the log's own actuals.
"""

import pandas as pd
import pytest

from cbb.benchmark.slate import (
    COMPARATOR_COLS,
    fanmatch_to_comparator,
    match_comparator_to_log,
    score_slate,
)

DZ = pd.Timestamp("2025-11-01")  # DayNum 10 → 2025-11-11
TMAP = {"Duke": 1101, "Kansas": 1102, "Iowa": 1103}


def _log(rows):
    """A DASH-001a-shaped predictions log slice (A perspective)."""
    return pd.DataFrame(rows)


def _comparator(rows):
    return pd.DataFrame(rows, columns=COMPARATOR_COLS)


# ── fanmatch_to_comparator ────────────────────────────────────────────────────────

def test_fanmatch_adapter_shapes_to_contract():
    fm = pd.DataFrame([{"DateOfGame": "2025-11-11", "Home": "Duke", "Visitor": "Kansas",
                        "HomePred": 80, "VisitorPred": 70, "HomeWP": 75}])
    cmp = fanmatch_to_comparator(fm, TMAP, DZ, season=2026)
    assert list(cmp.columns) == COMPARATOR_COLS
    row = cmp.iloc[0]
    assert row["Season"] == 2026 and row["DayNum"] == 10
    assert row["home_id"] == 1101 and row["vis_id"] == 1102
    assert row["cmp_margin"] == 10 and row["cmp_total"] == 150
    assert row["cmp_home_wp"] == pytest.approx(0.75)  # 0–100 → 0–1


def test_fanmatch_adapter_drops_unmapped():
    fm = pd.DataFrame([
        {"DateOfGame": "2025-11-11", "Home": "Duke", "Visitor": "Kansas",
         "HomePred": 80, "VisitorPred": 70, "HomeWP": 75},
        {"DateOfGame": "2025-11-11", "Home": "Duke", "Visitor": "NobodyU",
         "HomePred": 80, "VisitorPred": 70, "HomeWP": 75},
    ])
    cmp = fanmatch_to_comparator(fm, TMAP, DZ, season=2026)
    assert len(cmp) == 1


# ── match_comparator_to_log: orientation (the core) ────────────────────────────────

def test_match_no_flip_when_home_is_A():
    # Our A == comparator home → comparator carries straight through.
    log = _log([{"Season": 2026, "DayNum": 10, "A_TeamID": 1101, "B_TeamID": 1102,
                 "pred_margin": 6.0, "pred_total": 148.0, "pred_prob": 0.7,
                 "Margin": 6.0, "Total": 150.0, "Outcome": 1}])
    cmp = _comparator([{"Season": 2026, "DayNum": 10, "home_id": 1101, "vis_id": 1102,
                        "cmp_margin": 10.0, "cmp_total": 150.0, "cmp_home_wp": 0.75}])
    matched, counts = match_comparator_to_log(log, cmp)
    row = matched.iloc[0]
    assert row["cmp_margin"] == 10.0
    assert row["cmp_prob"] == pytest.approx(0.75)
    assert row["cmp_total"] == 150.0
    assert counts == {"log": 1, "comparator": 1, "matched": 1}


def test_match_flips_when_home_is_B():
    # Our A == comparator *visitor* → margin sign flips, prob becomes 1 − home_wp, total invariant.
    log = _log([{"Season": 2026, "DayNum": 10, "A_TeamID": 1102, "B_TeamID": 1101,
                 "pred_margin": -6.0, "pred_total": 148.0, "pred_prob": 0.3,
                 "Margin": -6.0, "Total": 150.0, "Outcome": 0}])
    cmp = _comparator([{"Season": 2026, "DayNum": 10, "home_id": 1101, "vis_id": 1102,
                        "cmp_margin": 10.0, "cmp_total": 150.0, "cmp_home_wp": 0.75}])
    matched, _ = match_comparator_to_log(log, cmp)
    row = matched.iloc[0]
    assert row["cmp_margin"] == -10.0                  # flipped to A perspective
    assert row["cmp_prob"] == pytest.approx(0.25)      # 1 − 0.75
    assert row["cmp_total"] == 150.0                   # unchanged


def test_match_keys_on_unordered_pair():
    # Log lists the pair as (1102, 1101); comparator as home=1101/vis=1102 — still matches.
    log = _log([{"Season": 2026, "DayNum": 10, "A_TeamID": 1102, "B_TeamID": 1101,
                 "pred_margin": -6.0, "pred_total": 148.0, "pred_prob": 0.3,
                 "Margin": -6.0, "Total": 150.0, "Outcome": 0}])
    cmp = _comparator([{"Season": 2026, "DayNum": 10, "home_id": 1101, "vis_id": 1102,
                        "cmp_margin": 10.0, "cmp_total": 150.0, "cmp_home_wp": 0.75}])
    matched, counts = match_comparator_to_log(log, cmp)
    assert counts["matched"] == 1


def test_match_drops_log_row_without_comparator():
    log = _log([
        {"Season": 2026, "DayNum": 10, "A_TeamID": 1101, "B_TeamID": 1102,
         "pred_margin": 6.0, "pred_total": 148.0, "pred_prob": 0.7,
         "Margin": 6.0, "Total": 150.0, "Outcome": 1},
        {"Season": 2026, "DayNum": 11, "A_TeamID": 1101, "B_TeamID": 1103,  # no comparator
         "pred_margin": 4.0, "pred_total": 140.0, "pred_prob": 0.6,
         "Margin": 4.0, "Total": 138.0, "Outcome": 1},
    ])
    cmp = _comparator([{"Season": 2026, "DayNum": 10, "home_id": 1101, "vis_id": 1102,
                        "cmp_margin": 10.0, "cmp_total": 150.0, "cmp_home_wp": 0.75}])
    matched, counts = match_comparator_to_log(log, cmp)
    assert counts == {"log": 2, "comparator": 1, "matched": 1}


def test_match_does_not_cross_seasons():
    # Same pair + DayNum but a different season must not match.
    log = _log([{"Season": 2025, "DayNum": 10, "A_TeamID": 1101, "B_TeamID": 1102,
                 "pred_margin": 6.0, "pred_total": 148.0, "pred_prob": 0.7,
                 "Margin": 6.0, "Total": 150.0, "Outcome": 1}])
    cmp = _comparator([{"Season": 2026, "DayNum": 10, "home_id": 1101, "vis_id": 1102,
                        "cmp_margin": 10.0, "cmp_total": 150.0, "cmp_home_wp": 0.75}])
    _, counts = match_comparator_to_log(log, cmp)
    assert counts["matched"] == 0


def test_match_raises_without_identity_columns():
    log = _log([{"Season": 2026, "DayNum": 10, "pred_margin": 6.0, "pred_total": 148.0,
                 "pred_prob": 0.7, "Margin": 6.0, "Total": 150.0, "Outcome": 1}])
    with pytest.raises(ValueError, match="identity columns"):
        match_comparator_to_log(log, _comparator([]))


# ── score_slate ────────────────────────────────────────────────────────────────────

def test_score_slate_is_symmetric_between_predictors():
    matched = pd.DataFrame({
        "Margin": [6.0, -4.0], "Total": [150.0, 130.0], "Outcome": [1, 0],
        "pred_margin": [7.0, -2.0], "pred_total": [148.0, 134.0], "pred_prob": [0.8, 0.3],
        "cmp_margin": [5.0, -5.0], "cmp_total": [151.0, 129.0], "cmp_prob": [0.7, 0.2],
    })
    us = score_slate(matched, "pred")
    them = score_slate(matched, "cmp")
    assert us["margin_mae"] == pytest.approx((1 + 2) / 2)
    assert them["margin_mae"] == pytest.approx((1 + 1) / 2)
    assert us["total_mae"] == pytest.approx((2 + 4) / 2)
    assert us["n"] == them["n"] == 2
    # Brier against the same Outcome column for both.
    assert us["brier"] == pytest.approx(((0.8 - 1) ** 2 + (0.3 - 0) ** 2) / 2)
    assert them["brier"] == pytest.approx(((0.7 - 1) ** 2 + (0.2 - 0) ** 2) / 2)
