"""Tests for the women's benchmark harness (WM-001)."""

import numpy as np
import pandas as pd
import pytest

from cbb.benchmark.women_bench import (
    add_dimensions,
    holdout_metrics,
    naive_metrics,
    report_by,
)


def _games():
    return pd.DataFrame({
        "Season": [2026] * 4,
        "DayNum": [10, 50, 120, 10],       # early, mid, late, early
        "A_TeamID": [3101, 3102, 3103, 3101],
        "B_TeamID": [3201, 3202, 3203, 3202],
        "Margin": [4.0, -20.0, 12.0, -4.0],  # close, blowout, medium, close
        "Total": [130.0, 150.0, 140.0, 120.0],
        "Outcome": [1, 0, 1, 0],
    })


def test_holdout_metrics_arithmetic():
    df = pd.DataFrame({"Margin": [10.0, -4.0], "Total": [150.0, 130.0], "Outcome": [1, 0]})
    m = holdout_metrics(df, margin=np.array([7.0, -2.0]), total=np.array([148.0, 134.0]),
                        wp=np.array([0.8, 0.3]))
    assert m["margin_mae"] == pytest.approx((3 + 2) / 2)
    assert m["total_mae"] == pytest.approx((2 + 4) / 2)
    assert m["brier"] == pytest.approx(((0.8 - 1) ** 2 + (0.3 - 0) ** 2) / 2)
    assert m["n"] == 2


def test_naive_margin_is_mean_abs_margin():
    df = pd.DataFrame({"Margin": [4.0, -20.0, 12.0], "Total": [130.0, 150.0, 140.0], "Outcome": [1, 0, 1]})
    assert naive_metrics(df)["margin_mae"] == pytest.approx(np.mean([4, 20, 12]))
    assert naive_metrics(df)["brier"] == 0.25


def test_add_dimensions_buckets():
    g = add_dimensions(_games(), conf_lookup={})
    assert list(g["phase"]) == ["early", "mid", "late", "early"]
    assert list(g["margin_bucket"].astype(str)) == ["close(≤8)", "blowout(>16)", "medium(9-16)", "close(≤8)"]


def test_conf_game_flag():
    # 3101 and 3201 both in conf "acc" → conf game; 3102 in "acc", 3202 in "sec" → non-conf.
    lookup = {(2026, 3101): "acc", (2026, 3201): "acc", (2026, 3102): "acc", (2026, 3202): "sec"}
    g = add_dimensions(_games(), lookup)
    assert list(g["conf_game"]) == [True, False, False, False]  # only game 1 has both teams mapped+same


def test_report_by_groups():
    g = add_dimensions(_games(), conf_lookup={})
    g["pred_margin"] = g["Margin"]     # perfect predictions → 0 MAE
    g["pred_total"] = g["Total"]
    g["pred_wp"] = g["Outcome"].astype(float)
    rep = report_by(g, "phase")
    assert set(rep["phase"]) == {"early", "mid", "late"}
    assert (rep["margin_mae"] == 0).all()
    assert rep[rep.phase == "early"]["n"].iloc[0] == 2
