"""Tests for cbb.holdout — the 2026 frozen holdout (data contract + scoring)."""

import pandas as pd
import pytest

from cbb.holdout import (
    HOLDOUT_SEASON,
    finalize_template,
    load_holdout_results,
    results_to_pairs,
    score_holdout,
)


def _write_template(path, winner_a=None, winner_b=None):
    # One game: A=1101, B=1205. winner_* let a test set who won (or leave blank).
    path.write_text(
        "Season,A_TeamID,A_Name,B_TeamID,B_Name,Winner\n"
        f"2026,1101,Alpha,1205,Bravo,{winner_a or ''}\n"
        f"2026,1300,Charlie,1402,Delta,{winner_b or ''}\n"
    )


# ── results_to_pairs ───────────────────────────────────────────────────────────

def test_pairs_orientation_and_outcome():
    # Game 1: lower ID (1101) won → A=1101, Outcome=1. Game 2: higher ID (1300) won → A=1200, Outcome=0.
    results = pd.DataFrame({
        "Season": [2026, 2026],
        "WTeamID": [1101, 1300],
        "LTeamID": [1205, 1200],
    })
    pairs = results_to_pairs(results)
    assert list(pairs["A_TeamID"]) == [1101, 1200]
    assert list(pairs["B_TeamID"]) == [1205, 1300]
    assert list(pairs["Outcome"]) == [1, 0]


def test_pairs_infers_men_women_from_teamid():
    results = pd.DataFrame({"Season": [2026, 2026], "WTeamID": [1101, 3101], "LTeamID": [1205, 3205]})
    pairs = results_to_pairs(results)
    assert list(pairs["men_women"]) == [0, 1]  # men <2000, women >=3000


# ── load_holdout_results ───────────────────────────────────────────────────────

def test_load_absent_returns_none(tmp_path):
    assert load_holdout_results(tmp_path) is None


def test_load_missing_columns_raises(tmp_path):
    (tmp_path / "tourney_results_2026.csv").write_text("Season,Foo\n2026,1\n")
    with pytest.raises(ValueError, match="missing required columns"):
        load_holdout_results(tmp_path)


def test_load_filters_to_holdout_season(tmp_path):
    (tmp_path / "tourney_results_2026.csv").write_text(
        "Season,WTeamID,LTeamID\n2025,1101,1205\n2026,1101,1205\n"
    )
    df = load_holdout_results(tmp_path)
    assert (df["Season"] == HOLDOUT_SEASON).all()
    assert len(df) == 1


def test_load_no_holdout_season_rows_raises(tmp_path):
    (tmp_path / "tourney_results_2026.csv").write_text("Season,WTeamID,LTeamID\n2025,1101,1205\n")
    with pytest.raises(ValueError, match="no rows for Season"):
        load_holdout_results(tmp_path)


# ── score_holdout ──────────────────────────────────────────────────────────────

class _StubModel:
    """Returns a fixed prob per row regardless of features."""
    def __init__(self, prob): self.prob = prob
    def predict_batch(self, df): return [self.prob] * len(df)


def test_score_holdout_brier():
    holdout = pd.DataFrame({"Outcome": [1, 0], "d_AdjEM": [5.0, -3.0]})
    # Predict 0.75 for both; Brier = mean[(0.75-1)^2, (0.75-0)^2] = mean[0.0625, 0.5625] = 0.3125
    out = score_holdout(_StubModel(0.75), holdout, ["d_AdjEM"])
    assert out["holdout_n_games"] == 2
    assert out["holdout_brier"] == pytest.approx(0.3125)


def test_score_holdout_fills_missing_features():
    # Feature not in the holdout frame → filled with 0, no KeyError.
    holdout = pd.DataFrame({"Outcome": [1], "d_AdjEM": [5.0]})
    out = score_holdout(_StubModel(0.6), holdout, ["d_AdjEM", "d_kp_SOS"])
    assert out["holdout_n_games"] == 1


def test_score_holdout_always_reports_ece():
    holdout = pd.DataFrame({"Outcome": [1, 0], "d_AdjEM": [5.0, -3.0]})
    assert "holdout_ece" in score_holdout(_StubModel(0.6), holdout, ["d_AdjEM"])


# ── SC-003: actual scores → margin/total MAE ───────────────────────────────────

def test_pairs_attach_actual_scores():
    res = pd.DataFrame({
        "Season": [2026, 2026], "WTeamID": [1101, 1300], "LTeamID": [1205, 1200],
        "WScore": [80, 78], "LScore": [72, 70],
    })
    pairs = results_to_pairs(res)
    # game1 A=1101=W: Margin 80-72=8; game2 A=1200=L (W=1300): A=70, B=78 → -8
    assert list(pairs["Margin"]) == [8.0, -8.0]
    assert list(pairs["Total"]) == [152.0, 148.0]


def test_pairs_no_scores_no_margin_columns():
    pairs = results_to_pairs(pd.DataFrame({"Season": [2026], "WTeamID": [1101], "LTeamID": [1205]}))
    assert "Margin" not in pairs.columns and "Total" not in pairs.columns


class _ScoreStub(_StubModel):
    """A model with a total head + fixed score predictions."""
    total_booster = "present"
    def predict_scores(self, df):
        return pd.DataFrame({"pred_margin": [5.0] * len(df), "pred_total": [140.0] * len(df)}, index=df.index)


def test_score_holdout_margin_total_mae_with_scores():
    holdout = pd.DataFrame({"Outcome": [1, 0], "d_AdjEM": [5.0, -3.0], "Margin": [8.0, -2.0], "Total": [150.0, 130.0]})
    out = score_holdout(_ScoreStub(0.6), holdout, ["d_AdjEM"])
    assert out["holdout_margin_mae"] == pytest.approx(5.0)   # |5-8|,|5-(-2)| → mean 5
    assert out["holdout_total_mae"] == pytest.approx(10.0)   # |140-150|,|140-130| → mean 10
    assert out["holdout_scored_games"] == 2


def test_score_holdout_skips_mae_without_scores():
    holdout = pd.DataFrame({"Outcome": [1], "d_AdjEM": [5.0]})
    assert "holdout_margin_mae" not in score_holdout(_ScoreStub(0.6), holdout, ["d_AdjEM"])


# ── finalize_template ──────────────────────────────────────────────────────────

def test_finalize_template_converts_winners(tmp_path):
    tmpl = tmp_path / "2026_template.csv"
    _write_template(tmpl, winner_a=1205, winner_b=1300)  # B wins game 1, A wins game 2
    out = finalize_template(tmpl, out_path=tmp_path / "results.csv")
    assert list(out.columns) == ["Season", "WTeamID", "LTeamID"]
    g1 = out.iloc[0]
    assert g1["WTeamID"] == 1205 and g1["LTeamID"] == 1101  # loser is the other team
    g2 = out.iloc[1]
    assert g2["WTeamID"] == 1300 and g2["LTeamID"] == 1402
    assert (tmp_path / "results.csv").exists()


def test_finalize_template_skips_unfilled_rows(tmp_path):
    tmpl = tmp_path / "2026_template.csv"
    _write_template(tmpl, winner_a=1101, winner_b=None)  # only game 1 played
    out = finalize_template(tmpl, out_path=tmp_path / "results.csv")
    assert len(out) == 1


def test_finalize_template_rejects_winner_not_in_game(tmp_path):
    tmpl = tmp_path / "2026_template.csv"
    _write_template(tmpl, winner_a=9999)  # 9999 isn't one of the game's teams
    with pytest.raises(ValueError, match="not one of the game's teams"):
        finalize_template(tmpl, out_path=tmp_path / "results.csv")


def test_finalize_template_raises_when_none_filled(tmp_path):
    tmpl = tmp_path / "2026_template.csv"
    _write_template(tmpl)  # no winners
    with pytest.raises(ValueError, match="no rows have a Winner"):
        finalize_template(tmpl, out_path=tmp_path / "results.csv")
