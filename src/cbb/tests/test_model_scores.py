"""Tests for SC-001 — the total head + score prediction (sum features, predict_scores)."""

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from cbb.features.matchup import add_total_sum_features
from cbb.train.model import CBBModel


# ── add_total_sum_features ─────────────────────────────────────────────────────

def test_sum_features_are_a_plus_b():
    df = pd.DataFrame({"A_AdjTempo": [70.0], "B_AdjTempo": [68.0], "A_AdjOE": [110.0], "B_AdjOE": [105.0]})
    add_total_sum_features(df)
    assert df["s_AdjTempo"].iloc[0] == 138.0
    assert df["s_AdjOE"].iloc[0] == 215.0


def test_sum_features_skip_missing_base():
    # Only A_/B_ AdjTempo present → only s_AdjTempo added, no error for the others.
    df = pd.DataFrame({"A_AdjTempo": [70.0], "B_AdjTempo": [68.0]})
    add_total_sum_features(df)
    assert "s_AdjTempo" in df.columns
    assert "s_AdjOE" not in df.columns


# ── CBBModel.predict_scores ────────────────────────────────────────────────────

def _tiny_booster(x, y):
    return xgb.train({"objective": "reg:squarederror"}, xgb.DMatrix(x, label=y), num_boost_round=3)


def test_predict_scores_reconstructs_from_margin_and_total():
    rng = np.random.default_rng(0)
    x = rng.random((40, 2))
    feats = ["f0", "f1"]
    margin_head = _tiny_booster(x, x[:, 0] * 20 - 10)
    total_head = _tiny_booster(x, x[:, 1] * 20 + 130)
    df = pd.DataFrame(x, columns=feats)
    model = CBBModel(
        booster=margin_head, calibrator=None, total_booster=total_head, total_features=feats,
        temp_params={}, vegas_alpha=1.0, features=feats,
    )
    out = model.predict_scores(df)
    assert set(out.columns) == {"pred_margin", "pred_total", "pred_ScoreA", "pred_ScoreB"}
    # ScoreA + ScoreB == total ; ScoreA - ScoreB == margin (the reconstruction identity)
    np.testing.assert_allclose(out.pred_ScoreA + out.pred_ScoreB, out.pred_total, atol=1e-4)
    np.testing.assert_allclose(out.pred_ScoreA - out.pred_ScoreB, out.pred_margin, atol=1e-4)


def test_predict_scores_uses_total_features_for_total_head():
    # Margin head reads f0/f1; total head reads its own s_* feature → both must be in df.
    rng = np.random.default_rng(1)
    x = rng.random((30, 3))
    margin_head = _tiny_booster(x[:, :2], x[:, 0])
    total_head = _tiny_booster(x[:, 2:3], x[:, 2] * 10 + 140)
    df = pd.DataFrame(x, columns=["f0", "f1", "s_AdjTempo"])
    model = CBBModel(
        booster=margin_head, calibrator=None, total_booster=total_head,
        total_features=["s_AdjTempo"], temp_params={}, vegas_alpha=1.0, features=["f0", "f1"],
    )
    out = model.predict_scores(df)  # no KeyError → total head correctly read s_AdjTempo
    assert len(out) == 30


def test_predict_scores_requires_total_booster():
    model = CBBModel(
        booster=None, calibrator=None, total_booster=None,
        temp_params={}, vegas_alpha=1.0, features=["f0", "f1"],
    )
    with pytest.raises(ValueError, match="total_booster"):
        model.predict_scores(pd.DataFrame({"f0": [0.1], "f1": [0.2]}))
