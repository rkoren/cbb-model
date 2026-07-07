"""Tests for the regular-season parallel model (GM-001b): feature derivation, LOTO, holdout."""

import numpy as np
import pandas as pd

from cbb.train.reg_model import (
    RegConfig,
    RegModel,
    build_reg_predictions_log,
    derive_features,
    predict_reg_holdout,
    score_reg_holdout,
    train_reg_loto,
)

# Small round count keeps the LOTO unit tests fast (default 300 is for real data).
_CFG = RegConfig(num_rounds=40)


def _synthetic(seasons=(2018, 2019, 2020, 2026), n=120, seed=0):
    """A reg_games-shaped frame: Margin tracks d_Elo_pre + home; Total tracks s_Tempo_prev."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        d_elo = rng.normal(0, 150, n)
        home = rng.choice([-1, 0, 1], n)
        s_tempo = rng.normal(138, 6, n)
        margin = 0.06 * d_elo + 3.5 * home + rng.normal(0, 6, n)
        total = 0.9 * s_tempo + rng.normal(0, 8, n)
        rows.append(pd.DataFrame({
            "Season": s, "A_home": home,
            "d_Elo_pre": d_elo, "d_AdjEM_prev": rng.normal(0, 8, n),
            "s_AdjTempo_prev": s_tempo,
            "Margin": margin, "Total": total,
            "Outcome": (margin > 0).astype(int),
        }))
    return pd.concat(rows, ignore_index=True)


# ── derive_features ─────────────────────────────────────────────────────────────

def test_derive_features_splits_by_prefix():
    games = _synthetic(seasons=(2019,), n=5)
    mfeats, tfeats = derive_features(games)
    assert "A_home" in mfeats and "d_Elo_pre" in mfeats and "d_AdjEM_prev" in mfeats
    assert tfeats == ["s_AdjTempo_prev"]
    assert not any(c.startswith("s_") for c in mfeats)  # sums never in the margin head


# ── train_reg_loto ──────────────────────────────────────────────────────────────

def test_loto_emits_reg_suffixed_metrics():
    res = train_reg_loto(_synthetic(), config=_CFG, holdout_season=2026)
    for k in ("loto_brier_reg", "loto_margin_mae_reg", "loto_total_mae_reg", "n_games_reg"):
        assert k in res.metrics
    # No metric leaks the tournament name.
    assert "loto_brier" not in res.metrics


def test_loto_excludes_holdout_season():
    games = _synthetic()
    res = train_reg_loto(games, config=_CFG, holdout_season=2026)
    # OOF covers only the 3 non-holdout seasons.
    assert set(res.brier_by_season) == {2018, 2019, 2020}
    assert res.metrics["n_games_reg"] == 3 * 120


def test_loto_learns_signal():
    # Brier should beat a coinflip (0.25) and margin MAE beat predicting the mean.
    res = train_reg_loto(_synthetic(), config=_CFG, holdout_season=2026)
    assert res.metrics["loto_brier_reg"] < 0.25
    assert res.metrics["loto_margin_mae_reg"] < 12.0


def test_loto_returns_usable_model():
    res = train_reg_loto(_synthetic(), config=_CFG, holdout_season=2026)
    assert isinstance(res.model, RegModel)
    assert res.model.margin_features and res.model.total_features


# ── RegModel inference ──────────────────────────────────────────────────────────

def test_predict_scores_reconstructs_team_scores():
    res = train_reg_loto(_synthetic(), config=_CFG, holdout_season=2026)
    df = _synthetic(seasons=(2019,), n=10)
    ps = res.model.predict_scores(df)
    # ScoreA + ScoreB == total; ScoreA - ScoreB == margin (the SC-001 identity).
    assert np.allclose(ps["pred_ScoreA"] + ps["pred_ScoreB"], ps["pred_total"])
    assert np.allclose(ps["pred_ScoreA"] - ps["pred_ScoreB"], ps["pred_margin"])


def test_predict_batch_returns_probabilities():
    res = train_reg_loto(_synthetic(), config=_CFG, holdout_season=2026)
    p = res.model.predict_batch(_synthetic(seasons=(2019,), n=20))
    assert p.shape == (20,)
    assert (p >= 0).all() and (p <= 1).all()


# ── score_reg_holdout ───────────────────────────────────────────────────────────

def test_holdout_emits_reg_metrics_with_scores():
    games = _synthetic()
    res = train_reg_loto(games, config=_CFG, holdout_season=2026)
    h = score_reg_holdout(res.model, games, holdout_season=2026)
    for k in ("holdout_brier_reg", "holdout_ece_reg", "holdout_margin_mae_reg",
              "holdout_total_mae_reg", "holdout_n_games_reg"):
        assert k in h
    assert h["holdout_n_games_reg"] == 120


def test_holdout_empty_when_season_absent():
    games = _synthetic(seasons=(2018, 2019, 2020))  # no 2026
    res = train_reg_loto(games, config=_CFG, holdout_season=2026)
    assert score_reg_holdout(res.model, games, holdout_season=2026) == {}


def test_holdout_emits_women_metrics_when_gendered():
    # WM-006: with men_women/DayNum present, women-only (_w) metrics are surfaced alongside the
    # combined ones — including the early-season operational metric.
    games = _synthetic()
    hm = (games["Season"] == 2026).to_numpy()
    idx = np.where(hm)[0]
    games["men_women"] = 0
    games["DayNum"] = 100
    games.loc[idx[0::2], "men_women"] = 1   # half the holdout games are women's
    games.loc[idx[:20], "DayNum"] = 10      # some early-season games (DayNum <= 29)

    res = train_reg_loto(games, config=_CFG, holdout_season=2026)
    h = score_reg_holdout(res.model, games, holdout_season=2026)

    for k in ("holdout_brier_reg_w", "holdout_margin_mae_reg_w",
              "holdout_margin_early_reg_w", "holdout_n_games_reg_w"):
        assert k in h
    assert h["holdout_n_games_reg_w"] == int((games.loc[hm, "men_women"] == 1).sum())
    # men_women/DayNum are not d_/s_ prefixed → never picked up as model features.
    assert not any(c in res.model.margin_features + res.model.total_features
                   for c in ("men_women", "DayNum"))


def test_holdout_omits_women_metrics_when_ungendered():
    # Without a men_women column (the synthetic default), no _w metrics are emitted.
    games = _synthetic()
    res = train_reg_loto(games, config=_CFG, holdout_season=2026)
    h = score_reg_holdout(res.model, games, holdout_season=2026)
    assert not any(k.endswith("_reg_w") for k in h)


# ── DASH-001a: walk-forward predictions log ───────────────────────────────────────

def _with_identity(games):
    """Add the reg_games identity columns the predictions log threads through."""
    g = games.copy()
    g["men_women"] = 0
    g["DayNum"] = 100
    g["A_TeamID"] = np.arange(len(g)) % 300 + 1101
    g["B_TeamID"] = np.arange(len(g)) % 300 + 2101
    return g


def test_oof_is_walk_forward_and_schemad():
    res = train_reg_loto(_synthetic(), config=_CFG, holdout_season=2026)
    assert res.oof is not None
    # Walk-forward: OOF covers only the non-holdout seasons, one row per training game.
    assert set(res.oof["Season"]) == {2018, 2019, 2020}
    assert len(res.oof) == 3 * 120
    for c in ("pred_margin", "pred_total", "pred_prob", "Margin", "Total", "Outcome"):
        assert c in res.oof.columns
    assert ((res.oof["pred_prob"] >= 0) & (res.oof["pred_prob"] <= 1)).all()


def test_oof_threads_identity_columns_when_present():
    res = train_reg_loto(_with_identity(_synthetic()), config=_CFG, holdout_season=2026)
    for c in ("men_women", "DayNum", "A_TeamID", "B_TeamID"):
        assert c in res.oof.columns
    # Identity is metadata, never a model feature.
    assert not any(c in res.model.margin_features + res.model.total_features
                   for c in ("men_women", "DayNum", "A_TeamID", "B_TeamID"))


def test_predict_reg_holdout_scores_only_holdout_season():
    games = _with_identity(_synthetic())
    res = train_reg_loto(games, config=_CFG, holdout_season=2026)
    hold = predict_reg_holdout(res.model, games, holdout_season=2026)
    assert (hold["Season"] == 2026).all()
    assert len(hold) == 120
    assert "A_TeamID" in hold.columns


def test_predict_reg_holdout_empty_when_season_absent():
    games = _synthetic(seasons=(2018, 2019, 2020))
    res = train_reg_loto(games, config=_CFG, holdout_season=2026)
    hold = predict_reg_holdout(res.model, games, holdout_season=2026)
    assert hold.empty
    assert "pred_margin" in hold.columns  # correctly-columned even when empty


def test_predictions_log_concats_oof_and_holdout_with_source():
    games = _with_identity(_synthetic())
    res = train_reg_loto(games, config=_CFG, holdout_season=2026)
    log = build_reg_predictions_log(res, games, holdout_season=2026)
    # Every game across every season, tagged by origin.
    assert set(log["source"]) == {"oof", "holdout"}
    assert (log[log["source"] == "holdout"]["Season"] == 2026).all()
    assert set(log[log["source"] == "oof"]["Season"]) == {2018, 2019, 2020}
    assert len(log) == 4 * 120
