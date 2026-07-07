"""Regular-season game-level model (GM-001b) — a *parallel* model to the tournament one.

Trains two XGBoost heads (margin + total, the SC-001 design) plus a 1-feature margin→prob
logistic calibrator (the SC-002 design) on the leak-free regular-season game dataset
(:mod:`cbb.features.reg_games`). It deliberately reuses only the small SC-002 calibrator helpers
from :mod:`cbb.train.model`; the tournament ``train_loto``/``train_production`` carry Vegas
blending, seed-gap temperature and NIL weighting that don't apply to an arbitrary in-season game.

Methodology notes:
  * **Holdout = Season 2026**, excluded from all training (consistent with the 2026 discipline).
    Unlike the tournament holdout, the reg-season 2026 rows carry *real* scores already, so
    margin/total MAE work with no manual data entry.
  * **LOTO-by-season** CV — leave one season out; the symmetric A/B pair always lands in the same
    fold (split by season, never by row, so the two orientations can't leak across folds).
  * **Cross-fit (OOF) calibrator** — the production margin→prob calibrator is fit on the
    out-of-fold LOTO margin predictions, not in-sample. On ~400k rows an in-sample calibrator is
    over-confident; OOF keeps it honest.

Metrics are logged with a ``_reg`` suffix (``loto_brier_reg`` etc.) so they never collide with
the tournament model's same-named metrics in a shared experiment (the CBB-019 hazard).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb

from cbb.holdout import _expected_calibration_error
from cbb.train.model import _brier, _fit_margin_calibrator, _margin_to_prob

HOLDOUT_SEASON = 2026

# Early-season cutoff for the women's operational metric (WM-001): DayNum <= this is the
# "early" phase, matching the season-phase buckets in cbb.benchmark.women_bench.
_EARLY_DAYNUM = 29

_REG_XGB_PARAMS = {
    "max_depth": 4,
    "eta": 0.05,
    "subsample": 0.7,
    "colsample_bytree": 0.8,
    "tree_method": "hist",   # fast on the ~400k-row dataset
    "objective": "reg:squarederror",
}
_REG_NUM_ROUNDS = 300


@dataclass
class RegConfig:
    xgb_params: dict = field(default_factory=lambda: dict(_REG_XGB_PARAMS))
    num_rounds: int = _REG_NUM_ROUNDS
    logistic_C: float = 1.0


@dataclass
class RegModel:
    """Production reg-season bundle: margin head + total head + margin→prob calibrator.

    Mirrors :class:`cbb.train.model.CBBModel`'s inference surface (``predict_scores`` /
    ``predict_batch``) so the kitchen evaluate / sklearn-logging / registry paths treat it the
    same — but it is a *separate* registered model (``cbb-reg-model``), not the tournament champion.
    """

    margin_booster: xgb.Booster
    total_booster: xgb.Booster
    calibrator: object
    margin_features: list[str]
    total_features: list[str]

    def predict_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict margin, total, and both team scores for a batch of games (SC-001 design)."""
        m = self.margin_booster.predict(xgb.DMatrix(df[self.margin_features].fillna(0).to_numpy()))
        t = self.total_booster.predict(xgb.DMatrix(df[self.total_features].fillna(0).to_numpy()))
        return pd.DataFrame(
            {"pred_margin": m, "pred_total": t,
             "pred_ScoreA": (t + m) / 2.0, "pred_ScoreB": (t - m) / 2.0},
            index=df.index,
        )

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Win probability P(A wins) via the calibrator applied to the predicted margin."""
        m = self.margin_booster.predict(xgb.DMatrix(df[self.margin_features].fillna(0).to_numpy()))
        return _margin_to_prob(self.calibrator, m)


# Identity columns carried through the predictions log (DASH-001a) so the downstream matcher
# (DASH-001b) can key on the unordered team-pair + date. Threaded through only when present, so
# synthetic frames without them still train/score.
_ID_COLS = ["men_women", "DayNum", "A_TeamID", "B_TeamID"]

# Canonical predictions-log schema (identity cols added when available).
_PRED_COLS = ["Season", "pred_margin", "pred_total", "pred_prob", "Margin", "Total", "Outcome"]


@dataclass
class RegLoToResult:
    model: RegModel
    metrics: dict[str, float]
    brier_by_season: dict[int, float]
    # DASH-001a: per-game out-of-fold (walk-forward) predictions over the non-holdout seasons.
    # ``None`` only for back-compat with callers that build a result by hand.
    oof: pd.DataFrame | None = None


def derive_features(games: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Margin head = venue + all ``d_*`` differentials; total head = all ``s_*`` sums.

    The dataset *is* the feature schema (no menu re-listing needed): differentials drive the
    margin head, sums drive the total head, and ``A_home`` is the venue term for the margin.
    """
    margin = (["A_home"] if "A_home" in games.columns else []) + \
        [c for c in games.columns if c.startswith("d_")]
    total = [c for c in games.columns if c.startswith("s_")]
    return margin, total


def _train_heads(
    train: pd.DataFrame, mfeats: list[str], tfeats: list[str], config: RegConfig
) -> tuple[xgb.Booster, xgb.Booster]:
    mbst = xgb.train(
        config.xgb_params,
        xgb.DMatrix(train[mfeats].to_numpy(), label=train["Margin"].to_numpy()),
        num_boost_round=config.num_rounds,
    )
    tbst = xgb.train(
        config.xgb_params,
        xgb.DMatrix(train[tfeats].to_numpy(), label=train["Total"].to_numpy()),
        num_boost_round=config.num_rounds,
    )
    return mbst, tbst


def train_reg_loto(
    games: pd.DataFrame,
    config: RegConfig | None = None,
    holdout_season: int = HOLDOUT_SEASON,
) -> RegLoToResult:
    """LOTO-by-season CV over the reg-season dataset → metrics + a production RegModel.

    Holds out ``holdout_season`` entirely (never trained on, scored separately). For each
    remaining season it trains both heads on the others and predicts the held-out season,
    collecting out-of-fold predictions; the OOF set yields ``loto_*_reg`` metrics and fits the
    *production* calibrator (cross-fit, not in-sample). The production heads are then trained on
    all non-holdout seasons.
    """
    config = config or RegConfig()
    mfeats, tfeats = derive_features(games)

    df = games[games["Season"] != holdout_season].copy()
    df[mfeats + tfeats] = df[mfeats + tfeats].fillna(0)
    seasons = sorted(df["Season"].unique())

    oof_parts = []
    for s in seasons:
        tr, va = df[df["Season"] != s], df[df["Season"] == s]
        mbst, tbst = _train_heads(tr, mfeats, tfeats, config)
        m_pred = mbst.predict(xgb.DMatrix(va[mfeats].to_numpy()))
        t_pred = tbst.predict(xgb.DMatrix(va[tfeats].to_numpy()))
        # Per-fold calibrator (1-param logistic — negligible overfit) applied OOF for the brier.
        cal = _fit_margin_calibrator(
            mbst.predict(xgb.DMatrix(tr[mfeats].to_numpy())), tr["Outcome"].to_numpy(), config.logistic_C
        )
        part = pd.DataFrame({
            "Season": s,
            "pred_margin": m_pred, "pred_total": t_pred,
            "pred_prob": _margin_to_prob(cal, m_pred),
            "Margin": va["Margin"].to_numpy(), "Total": va["Total"].to_numpy(),
            "Outcome": va["Outcome"].to_numpy(),
        })
        for c in _ID_COLS:
            if c in va.columns:
                part[c] = va[c].to_numpy()
        oof_parts.append(part)
    oof = pd.concat(oof_parts, ignore_index=True)

    brier_by_season = {
        int(s): _brier(g["Outcome"].to_numpy(), g["pred_prob"].to_numpy())
        for s, g in oof.groupby("Season")
    }
    metrics = {
        "loto_brier_reg": _brier(oof["Outcome"].to_numpy(), oof["pred_prob"].to_numpy()),
        "loto_margin_mae_reg": float(np.abs(oof["pred_margin"] - oof["Margin"]).mean()),
        "loto_total_mae_reg": float(np.abs(oof["pred_total"] - oof["Total"]).mean()),
        "n_games_reg": int(len(oof)),
    }

    # Production heads on all non-holdout seasons; calibrator cross-fit on the OOF margins.
    mbst, tbst = _train_heads(df, mfeats, tfeats, config)
    prod_cal = _fit_margin_calibrator(oof["pred_margin"].to_numpy(), oof["Outcome"].to_numpy(), config.logistic_C)
    model = RegModel(mbst, tbst, prod_cal, mfeats, tfeats)
    return RegLoToResult(model=model, metrics=metrics, brier_by_season=brier_by_season, oof=oof)


def score_reg_holdout(
    model: RegModel, games: pd.DataFrame, holdout_season: int = HOLDOUT_SEASON
) -> dict[str, float]:
    """Score a RegModel on the held-out season → win-prob + score-accuracy (all ``_reg``).

    The reg-season holdout carries real scores, so margin/total MAE need no extra data.
    """
    h = games[games["Season"] == holdout_season].copy()
    if h.empty:
        return {}
    h[model.margin_features + model.total_features] = \
        h[model.margin_features + model.total_features].fillna(0)
    probs = np.asarray(model.predict_batch(h))
    ps = model.predict_scores(h)
    y = h["Outcome"].to_numpy()
    margin_err = np.abs(ps["pred_margin"].to_numpy() - h["Margin"].to_numpy())
    out = {
        "holdout_brier_reg": _brier(y, probs),
        "holdout_ece_reg": _expected_calibration_error(y, probs),
        "holdout_margin_mae_reg": float(margin_err.mean()),
        "holdout_total_mae_reg": float(np.abs(ps["pred_total"].to_numpy() - h["Total"].to_numpy()).mean()),
        "holdout_n_games_reg": int(len(h)),
    }

    # WM-006: women-only holdout metrics. The combined-gender gauge above can't see women-only
    # gains (WM-003's Torvik lift was invisible to loto_brier_reg), so surface them with a `_w`
    # suffix → they show in kitchen leaderboard/diff and become promotable. Guarded on the
    # column so synthetic frames without `men_women` still score.
    if "men_women" in h.columns:
        w = (h["men_women"] == 1).to_numpy()
        if w.any():
            out["holdout_brier_reg_w"] = _brier(y[w], probs[w])
            out["holdout_margin_mae_reg_w"] = float(margin_err[w].mean())
            out["holdout_n_games_reg_w"] = int(w.sum())
            if "DayNum" in h.columns:
                early_w = w & (h["DayNum"].to_numpy() <= _EARLY_DAYNUM)
                if early_w.any():
                    out["holdout_margin_early_reg_w"] = float(margin_err[early_w].mean())
    return out


def predict_reg_holdout(
    model: RegModel, games: pd.DataFrame, holdout_season: int = HOLDOUT_SEASON
) -> pd.DataFrame:
    """Per-game predictions on the held-out season (DASH-001a), same schema as ``result.oof``.

    Walk-forward by construction: the production model is trained on all *non*-holdout seasons,
    so its predictions on ``holdout_season`` never saw those games. Returns an empty, correctly-
    columned frame when the season is absent.
    """
    h = games[games["Season"] == holdout_season].copy()
    id_cols = [c for c in _ID_COLS if c in h.columns]
    if h.empty:
        return pd.DataFrame(columns=_PRED_COLS + id_cols)
    h[model.margin_features + model.total_features] = \
        h[model.margin_features + model.total_features].fillna(0)
    ps = model.predict_scores(h)
    out = pd.DataFrame({
        "Season": h["Season"].to_numpy(),
        "pred_margin": ps["pred_margin"].to_numpy(),
        "pred_total": ps["pred_total"].to_numpy(),
        "pred_prob": np.asarray(model.predict_batch(h)),
        "Margin": h["Margin"].to_numpy(), "Total": h["Total"].to_numpy(),
        "Outcome": h["Outcome"].to_numpy(),
    })
    for c in id_cols:
        out[c] = h[c].to_numpy()
    return out


def build_reg_predictions_log(
    result: RegLoToResult, games: pd.DataFrame, holdout_season: int = HOLDOUT_SEASON
) -> pd.DataFrame:
    """The DASH-001a predictions log: walk-forward per-game predictions, all seasons.

    Concatenates the LOTO out-of-fold rows (non-holdout seasons, each scored by a model that
    never trained on that season) with the ``holdout_season`` rows (scored by the all-other-
    seasons production model). A ``source`` column (``"oof"`` / ``"holdout"``) marks the origin
    so downstream can filter. Every row is leak-free as-of its game — the correctness core of
    the dashboard's "we beat KenPom by X" claim (M6 resolved-risk #5).

    In-memory only; persistence + the KenPom/actual join are DASH-001b's job.
    """
    oof = result.oof.copy() if result.oof is not None else pd.DataFrame(columns=_PRED_COLS)
    oof["source"] = "oof"
    hold = predict_reg_holdout(result.model, games, holdout_season)
    hold["source"] = "holdout"
    return pd.concat([oof, hold], ignore_index=True)
