"""XGBoost tournament model with leaf calibration, temperature scaling, Vegas blend.

Training is leave-one-tournament-out (LOTO) for evaluation. A separate production
training path (all seasons) is used for the serving artifact.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize, minimize_scalar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline

log = logging.getLogger(__name__)

# ── Defaults from hyperparameter sweeps in efficiency_approach.ipynb ──────────
# depth=5 tested → +0.000498 worse. 700 rounds at eta=0.01 is the sweet spot.
_DEFAULT_XGB_PARAMS: dict = {
    "objective": "reg:squarederror",
    "booster": "gbtree",
    "eta": 0.01,
    "subsample": 0.6,
    "colsample_bynode": 0.8,
    "num_parallel_tree": 2,
    "min_child_weight": 4,
    "max_depth": 4,
    "tree_method": "hist",
    "grow_policy": "lossguide",
    "max_bin": 32,
    "seed": 42,
}
_DEFAULT_NUM_ROUNDS = 700


@dataclass
class ModelConfig:
    xgb_params: dict = field(default_factory=lambda: dict(_DEFAULT_XGB_PARAMS))
    num_rounds: int = _DEFAULT_NUM_ROUNDS
    # Temperature scaling bounds
    temp_bounds_close: tuple[float, float] = (0.5, 5.0)
    temp_bounds_blowout: tuple[float, float] = (0.05, 5.0)
    # Seed-gap thresholds separating close from blowout games
    men_blowout_gap: int = 10
    women_blowout_gap: int = 8
    # Logistic calibrator
    logistic_C: float = 1.0
    # NIL sample weighting (disabled by default — see matchup.py)
    nil_year: int = 2022


@dataclass
class LoTOResult:
    matchups: pd.DataFrame          # input matchups + pred_prob_raw, pred_prob, pred_prob_final
    xgb_models: dict[int, xgb.Booster]
    calibrators: dict[int, object]  # season → fitted sklearn pipeline
    temp_params: dict               # T_M_close, T_M_blowout, T_W_close, T_W_blowout
    vegas_alpha: float              # 0 = pure Vegas, 1 = pure model
    overall_brier: float
    brier_by_season: dict[int, float]
    features: list[str]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_leaf_strings(leaf_mat: np.ndarray) -> list[str]:
    return [
        " ".join(f"{j}_{int(leaf_mat[i, j])}" for j in range(leaf_mat.shape[1]))
        for i in range(leaf_mat.shape[0])
    ]


def _apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    p = np.clip(probs, 1e-7, 1 - 1e-7)
    return 1.0 / (1.0 + np.exp(-np.log(p / (1 - p)) / T))


def _brier(targets, probs, weights=None):
    return brier_score_loss(targets, probs, sample_weight=weights)


# ── LOTO training ─────────────────────────────────────────────────────────────

def train_loto(
    matchups: pd.DataFrame,
    features: list[str],
    config: ModelConfig | None = None,
    exp_config=None,
    mlflow_experiment: str | None = None,
) -> LoTOResult:
    """Leave-one-tournament-out training, calibration, temperature scaling, Vegas blend.

    Each season is held out in turn: XGBoost trained on all other seasons,
    leaf-based logistic calibration fit on the same training set, applied to
    the held-out season. Temperature scaling and Vegas blend are fit on the
    full OOF predictions afterward.

    Args:
        matchups: Output of build_matchup_dataset(). Must contain Outcome, PointDiff,
                  sample_weight, and all columns in features. If vegas_prob column
                  is present the Vegas blend is computed.
        features: Column names to use as XGBoost input features.
        config: Hyperparameter config. Defaults to notebook-tuned values.
        mlflow_experiment: If set, log metrics to this MLflow experiment.

    Returns:
        LoTOResult with predictions attached to matchups copy.
    """
    config = config or ModelConfig()
    matchups = matchups.copy()
    seasons = sorted(matchups["Season"].unique())

    if mlflow_experiment:
        mlflow.set_experiment(mlflow_experiment)

    xgb_models: dict[int, xgb.Booster] = {}
    calibrators: dict[int, object] = {}
    all_probs_raw = np.full(len(matchups), np.nan)

    with mlflow.start_run(run_name="loto_training") if mlflow_experiment else _null_ctx():
        if mlflow_experiment:
            mlflow.log_params({
                "num_rounds": config.num_rounds,
                "num_features": len(features),
                "num_seasons": len(seasons),
                **{f"xgb_{k}": v for k, v in config.xgb_params.items()},
            })
            if exp_config is not None:
                from kitchen.experiment import log_config
                log_config(exp_config)

        for oof_season in seasons:
            train_mask = matchups["Season"] != oof_season
            val_mask = matchups["Season"] == oof_season

            x_tr = matchups.loc[train_mask, features].values
            y_tr = matchups.loc[train_mask, "PointDiff"].values
            sw_tr = matchups.loc[train_mask, "sample_weight"].values
            x_va = matchups.loc[val_mask, features].values
            y_tr_outcome = matchups.loc[train_mask, "Outcome"].values

            dtrain = xgb.DMatrix(x_tr, label=y_tr, weight=sw_tr)
            dval = xgb.DMatrix(x_va)

            booster = xgb.train(
                config.xgb_params,
                dtrain,
                num_boost_round=config.num_rounds,
                verbose_eval=False,
            )
            xgb_models[oof_season] = booster

            # Leaf calibration
            dtr_leaf = xgb.DMatrix(x_tr)
            leaf_tr = _make_leaf_strings(booster.predict(dtr_leaf, pred_leaf=True))
            leaf_va = _make_leaf_strings(booster.predict(dval, pred_leaf=True))

            calibrator = make_pipeline(
                TfidfVectorizer(),
                LogisticRegression(solver="liblinear", C=config.logistic_C, max_iter=300, random_state=0),
            )
            calibrator.fit(leaf_tr, y_tr_outcome)
            raw = calibrator.predict_proba(leaf_va)[:, 1]

            calibrators[oof_season] = calibrator
            val_idx = matchups[val_mask].index
            all_probs_raw[matchups.index.get_indexer(val_idx)] = raw
            gc.collect()

        matchups["pred_prob_raw"] = all_probs_raw

        # ── Temperature scaling ────────────────────────────────────────────────
        matchups["is_playin"] = matchups["A_Seed"] == matchups["B_Seed"]
        matchups["seed_gap"] = matchups["d_Seed"].abs()

        M_close_mask = (matchups["men_women"] == 0) & ~matchups["is_playin"] & (matchups["seed_gap"] <= config.men_blowout_gap)
        M_blowout_mask = (matchups["men_women"] == 0) & ~matchups["is_playin"] & (matchups["seed_gap"] > config.men_blowout_gap)
        W_close_mask = (matchups["men_women"] == 1) & ~matchups["is_playin"] & (matchups["seed_gap"] <= config.women_blowout_gap)
        W_blowout_mask = (matchups["men_women"] == 1) & ~matchups["is_playin"] & (matchups["seed_gap"] > config.women_blowout_gap)

        def _opt_temp(mask, bounds):
            sub = matchups[mask]
            res = minimize_scalar(
                lambda T: _brier(sub["Outcome"].values, _apply_temperature(sub["pred_prob_raw"].values, T), sub["sample_weight"].values),
                bounds=bounds,
                method="bounded",
            )
            return float(res.x)

        T_M_close = _opt_temp(M_close_mask, config.temp_bounds_close)
        T_M_blowout = _opt_temp(M_blowout_mask, config.temp_bounds_blowout)
        T_W_close = _opt_temp(W_close_mask, config.temp_bounds_close)
        T_W_blowout = _opt_temp(W_blowout_mask, config.temp_bounds_blowout)

        temp_params = {
            "T_M_close": T_M_close,
            "T_M_blowout": T_M_blowout,
            "T_W_close": T_W_close,
            "T_W_blowout": T_W_blowout,
        }

        pred_prob = np.zeros(len(matchups))
        Mc_idx = matchups.index[(matchups["men_women"] == 0) & ~M_blowout_mask]
        Mb_idx = matchups.index[M_blowout_mask]
        Wc_idx = matchups.index[W_close_mask | ((matchups["men_women"] == 1) & matchups["is_playin"])]
        Wb_idx = matchups.index[W_blowout_mask]

        pred_prob[matchups.index.get_indexer(Mc_idx)] = _apply_temperature(matchups.loc[Mc_idx, "pred_prob_raw"].values, T_M_close)
        pred_prob[matchups.index.get_indexer(Mb_idx)] = _apply_temperature(matchups.loc[Mb_idx, "pred_prob_raw"].values, T_M_blowout)
        pred_prob[matchups.index.get_indexer(Wc_idx)] = _apply_temperature(matchups.loc[Wc_idx, "pred_prob_raw"].values, T_W_close)
        pred_prob[matchups.index.get_indexer(Wb_idx)] = _apply_temperature(matchups.loc[Wb_idx, "pred_prob_raw"].values, T_W_blowout)
        matchups["pred_prob"] = pred_prob

        # ── Vegas blend ────────────────────────────────────────────────────────
        vegas_alpha = 1.0  # pure model if no Vegas column
        if "vegas_prob" in matchups.columns:
            has_vegas = matchups["vegas_prob"].notna()
            if has_vegas.any():
                vegas_df = matchups[has_vegas]
                res = minimize_scalar(
                    lambda a: _brier(
                        vegas_df["Outcome"].values,
                        a * vegas_df["pred_prob"].values + (1 - a) * vegas_df["vegas_prob"].values,
                        vegas_df["sample_weight"].values,
                    ),
                    bounds=(0.0, 1.0),
                    method="bounded",
                )
                vegas_alpha = float(res.x)
                log.info("Vegas blend alpha=%.3f (0=pure Vegas, 1=pure model)", vegas_alpha)

        matchups["pred_prob_final"] = np.where(
            matchups.get("vegas_prob", pd.Series(np.nan, index=matchups.index)).notna() if "vegas_prob" in matchups.columns else False,
            vegas_alpha * matchups["pred_prob"] + (1 - vegas_alpha) * matchups["vegas_prob"],
            matchups["pred_prob"],
        ) if "vegas_prob" in matchups.columns else matchups["pred_prob"]

        # ── Metrics ────────────────────────────────────────────────────────────
        scored = ~matchups["is_playin"]
        overall_brier = _brier(
            matchups.loc[scored, "Outcome"].values,
            matchups.loc[scored, "pred_prob_final"].values,
            matchups.loc[scored, "sample_weight"].values,
        )
        brier_by_season = {
            s: _brier(
                matchups.loc[(matchups["Season"] == s) & scored, "Outcome"].values,
                matchups.loc[(matchups["Season"] == s) & scored, "pred_prob_final"].values,
            )
            for s in seasons
        }

        if mlflow_experiment:
            mlflow.log_metric("loto_brier", overall_brier)
            mlflow.log_metric("vegas_alpha", vegas_alpha)
            for k, v in temp_params.items():
                mlflow.log_metric(k, v)
            for s, b in brier_by_season.items():
                mlflow.log_metric(f"brier_{s}", b)

        log.info("LOTO Brier: %.6f", overall_brier)

    return LoTOResult(
        matchups=matchups,
        xgb_models=xgb_models,
        calibrators=calibrators,
        temp_params=temp_params,
        vegas_alpha=vegas_alpha,
        overall_brier=overall_brier,
        brier_by_season=brier_by_season,
        features=features,
    )


# ── Production model (all seasons) ───────────────────────────────────────────

def train_production(
    matchups: pd.DataFrame,
    features: list[str],
    temp_params: dict,
    config: ModelConfig | None = None,
    mlflow_experiment: str | None = None,
) -> tuple[xgb.Booster, object]:
    """Train a single model on all seasons for deployment.

    Uses temperature params from the LOTO run — they were estimated on held-out
    data so there's no leakage from reusing them here.

    Args:
        matchups: Full matchup dataset.
        features: Same feature list used in train_loto().
        temp_params: temp_params from LoTOResult.
        config: Hyperparameter config.
        mlflow_experiment: If set, log artifact to this MLflow experiment.

    Returns:
        (xgb_booster, calibrator) ready for prediction.
    """
    config = config or ModelConfig()

    x = matchups[features].values
    y_margin = matchups["PointDiff"].values
    y_outcome = matchups["Outcome"].values
    sw = matchups["sample_weight"].values

    dtrain = xgb.DMatrix(x, label=y_margin, weight=sw)
    booster = xgb.train(config.xgb_params, dtrain, num_boost_round=config.num_rounds, verbose_eval=False)

    leaf_strings = _make_leaf_strings(booster.predict(dtrain, pred_leaf=True))
    calibrator = make_pipeline(
        TfidfVectorizer(),
        LogisticRegression(solver="liblinear", C=config.logistic_C, max_iter=300, random_state=0),
    )
    calibrator.fit(leaf_strings, y_outcome)

    if mlflow_experiment:
        mlflow.set_experiment(mlflow_experiment)
        with mlflow.start_run(run_name="production_model"):
            mlflow.log_params({"num_rounds": config.num_rounds, "num_features": len(features)})
            mlflow.sklearn.log_model(calibrator, "calibrator")
            booster.save_model("/tmp/booster.ubj")
            mlflow.log_artifact("/tmp/booster.ubj", "xgb_model")

    return booster, calibrator


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(
    feature_row: pd.DataFrame,
    booster: xgb.Booster,
    calibrator: object,
    temp_params: dict,
    men_women: int,
    seed_gap: float | None = None,
    vegas_prob: float | None = None,
    vegas_alpha: float = 1.0,
) -> float:
    """Win probability for a single matchup.

    Args:
        feature_row: Single-row DataFrame with the same columns used in training.
        booster: Production XGBoost booster.
        calibrator: Production logistic calibrator.
        temp_params: Temperature scaling params from LoTOResult.
        men_women: 0 for men, 1 for women.
        seed_gap: |A_Seed - B_Seed| — determines which temperature to apply.
        vegas_prob: Market-implied win probability for team A, if available.
        vegas_alpha: Blend weight (1 = pure model, 0 = pure Vegas).

    Returns:
        Calibrated win probability for team A.
    """
    dm = xgb.DMatrix(feature_row.values)
    leaf_str = _make_leaf_strings(booster.predict(dm, pred_leaf=True))
    raw_prob = float(calibrator.predict_proba(leaf_str)[:, 1][0])

    if men_women == 0:
        T = temp_params["T_M_blowout"] if (seed_gap or 0) > 10 else temp_params["T_M_close"]
    else:
        T = temp_params["T_W_blowout"] if (seed_gap or 0) > 8 else temp_params["T_W_close"]

    prob = float(_apply_temperature(np.array([raw_prob]), T)[0])

    if vegas_prob is not None:
        prob = vegas_alpha * prob + (1 - vegas_alpha) * vegas_prob

    return prob


# ── Rating optimization (Nelder-Mead blend) ───────────────────────────────────

def optimize_rating_weights(
    matchups: pd.DataFrame,
    z_feat_cols: list[str],
    initial_weights: list[float] | None = None,
) -> np.ndarray:
    """Find convex combination of z-scored differential features minimizing LOTO Brier.

    Produces d_Rating = weighted sum of z-scored differentials, used as a pre-computed
    summary node in XGBoost. Removing it was tested (2026-03-15) and reverted: +0.000171 worse.

    Args:
        matchups: Must contain z_feat_cols, Outcome, sample_weight, Season.
        z_feat_cols: Z-scored differential feature columns (e.g. ['d_AdjEM_z', ...]).
        initial_weights: Starting point for Nelder-Mead. Defaults to uniform.

    Returns:
        Optimal weight array (same length as z_feat_cols).
    """
    x0 = initial_weights or [1.0] * len(z_feat_cols)

    def _loto_brier(weights):
        w = np.array(weights)
        preds, targets, sw = [], [], []
        for season in sorted(matchups["Season"].unique()):
            val = matchups[matchups["Season"] == season]
            d_rating = (val[z_feat_cols] * w).sum(axis=1)
            preds.extend((1 / (1 + np.exp(-d_rating))).tolist())
            targets.extend(val["Outcome"].tolist())
            sw.extend(val["sample_weight"].tolist())
        return brier_score_loss(targets, preds, sample_weight=sw)

    result = minimize(_loto_brier, x0=x0, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-6})
    log.info("Rating optimization: %s  Brier=%.6f", result.message, result.fun)
    return result.x


class _null_ctx:
    def __enter__(self): return self
    def __exit__(self, *_): pass
