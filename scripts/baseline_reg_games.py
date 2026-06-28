"""GM-001 baseline: train a leak-free margin+total model on the regular-season game-level
dataset and report MAE / Brier on the Season-2026 holdout.

Throwaway validation harness (numbers regress the dataset's usefulness) until GM-001b wires the
reg-season model into its own kitchen train/evaluate path. Run from the repo root after
`kitchen run features` (or any build that wrote data/processed/adj_eff.parquet):

    python scripts/baseline_reg_games.py
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from cbb.features.reg_games import build_reg_games

MARGIN_FEATS = ["d_Elo_pre", "A_home", "d_AdjEM_prev", "d_AdjOE_prev", "d_AdjDE_prev", "d_AdjTempo_prev"]
TOTAL_FEATS = ["s_AdjTempo_prev", "s_AdjOE_prev", "s_AdjDE_prev"]
HOLDOUT_SEASON = 2026


def _fit(feats, target, tr, te):
    dtr = xgb.DMatrix(tr[feats].to_numpy(), label=tr[target].to_numpy())
    params = {"max_depth": 4, "eta": 0.05, "subsample": 0.7, "colsample_bytree": 0.8,
              "objective": "reg:squarederror"}
    bst = xgb.train(params, dtr, num_boost_round=300)
    return bst, bst.predict(xgb.DMatrix(te[feats].to_numpy())), bst.predict(dtr)


def main() -> None:
    raw, proc = Path("data/raw"), Path("data/processed")
    data = {
        "M_reg_raw": pd.read_csv(raw / "MRegularSeasonDetailedResults.csv"),
        "W_reg_raw": pd.read_csv(raw / "WRegularSeasonDetailedResults.csv"),
    }
    games = build_reg_games(data, pd.read_parquet(proc / "adj_eff.parquet"))
    g = games.copy()
    g[MARGIN_FEATS + TOTAL_FEATS] = g[MARGIN_FEATS + TOTAL_FEATS].fillna(0)
    train, test = g[g.Season != HOLDOUT_SEASON], g[g.Season == HOLDOUT_SEASON]
    print(f"reg_games {games.shape}  train {len(train)}  holdout({HOLDOUT_SEASON}) {len(test)}")

    mbst, m_pred, m_tr = _fit(MARGIN_FEATS, "Margin", train, test)
    _, t_pred, _ = _fit(TOTAL_FEATS, "Total", train, test)
    cal = LogisticRegression(C=1.0, max_iter=1000).fit(m_tr.reshape(-1, 1), train.Outcome.to_numpy())
    prob = cal.predict_proba(m_pred.reshape(-1, 1))[:, 1]

    margin_mae = np.abs(m_pred - test.Margin.to_numpy()).mean()
    total_mae = np.abs(t_pred - test.Total.to_numpy()).mean()
    brier = brier_score_loss(test.Outcome.to_numpy(), prob)

    # Honest home-court effect = the model's PARTIAL dependence on A_home (everything else held at
    # each row's actual value), NOT the strength-confounded raw groupby marginal. Half the +1 vs -1
    # prediction gap = the per-team home edge in points.
    base = test[MARGIN_FEATS].copy()
    home, away = base.copy(), base.copy()
    home["A_home"], away["A_home"] = 1, -1
    pdp = 0.5 * (mbst.predict(xgb.DMatrix(home.to_numpy())) - mbst.predict(xgb.DMatrix(away.to_numpy()))).mean()
    raw_gap = train[train.A_home == 1].Margin.mean()

    print(f"\n=== {HOLDOUT_SEASON} reg-season holdout ===")
    print(f"margin_MAE {margin_mae:.3f}  (naive predict-mean {np.abs(test.Margin - train.Margin.mean()).abs().mean():.3f})")
    print(f"total_MAE  {total_mae:.3f}  (naive predict-mean {np.abs(test.Total - train.Total.mean()).abs().mean():.3f})")
    print(f"Brier      {brier:.4f}  (coinflip 0.25)")
    print(f"home-court: model partial effect {pdp:.2f} pts/team  |  raw confounded gap {raw_gap:.2f}")


if __name__ == "__main__":
    main()
