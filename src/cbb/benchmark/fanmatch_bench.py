"""Benchmark our reg-season score model against KenPom FanMatch (GM-004).

FanMatch is KenPom's own per-game forecast made *as of the game date*; our reg model is also
point-in-time (pre-game Elo + as-of KenPom + home). So a head-to-head on the **same games** is a
fair "can we beat the public model at predicting scores?" test. This module is the pure matching +
scoring core; ``scripts/benchmark_fanmatch.py`` wires it to data and the trained model.

Same-games discipline (the load-bearing piece): build one matched frame by FanMatch → map names →
inner-join actuals → (in the script) inner-join the model's row, reporting N dropped at each stage.
Both predictors are then scored on that single final set — never on differently-filtered samples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fanmatch_predictions(fanmatch: pd.DataFrame) -> pd.DataFrame:
    """Derive FanMatch's margin/total/win-prob (home perspective) from raw HomePred/VisitorPred/HomeWP."""
    out = fanmatch.copy()
    out["fm_margin"] = out["HomePred"] - out["VisitorPred"]   # home minus visitor
    out["fm_total"] = out["HomePred"] + out["VisitorPred"]
    out["fm_home_wp"] = out["HomeWP"] / 100.0                 # HomeWP is 0–100
    return out


def match_fanmatch_to_results(
    fanmatch: pd.DataFrame,
    results: pd.DataFrame,
    team_map: dict[str, int],
    dayzero: "pd.Timestamp",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Match FanMatch games to actual Kaggle results, oriented to the FanMatch home team.

    Args:
        fanmatch: Raw FanMatch frame (DateOfGame, Visitor, Home, HomePred, VisitorPred, HomeWP).
        results: Kaggle ``MRegularSeasonDetailedResults`` for the season (WTeamID/LTeamID/
                 WScore/LScore/WLoc/DayNum).
        team_map: KenPom name → Kaggle TeamID (:func:`cbb.kenpom.features.build_team_name_map`).
        dayzero: That season's ``DayZero`` (to turn DateOfGame into Kaggle DayNum).

    Returns:
        ``(matched, counts)``. ``matched`` has one row per matched game with FanMatch predictions
        plus the realized ``act_margin`` (home − away), ``act_total``, ``home_won``, ``neutral``,
        and ``days_stale`` placeholder columns; ``counts`` reports N at each filter stage.
    """
    fm = fanmatch_predictions(fanmatch)
    fm["home_id"] = fm["Home"].map(team_map)
    fm["vis_id"] = fm["Visitor"].map(team_map)
    fm["DayNum"] = (pd.to_datetime(fm["DateOfGame"]) - dayzero).dt.days

    counts = {"fanmatch": len(fm)}
    fm = fm.dropna(subset=["home_id", "vis_id"]).copy()
    fm["home_id"] = fm["home_id"].astype(int)
    fm["vis_id"] = fm["vis_id"].astype(int)
    counts["names_mapped"] = len(fm)

    # Lookup actual game by (DayNum, unordered team pair) — results don't know who FanMatch calls home.
    res_lkp = {
        (int(r.DayNum), frozenset((int(r.WTeamID), int(r.LTeamID)))): r
        for r in results.itertuples(index=False)
    }
    rows = []
    for r in fm.itertuples(index=False):
        act = res_lkp.get((int(r.DayNum), frozenset((r.home_id, r.vis_id))))
        if act is None:
            continue
        ws, ls = float(act.WScore), float(act.LScore)
        home_won = int(r.home_id == int(act.WTeamID))
        act_margin = (ws - ls) if home_won else (ls - ws)  # oriented home − away
        rows.append({
            "DayNum": int(r.DayNum), "home_id": r.home_id, "vis_id": r.vis_id,
            "neutral": act.WLoc == "N",
            "fm_margin": float(r.fm_margin), "fm_total": float(r.fm_total), "fm_home_wp": float(r.fm_home_wp),
            "act_margin": act_margin, "act_total": ws + ls, "home_won": home_won,
        })
    matched = pd.DataFrame(rows)
    counts["matched_to_actual"] = len(matched)
    return matched, counts


def score_predictions(
    matched: pd.DataFrame, margin: np.ndarray, total: np.ndarray, home_wp: np.ndarray
) -> dict[str, float]:
    """Margin MAE / total MAE / Brier for a predictor's (margin, total, home win-prob) on ``matched``.

    All three predictors (FanMatch, our model) are scored through this same function on the same
    ``matched`` rows, so the comparison is strictly apples-to-apples.
    """
    margin, total, home_wp = np.asarray(margin), np.asarray(total), np.asarray(home_wp)
    act_margin = matched["act_margin"].to_numpy()
    act_total = matched["act_total"].to_numpy()
    home_won = matched["home_won"].to_numpy()
    c = 7.0  # logistic-spread transform scale (spread-quality metric; orientation-invariant)

    def L(x):
        return 1.0 / (1.0 + np.exp(-x / c))
    return {
        "margin_mae": float(np.abs(margin - act_margin).mean()),
        "total_mae": float(np.abs(total - act_total).mean()),
        "brier": float(((home_wp - home_won) ** 2).mean()),
        "logistic": float(((L(margin) - L(act_margin)) ** 2).mean()),
        "acc": float(((home_wp >= 0.5).astype(int) == home_won).mean()),
        "n": int(len(matched)),
    }
