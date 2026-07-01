"""Women's model benchmark harness (WM-001) — the measuring stick for the women's epic.

There's no external women's opponent to score against yet (KenPom is men-only; BartTorvik women's
arrives in WM-003), so this is an **internal baseline tracker**: the reg model's women's holdout
error, decomposed into the buckets the diagnostic flagged as actionable — season phase (early
season is the worst), margin size (close games isolate accuracy from the higher women's variance),
and conference vs non-conference. Naive baselines and the men's numbers are reported alongside so a
future story (WM-002/003/004) can see *where* it moved the needle, not just the aggregate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cbb.train.model import _brier


def holdout_metrics(df: pd.DataFrame, margin: np.ndarray, total: np.ndarray, wp: np.ndarray) -> dict:
    """margin MAE / total MAE / Brier for predictions on a holdout slice (reg_games columns)."""
    return {
        "margin_mae": float(np.abs(np.asarray(margin) - df["Margin"].to_numpy()).mean()),
        "total_mae": float(np.abs(np.asarray(total) - df["Total"].to_numpy()).mean()),
        "brier": float(_brier(df["Outcome"].to_numpy(), np.asarray(wp))),
        "n": int(len(df)),
    }


def naive_metrics(df: pd.DataFrame) -> dict:
    """Floor baselines: predict-zero margin, predict-mean total, coin-flip Brier."""
    return {
        "margin_mae": float(df["Margin"].abs().mean()),          # predicting 0 → MAE = mean|margin|
        "total_mae": float((df["Total"] - df["Total"].mean()).abs().mean()),
        "brier": 0.25,
        "n": int(len(df)),
    }


def add_dimensions(games: pd.DataFrame, conf_lookup: dict[tuple[int, int], str]) -> pd.DataFrame:
    """Add the breakdown dimensions: season ``phase``, ``margin_bucket``, ``conf_game``."""
    g = games.copy()
    g["phase"] = pd.cut(g["DayNum"], [-1, 29, 90, 999], labels=["early", "mid", "late"])
    g["margin_bucket"] = pd.cut(
        g["Margin"].abs(), [-1, 8, 16, 999], labels=["close(≤8)", "medium(9-16)", "blowout(>16)"]
    )
    a = list(zip(g["Season"], g["A_TeamID"]))
    b = list(zip(g["Season"], g["B_TeamID"]))
    ca = [conf_lookup.get(k) for k in a]
    cb = [conf_lookup.get(k) for k in b]
    g["conf_game"] = [x is not None and x == y for x, y in zip(ca, cb)]
    return g


def report_by(df: pd.DataFrame, dim: str) -> pd.DataFrame:
    """Per-group holdout metrics for a breakdown dimension (expects pred_* columns present)."""
    rows = []
    for val, grp in df.groupby(dim, observed=True):
        m = holdout_metrics(grp, grp["pred_margin"], grp["pred_total"], grp["pred_wp"])
        rows.append({dim: val, **m})
    return pd.DataFrame(rows)
