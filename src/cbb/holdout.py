"""2026 frozen holdout — score the ≤2025-trained model on the real 2026 tournament.

`matchups.parquet` only contains seasons whose tournament results are in the Kaggle data
(2003–2025), so the production model is trained *as if the 2026 tournament has not happened*.
This module builds a held-out matchup set for the 2026 tournament from its **actual results**
(dropped into `data/holdout/`, never read by training) joined to the 2026 team features the
features pipeline already computes from the 2026 regular season + Selection-Sunday KenPom
snapshot — so the holdout is leak-free by construction. Every run is then scored against it
(`holdout_brier`): a trusted generalization number, distinct from the in-CV `loto_brier`.

Discipline: iterate on `loto_brier` (CV over ≤2025); check `holdout_brier` sparingly. Picking
the best-of-N experiments by the holdout overfits it by selection even though no row leaks —
treat 2026 like a Kaggle private leaderboard.

Data contract
-------------
`data/holdout/tourney_results_2026.csv` — compact schema, one row per played tournament game::

    Season,WTeamID,LTeamID[,WScore,LScore,DayNum]

Men's TeamID < 2000, Women's >= 3000 (Kaggle convention); `men_women` is inferred. `WScore`/
`LScore` are optional — win/loss alone gives `holdout_brier`/`holdout_ece`; adding the scores
also yields `holdout_margin_mae`/`holdout_total_mae` (SC-003). Absent file → the holdout is
skipped (no-op), so the pipeline runs unchanged until you drop the results in.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

log = logging.getLogger(__name__)

HOLDOUT_SEASON = 2026
HOLDOUT_DIR = "data/holdout"
RESULTS_FILE = "tourney_results_2026.csv"
HOLDOUT_PARQUET = "holdout_2026.parquet"

_REQUIRED_RESULT_COLS = ["Season", "WTeamID", "LTeamID"]


def _infer_men_women(team_id: int) -> int:
    """0 = men (TeamID < 2000), 1 = women (TeamID >= 3000) — Kaggle convention."""
    return 0 if team_id < 2000 else 1


def load_holdout_results(holdout_dir: str | Path = HOLDOUT_DIR) -> pd.DataFrame | None:
    """Read the frozen 2026 results CSV, or return ``None`` when it isn't present.

    Returns a DataFrame with at least ``Season, WTeamID, LTeamID``. Raises ``ValueError`` only
    when the file exists but is malformed (so a typo is caught, not silently skipped).
    """
    path = Path(holdout_dir) / RESULTS_FILE
    if not path.exists():
        return None
    df = pd.read_csv(path)
    missing = [c for c in _REQUIRED_RESULT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns {missing}; found {df.columns.tolist()}")
    df = df[df["Season"] == HOLDOUT_SEASON].copy()
    if df.empty:
        raise ValueError(f"{path} has no rows for Season {HOLDOUT_SEASON}")
    return df


def finalize_template(
    template_path: str | Path, out_path: str | Path | None = None
) -> pd.DataFrame:
    """Convert a filled holdout template into the compact results contract and write it.

    The template (from ``scripts/build_holdout_template.py``) has one row per game with
    ``A_TeamID``/``B_TeamID`` and a ``Winner`` column you fill with the winning TeamID. This
    keeps only filled rows, validates each ``Winner`` is one of that game's two teams, and
    writes ``Season,WTeamID,LTeamID`` to ``out_path`` (default ``data/holdout/<RESULTS_FILE>``)
    — the file ``load_holdout_results`` reads. Returns the results frame.
    """
    t = pd.read_csv(template_path)
    if "Winner" not in t.columns:
        raise ValueError(f"{template_path} has no 'Winner' column to finalize")
    filled = t[t["Winner"].notna() & (t["Winner"].astype(str).str.strip() != "")].copy()
    if filled.empty:
        raise ValueError(f"{template_path}: no rows have a Winner filled in")

    rows = []
    for r in filled.itertuples(index=False):
        w, a, b = int(r.Winner), int(r.A_TeamID), int(r.B_TeamID)
        if w not in (a, b):
            raise ValueError(f"Winner {w} is not one of the game's teams ({a} vs {b})")
        rows.append({"Season": int(r.Season), "WTeamID": w, "LTeamID": b if w == a else a})

    out = pd.DataFrame(rows)
    dest = Path(out_path) if out_path else Path(HOLDOUT_DIR) / RESULTS_FILE
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    log.info("Finalized %d games → %s", len(out), dest)
    return out


def results_to_pairs(results: pd.DataFrame) -> pd.DataFrame:
    """Convert compact W/L results into A/B prediction pairs with the realized outcome.

    Each game becomes one row with ``A_TeamID < B_TeamID`` (Kaggle ID order) and
    ``Outcome = 1`` iff the lower-ID team won — the same orientation as the training matchups
    and the Kaggle submission format. ``men_women`` is inferred from the team IDs. When the
    results carry ``WScore``/``LScore`` (optional), the realized ``Margin`` (ScoreA − ScoreB,
    oriented to A) and ``Total`` are attached too — enabling holdout ``margin_MAE``/``total_MAE``.
    """
    has_scores = "WScore" in results.columns and "LScore" in results.columns
    rows = []
    for r in results.itertuples(index=False):
        w, l = int(r.WTeamID), int(r.LTeamID)
        a, b = (w, l) if w < l else (l, w)
        rec = {
            "Season": int(r.Season),
            "men_women": _infer_men_women(a),
            "A_TeamID": a,
            "B_TeamID": b,
            "Outcome": 1 if a == w else 0,
        }
        if has_scores and pd.notna(r.WScore) and pd.notna(r.LScore):
            ws, ls = float(r.WScore), float(r.LScore)
            sa, sb = (ws, ls) if a == w else (ls, ws)  # orient scores to A
            rec["Margin"] = sa - sb
            rec["Total"] = ws + ls
        rows.append(rec)
    return pd.DataFrame(rows)


def build_holdout_matchups(
    results: pd.DataFrame,
    adj_eff: pd.DataFrame,
    season_avgs: pd.DataFrame,
    elo_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    form_df: pd.DataFrame,
    seed_lookup: dict,
    massey_lookup: dict,
    reg_sym: pd.DataFrame,
    scaler,
    opt_weights,
    z_features: list[str],
    kp_rich: pd.DataFrame,
) -> pd.DataFrame:
    """Build holdout matchup rows with the SAME features (and post-processing) as training.

    Reuses the submission feature path (``build_prediction_features``) plus the exact
    ``d_Rating`` blend (the *fitted* training ``scaler`` + ``opt_weights``) and the KenPom
    rich join — so the holdout has feature parity with ``matchups.parquet`` by construction —
    then attaches the realized ``Outcome``. All inputs are the in-memory frames the features
    stage already holds, so no work is recomputed.
    """
    from cbb.features import build_prediction_features  # noqa: PLC0415
    from cbb.kenpom.rich_features import join_kenpom_rich  # noqa: PLC0415

    pairs = results_to_pairs(results)
    pred = build_prediction_features(
        pairs[["Season", "men_women", "A_TeamID", "B_TeamID"]],
        adj_eff, season_avgs, elo_df, quality_df, form_df,
        seed_lookup, massey_lookup, reg_sym,
    )

    # d_Rating: identical blend to training — transform with the fitted scaler, weight by opt_weights.
    z_cols = [f"{c}_z" for c in z_features]
    z_scaled = scaler.transform(pred[z_features].fillna(0))
    for i, col in enumerate(z_cols):
        pred[col] = z_scaled[:, i]
    pred["d_Rating"] = (pred[z_cols] * np.asarray(opt_weights)).sum(axis=1)

    # KenPom rich differentials (no-op if kp_rich is empty)
    pred, _ = join_kenpom_rich(pred, kp_rich)

    merge_cols = ["Season", "men_women", "A_TeamID", "B_TeamID", "Outcome"]
    merge_cols += [c for c in ("Margin", "Total") if c in pairs.columns]  # actual scores, if provided
    pred = pred.merge(
        pairs[merge_cols], on=["Season", "men_women", "A_TeamID", "B_TeamID"], how="left"
    )
    log.info("Holdout matchups built: %d games (%d cols)", len(pred), len(pred.columns))
    return pred


def _expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error: weighted mean |confidence − accuracy| over equal-width bins."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += (mask.sum() / len(p)) * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def score_holdout(
    model, holdout: pd.DataFrame, features: list[str]
) -> dict[str, float]:
    """Score a trained model on the holdout → win-prob + (when scores exist) score-accuracy.

    Always returns ``holdout_brier``, ``holdout_n_games``, and ``holdout_ece`` (calibration of
    the win prob). When the model has a total head (SC-001) **and** the holdout carries actual
    ``Margin``/``Total`` (i.e. the results CSV had ``WScore``/``LScore``), also returns
    ``holdout_margin_mae``/``holdout_total_mae``/``holdout_scored_games`` (SC-003).

    Any feature the model expects but the holdout lacks is a parity break that would silently
    bias a *trusted* metric, so it is logged as a WARNING before being zero-filled.
    """
    df = holdout.copy()
    missing = [f for f in features if f not in df.columns]
    if missing:
        log.warning(
            "Holdout missing %d model feature(s) — zero-filled; the holdout_brier is suspect: %s",
            len(missing), ", ".join(sorted(missing)),
        )
        for f in missing:
            df[f] = 0.0
    df[features] = df[features].fillna(0)
    y = df["Outcome"].to_numpy()
    probs = np.asarray(model.predict_batch(df), dtype=float)
    out: dict[str, float] = {
        "holdout_brier": float(brier_score_loss(y, probs)),
        "holdout_n_games": int(len(df)),
        "holdout_ece": _expected_calibration_error(y, probs),
    }

    # Score-prediction accuracy — needs the total head + actual scores in the holdout.
    if getattr(model, "total_booster", None) is not None and {"Margin", "Total"} <= set(df.columns):
        scored = df[df["Margin"].notna() & df["Total"].notna()]
        if len(scored):
            ps = model.predict_scores(scored)
            out["holdout_margin_mae"] = float(np.abs(ps["pred_margin"].to_numpy() - scored["Margin"].to_numpy()).mean())
            out["holdout_total_mae"] = float(np.abs(ps["pred_total"].to_numpy() - scored["Total"].to_numpy()).mean())
            out["holdout_scored_games"] = int(len(scored))
    return out
