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

Men's TeamID < 2000, Women's >= 3000 (Kaggle convention); `men_women` is inferred. Scores are
optional — only win/loss is needed for Brier. Absent file → the holdout is skipped (no-op), so
the pipeline runs unchanged until you drop the results in.
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
    and the Kaggle submission format. ``men_women`` is inferred from the team IDs.
    """
    rows = []
    for r in results.itertuples(index=False):
        w, l = int(r.WTeamID), int(r.LTeamID)
        a, b = (w, l) if w < l else (l, w)
        rows.append(
            {
                "Season": int(r.Season),
                "men_women": _infer_men_women(a),
                "A_TeamID": a,
                "B_TeamID": b,
                "Outcome": 1 if a == w else 0,
            }
        )
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

    pred = pred.merge(
        pairs[["Season", "men_women", "A_TeamID", "B_TeamID", "Outcome"]],
        on=["Season", "men_women", "A_TeamID", "B_TeamID"],
        how="left",
    )
    log.info("Holdout matchups built: %d games (%d cols)", len(pred), len(pred.columns))
    return pred


def score_holdout(
    model, holdout: pd.DataFrame, features: list[str]
) -> dict[str, float]:
    """Score a trained model on the holdout matchups → ``{holdout_brier, holdout_n_games}``.

    ``model`` must expose ``predict_batch(df)`` (a ``CBBModel``). Any feature the model expects
    but the holdout lacks is a parity break that would silently bias a *trusted* metric, so it is
    logged as a WARNING (not swallowed) before being zero-filled to avoid a hard crash.
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
    probs = np.asarray(model.predict_batch(df), dtype=float)
    brier = float(brier_score_loss(df["Outcome"].to_numpy(), probs))
    return {"holdout_brier": brier, "holdout_n_games": int(len(df))}
