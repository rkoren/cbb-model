"""Unified slate matcher (DASH-001b) — align the walk-forward predictions log with a comparator.

DASH-001a produced one leak-free predictions log (our as-of margin/total/win-prob per game, in
the Kaggle **A/B** convention). This module joins that log to a Home/Visitor **comparator**
(KenPom FanMatch now; a Torvik-derived frame later) on the *unordered* team-pair + date, and
re-orients the comparator into our A-perspective so both predictors line up column-for-column.

It generalizes the one-off matching in ``benchmark_fanmatch/women/tournament`` into a single
builder that consumes the *predictions log* instead of re-scoring the model — keeping the
us-vs-KenPom deltas walk-forward (DASH-001a) and giving every downstream page (DASH-002/005) one
aligned frame to render. Actuals ride along on the log itself (``Margin``/``Total``/``Outcome``),
so scoring both sides is apples-to-apples via :func:`score_slate`.

Pure transforms only; persistence + the ratings log are DASH-001c's concern.
"""

from __future__ import annotations

import pandas as pd

from cbb.train.model import _brier

# The comparator contract the generic matcher consumes. Home/Visitor oriented: ``cmp_margin`` is
# home − visitor, ``cmp_home_wp`` is P(home wins) in 0–1, ``cmp_total`` is orientation-invariant.
COMPARATOR_COLS = ["Season", "DayNum", "home_id", "vis_id", "cmp_margin", "cmp_total", "cmp_home_wp"]

# Identity columns the predictions log must carry for the pair+date join (from DASH-001a).
_LOG_KEYS = ["Season", "DayNum", "A_TeamID", "B_TeamID"]


def fanmatch_to_comparator(
    fanmatch: pd.DataFrame, team_map: dict[str, int], dayzero: pd.Timestamp, season: int
) -> pd.DataFrame:
    """Adapt a raw KenPom FanMatch frame to the generic :data:`COMPARATOR_COLS` contract.

    Args:
        fanmatch: Raw FanMatch (``DateOfGame, Home, Visitor, HomePred, VisitorPred, HomeWP``).
        team_map: KenPom name → Kaggle TeamID (``cbb.kenpom.features.build_team_name_map``).
        dayzero: The season's ``DayZero`` (turns ``DateOfGame`` into Kaggle ``DayNum``).
        season: Season stamped on every row (FanMatch frames are per-season; men-2026 in practice).

    Unmapped names are dropped (they can't join). ``cmp_*`` reuse the FanMatch-perspective
    derivation in :func:`cbb.benchmark.fanmatch_bench.fanmatch_predictions`.
    """
    from .fanmatch_bench import fanmatch_predictions

    fm = fanmatch_predictions(fanmatch)
    fm["home_id"] = fm["Home"].map(team_map)
    fm["vis_id"] = fm["Visitor"].map(team_map)
    fm = fm.dropna(subset=["home_id", "vis_id"]).copy()
    fm["home_id"] = fm["home_id"].astype(int)
    fm["vis_id"] = fm["vis_id"].astype(int)
    fm["DayNum"] = (pd.to_datetime(fm["DateOfGame"]) - dayzero).dt.days
    fm["Season"] = season
    return fm.rename(columns={"fm_margin": "cmp_margin", "fm_total": "cmp_total",
                              "fm_home_wp": "cmp_home_wp"})[COMPARATOR_COLS].reset_index(drop=True)


def match_comparator_to_log(
    pred_log: pd.DataFrame, comparator: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Inner-join a comparator onto the predictions log, re-oriented to our A perspective.

    Keys on ``(Season, DayNum, {A_TeamID, B_TeamID})`` — an *unordered* pair, since the comparator
    doesn't know who we call A. When the comparator's home team is our B, the margin sign flips and
    the win-prob becomes ``1 − home_wp`` (total is orientation-invariant). Returns the log rows
    that matched, augmented with A-oriented ``cmp_margin``/``cmp_total``/``cmp_prob``, plus a
    per-stage ``counts`` dict.

    Raises:
        ValueError: if the log is missing the identity keys DASH-001a threads through.
    """
    missing = [c for c in _LOG_KEYS if c not in pred_log.columns]
    if missing:
        raise ValueError(
            f"predictions log missing identity columns {missing} — DASH-001a must thread them "
            "through (train_reg_loto on reg_games, not a synthetic frame)"
        )

    lkp = {
        (int(r.Season), int(r.DayNum), frozenset((int(r.home_id), int(r.vis_id)))): r
        for r in comparator.itertuples(index=False)
    }
    counts = {"log": len(pred_log), "comparator": len(comparator)}
    rows = []
    for r in pred_log.itertuples(index=False):
        c = lkp.get((int(r.Season), int(r.DayNum), frozenset((int(r.A_TeamID), int(r.B_TeamID)))))
        if c is None:
            continue
        if int(c.home_id) == int(r.A_TeamID):        # comparator home == our A → same orientation
            cmp_margin, cmp_prob = float(c.cmp_margin), float(c.cmp_home_wp)
        else:                                        # comparator home == our B → flip to A
            cmp_margin, cmp_prob = -float(c.cmp_margin), 1.0 - float(c.cmp_home_wp)
        row = r._asdict()
        row.update(cmp_margin=cmp_margin, cmp_total=float(c.cmp_total), cmp_prob=cmp_prob)
        rows.append(row)

    matched = pd.DataFrame(rows)
    counts["matched"] = len(matched)
    return matched, counts


def score_slate(matched: pd.DataFrame, prefix: str) -> dict[str, float]:
    """Margin MAE / total MAE / Brier for one predictor on a matched slate.

    ``prefix`` selects the columns: ``"pred"`` scores us (from the log), ``"cmp"`` scores the
    comparator — both against the actuals (``Margin``/``Total``/``Outcome``) that ride on the log,
    so the two are strictly apples-to-apples.
    """
    return {
        "margin_mae": float((matched[f"{prefix}_margin"] - matched["Margin"]).abs().mean()),
        "total_mae": float((matched[f"{prefix}_total"] - matched["Total"]).abs().mean()),
        "brier": float(_brier(matched["Outcome"].to_numpy(), matched[f"{prefix}_prob"].to_numpy())),
        "n": int(len(matched)),
    }
