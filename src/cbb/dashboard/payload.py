"""Handicapper dashboard payload (DASH-007) — turn the two logs into a JSON-serializable dict.

The dashboard is a single self-contained HTML file (the DASH-007 decision), so everything it
renders is embedded as one JSON blob. This module is the pure transform from the two DASH-001
logs into that blob; :mod:`cbb.dashboard.render` turns the blob into HTML. Keeping them separate
keeps the data shape unit-testable without touching markup.

Two linked views hang off one shared clock (DASH-004): pick a date → the **FanMatch slate** shows
that day's games (us vs KenPom vs actual, DASH-002), and the **ratings** view shows the latest
weekly snapshot on-or-before that date (ours vs KenPom/Torvik, DASH-003). So the payload carries
the slate keyed by game date and the ratings keyed by snapshot date; the "latest ≤ clock"
resolution is a tiny bit of client JS.

Pure: no I/O, no Kaggle reads. ``name_map`` (TeamID → display name) is passed in by the caller
(the future wiring script does the one Kaggle read); ``generated`` is passed in too, so the
output is deterministic for tests.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

# Predictions-log columns the slate needs; ``cmp_*`` (KenPom) are optional (unmatched games have
# no comparator) and ``game_date`` is the calendar date the wiring stamps from DayZero + DayNum.
_SLATE_REQUIRED = ["game_date", "A_TeamID", "B_TeamID", "pred_margin", "pred_total", "pred_prob",
                   "Margin", "Total", "Outcome"]


def _num(v: Any, digits: int = 2) -> float | None:
    """Round to ``digits``, mapping NaN/NA/None → None so the JSON has no NaN literals."""
    if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v):
        return None
    return round(float(v), digits)


def _iso(v: Any) -> str:
    """A date/datetime/str → 'YYYY-MM-DD'."""
    return pd.Timestamp(v).strftime("%Y-%m-%d")


def _name(name_map: dict[int, str], tid: Any) -> str:
    return name_map.get(int(tid), f"#{int(tid)}")


def build_slate(predictions_log: pd.DataFrame, name_map: dict[int, str]) -> dict[str, list[dict]]:
    """Group the predictions log into ``{game_date: [game, ...]}`` (DASH-002).

    Each game carries our line, KenPom's (``None`` when unmatched), the actual result, and the
    us−KenPom margin gap (for sort-by-disagreement). Games are ordered by that gap, biggest first.
    """
    missing = [c for c in _SLATE_REQUIRED if c not in predictions_log.columns]
    if missing:
        raise ValueError(f"predictions_log missing columns for the slate: {missing}")
    has_cmp = "cmp_margin" in predictions_log.columns

    slate: dict[str, list[dict]] = {}
    for r in predictions_log.itertuples(index=False):
        d = r._asdict()
        kp_margin = _num(d["cmp_margin"]) if has_cmp else None
        our_margin = _num(d["pred_margin"])
        gap = None if (kp_margin is None or our_margin is None) else round(our_margin - kp_margin, 2)
        game = {
            "a": _name(name_map, d["A_TeamID"]),
            "b": _name(name_map, d["B_TeamID"]),
            "gender": "W" if int(d.get("men_women", 0)) == 1 else "M",
            "our": {"margin": our_margin, "total": _num(d["pred_total"]),
                    "prob": _num(d["pred_prob"], 3)},
            "kp": None if not has_cmp else {
                "margin": kp_margin, "total": _num(d.get("cmp_total")),
                "prob": _num(d.get("cmp_prob"), 3)},
            "actual": {"margin": _num(d["Margin"]), "total": _num(d["Total"]),
                       "won": None if pd.isna(d["Outcome"]) else int(d["Outcome"])},
            "gap_margin": gap,
        }
        slate.setdefault(_iso(d["game_date"]), []).append(game)

    for games in slate.values():
        games.sort(key=lambda g: abs(g["gap_margin"]) if g["gap_margin"] is not None else -1,
                   reverse=True)
    return slate


def build_ratings(ratings_log: pd.DataFrame, name_map: dict[int, str]) -> dict[str, list[dict]]:
    """Group the ratings log into ``{snapshot_date: [team, ...]}`` (DASH-003), best rank first."""
    if ratings_log.empty:
        return {}
    ratings: dict[str, list[dict]] = {}
    for r in ratings_log.itertuples(index=False):
        d = r._asdict()
        team = {
            "team": _name(name_map, d["TeamID"]),
            "our": {"em": _num(d["our_AdjEM"]), "oe": _num(d.get("our_AdjOE")),
                    "de": _num(d.get("our_AdjDE")), "tempo": _num(d.get("our_AdjTempo")),
                    "rank": _num(d.get("our_rank"), 0)},
            "kp": {"em": _num(d.get("cmp_AdjEM")), "oe": _num(d.get("cmp_AdjOE")),
                   "de": _num(d.get("cmp_AdjDE")), "tempo": _num(d.get("cmp_AdjTempo")),
                   "rank": _num(d.get("cmp_rank"), 0)},
            "d_em": _num(d.get("d_AdjEM")),
            "d_rank": _num(d.get("d_rank"), 0),
        }
        ratings.setdefault(_iso(d["ArchiveDate"]), []).append(team)

    for teams in ratings.values():
        teams.sort(key=lambda t: t["our"]["rank"] if t["our"]["rank"] is not None else 1e9)
    return ratings


def build_payload(
    predictions_log: pd.DataFrame,
    ratings_log: pd.DataFrame,
    name_map: dict[int, str],
    generated: str | None = None,
) -> dict[str, Any]:
    """Assemble the full dashboard payload from the two DASH-001 logs.

    Returns a JSON-serializable dict: ``slate`` (by game date) and ``ratings`` (by snapshot date),
    plus the sorted date domains the shared clock scrubs over and a small ``meta`` block.
    """
    slate = build_slate(predictions_log, name_map)
    ratings = build_ratings(ratings_log, name_map)
    return {
        "meta": {
            "generated": generated,
            "n_games": int(len(predictions_log)),
            "n_snapshots": len(ratings),
        },
        "slate_dates": sorted(slate),      # daily — the scrub domain
        "rating_dates": sorted(ratings),   # weekly snapshots
        "slate": slate,
        "ratings": ratings,
    }
