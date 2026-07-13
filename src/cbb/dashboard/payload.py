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


def _abbr(name: str) -> str:
    """A short team label for the score cells: initials for multi-word names, else a truncation.

    "South Florida" → "SF", "Wichita St" → "WS", "Connecticut" → "Conn". The full name is always
    in the matchup column, so this only needs to disambiguate the two teams at a glance.
    """
    words = [w for w in name.replace(".", "").split() if w]
    if len(words) >= 2:
        return "".join(w[0] for w in words[:3]).upper()
    return name if len(name) <= 5 else name[:4]


def _scores(total: float | None, margin: float | None) -> dict[str, int | None]:
    """Reconstruct the two team scores (A, B) from a predicted/actual total + margin (A − B)."""
    if total is None or margin is None:
        return {"a": None, "b": None}
    return {"a": round((total + margin) / 2), "b": round((total - margin) / 2)}


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
        our_margin, our_total = _num(d["pred_margin"]), _num(d["pred_total"])
        kp_margin = _num(d["cmp_margin"]) if has_cmp else None
        kp_total = _num(d.get("cmp_total")) if has_cmp else None
        act_margin, act_total = _num(d["Margin"]), _num(d["Total"])
        a_name, b_name = _name(name_map, d["A_TeamID"]), _name(name_map, d["B_TeamID"])
        gap = None if (kp_margin is None or our_margin is None) else round(our_margin - kp_margin, 2)
        gap_total = None if (kp_total is None or our_total is None) else round(our_total - kp_total, 1)
        game = {
            "a": a_name, "b": b_name,
            "a_abbr": _abbr(a_name), "b_abbr": _abbr(b_name),
            "gender": "W" if int(d.get("men_women", 0)) == 1 else "M",
            # Each predictor carries reconstructed team scores (a, b), win prob P(A wins), margin.
            "our": {**_scores(our_total, our_margin), "prob": _num(d["pred_prob"], 3), "margin": our_margin},
            "kp": None if not has_cmp else {
                **_scores(kp_total, kp_margin), "prob": _num(d.get("cmp_prob"), 3), "margin": kp_margin},
            "actual": {**_scores(act_total, act_margin), "margin": act_margin,
                       "won": None if pd.isna(d["Outcome"]) else int(d["Outcome"])},
            "gap_margin": gap,   # our margin − KenPom's (spread disagreement; the sort key)
            "gap_total": gap_total,  # our total − KenPom's (total disagreement)
            "drivers": d.get("drivers"),  # DASH-006: top XGBoost margin contributions (or None)
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
            "gender": "W" if int(d["TeamID"]) >= 2000 else "M",   # Kaggle: men < 2000, women >= 3000
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


def _mae(pred: pd.Series, actual: pd.Series) -> float:
    return float((pred - actual).abs().mean())


def _brier(prob: pd.Series, outcome: pd.Series) -> float:
    return float(((prob - outcome) ** 2).mean())


def _win_acc(prob: pd.Series, outcome: pd.Series) -> float:
    """Fraction of games where the predicted favourite (prob ≥ .5) actually won."""
    return float(((prob >= 0.5).astype(int) == outcome).mean())


def _seg_row(label: str, sub: pd.DataFrame) -> dict[str, Any]:
    """One segment's us-vs-KenPom accuracy over ``sub`` (men's games with a KenPom line + final)."""
    return {
        "label": label,
        "n": int(len(sub)),
        "us": {"margin": _num(_mae(sub["pred_margin"], sub["Margin"]), 1),
               "total": _num(_mae(sub["pred_total"], sub["Total"]), 1),
               "brier": _num(_brier(sub["pred_prob"], sub["Outcome"]), 3),
               "acc": _num(_win_acc(sub["pred_prob"], sub["Outcome"]), 3)},
        "kp": {"margin": _num(_mae(sub["cmp_margin"], sub["Margin"]), 1),
               "total": _num(_mae(sub["cmp_total"], sub["Total"]), 1),
               "brier": _num(_brier(sub["cmp_prob"], sub["Outcome"]), 3),
               "acc": _num(_win_acc(sub["cmp_prob"], sub["Outcome"]), 3)},
    }


def build_metrics(predictions_log: pd.DataFrame) -> dict[str, Any]:
    """Season accuracy vs the market (DASH-005), sliced by segment.

    KenPom FanMatch covers men only, so this is computed over men's games that have both a KenPom
    line and a final — the apples-to-apples set where "vs the market" is meaningful. Both our and
    KenPom's margin/total MAE, Brier, and win-accuracy are reported per segment.
    """
    df = predictions_log
    df = df[df["cmp_margin"].notna() & df["Outcome"].notna()].copy() if "cmp_margin" in df else df.iloc[:0]
    if df.empty:
        return {"n_games": 0, "groups": []}
    df["_amargin"] = df["Margin"].abs()
    df["_pmargin"] = df["pred_margin"].abs()
    df["_ym"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m")

    groups: list[dict[str, Any]] = [{"name": "Overall", "rows": [_seg_row("All men's games", df)]}]

    months = [(ym, g) for ym, g in df.groupby("_ym")]
    groups.append({"name": "By month", "rows": [
        _seg_row(pd.Timestamp(ym + "-01").strftime("%b %Y"), g) for ym, g in sorted(months)]})

    comp = pd.cut(df["_amargin"], [-1, 8, 16, 1e9], labels=["Close (≤8)", "Medium (9–16)", "Blowout (>16)"])
    groups.append({"name": "By final margin", "rows": [
        _seg_row(str(lbl), df[comp == lbl]) for lbl in comp.cat.categories if (comp == lbl).any()]})

    fav = pd.cut(df["_pmargin"], [-1, 4, 12, 1e9], labels=["Toss-up (≤4)", "Moderate (5–12)", "Big favorite (>12)"])
    groups.append({"name": "By our predicted spread", "rows": [
        _seg_row(str(lbl), df[fav == lbl]) for lbl in fav.cat.categories if (fav == lbl).any()]})

    if "conf_game" in df.columns and df["conf_game"].notna().any():
        rows = []
        for flag, lbl in [(True, "In-conference"), (False, "Non-conference")]:
            sub = df[df["conf_game"] == flag]
            if len(sub):
                rows.append(_seg_row(lbl, sub))
        groups.append({"name": "By conference", "rows": rows})

    return {"n_games": int(len(df)), "groups": groups}


def build_payload(
    predictions_log: pd.DataFrame,
    ratings_log: pd.DataFrame,
    name_map: dict[int, str],
    generated: str | None = None,
) -> dict[str, Any]:
    """Assemble the full dashboard payload from the two DASH-001 logs.

    Returns a JSON-serializable dict: ``slate`` (by game date), ``ratings`` (by snapshot date), and
    season ``metrics`` (DASH-005), plus the sorted date domains the shared clock scrubs over and a
    small ``meta`` block.
    """
    slate = build_slate(predictions_log, name_map)
    ratings = build_ratings(ratings_log, name_map)
    metrics = build_metrics(predictions_log)
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
        "metrics": metrics,
    }
