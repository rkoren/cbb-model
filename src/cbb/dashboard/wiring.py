"""Wiring helpers that feed real on-disk data into the dashboard payload (DASH-002).

Pure transforms that bridge the DASH-001 logs and Kaggle reference data to what
:func:`cbb.dashboard.payload.build_payload` expects:

  * ``add_game_date`` — the predictions log carries ``Season``/``DayNum`` (Kaggle day index); the
    slate groups by calendar date, so stamp ``game_date = DayZero + DayNum`` per (season, gender).
  * ``build_name_map`` — TeamID → display name from Kaggle ``MTeams``/``WTeams`` (the one Kaggle
    read the payload deliberately doesn't do itself).

Kept here (pure, unit-tested) rather than inline in the build script so the date math and name
resolution don't need real data to verify.
"""

from __future__ import annotations

import pandas as pd

from cbb.benchmark.slate import match_comparator_to_log


def dayzero_by_gender_season(
    mseasons: pd.DataFrame, wseasons: pd.DataFrame | None = None
) -> dict[tuple[int, int], pd.Timestamp]:
    """``(Season, men_women) -> DayZero`` from Kaggle ``MSeasons``/``WSeasons`` (``men_women`` 0/1).

    Men's and women's calendars have their own DayZero, so the map is keyed by both — a men's and
    a women's game on the same DayNum can fall on different calendar dates.
    """
    out: dict[tuple[int, int], pd.Timestamp] = {}
    for df, mw in [(mseasons, 0), (wseasons, 1)]:
        if df is None:
            continue
        for r in df.itertuples(index=False):
            out[(int(r.Season), mw)] = pd.to_datetime(r.DayZero, format="%m/%d/%Y")
    return out


def add_game_date(
    log: pd.DataFrame, dayzero: dict[tuple[int, int], pd.Timestamp]
) -> pd.DataFrame:
    """Return ``log`` with a ``game_date`` column = ``DayZero[(Season, men_women)] + DayNum days``.

    Rows whose (season, gender) has no DayZero get ``NaT`` (dropped downstream rather than
    mis-dated). ``men_women`` defaults to 0 (men) when the column is absent.
    """
    mw = log["men_women"].astype(int) if "men_women" in log.columns else pd.Series(0, index=log.index)
    base = pd.to_datetime(
        pd.Series([dayzero.get((int(s), int(g))) for s, g in zip(log["Season"], mw)], index=log.index)
    )
    out = log.copy()
    out["game_date"] = base + pd.to_timedelta(log["DayNum"].astype(int), unit="D")
    return out


def dedupe_symmetric(log: pd.DataFrame) -> pd.DataFrame:
    """Collapse the symmetric A/B dataset to one row per game for display.

    ``reg_games`` carries both orientations of every game (winner-as-A and loser-as-A), so the
    predictions log does too — rendering each game twice with mirrored margins. Keep the canonical
    orientation ``A_TeamID < B_TeamID`` (deterministic; margins/prob stay oriented to that A).
    """
    return log[log["A_TeamID"].astype(int) < log["B_TeamID"].astype(int)].reset_index(drop=True)


def attach_kenpom_slate(log: pd.DataFrame, comparator: pd.DataFrame) -> pd.DataFrame:
    """Left-join the KenPom comparator's A-oriented ``cmp_*`` onto the predictions log.

    :func:`cbb.benchmark.slate.match_comparator_to_log` inner-joins (matched games only, re-oriented
    to our A); this keeps every log row and fills ``cmp_margin``/``cmp_total``/``cmp_prob`` where a
    comparator existed, leaving the rest NaN — so the slate shows a KenPom line only where FanMatch
    covered the game (men-2026 in practice). No-op when the comparator is empty.
    """
    if comparator.empty:
        return log
    matched, _ = match_comparator_to_log(log, comparator)
    if matched.empty:
        return log
    keys = ["Season", "DayNum", "A_TeamID", "B_TeamID"]
    cmp_cols = ["cmp_margin", "cmp_total", "cmp_prob"]
    return log.merge(matched[keys + cmp_cols], on=keys, how="left")


def build_name_map(*team_frames: pd.DataFrame | None) -> dict[int, str]:
    """``TeamID -> TeamName`` merged across Kaggle team frames (pass ``MTeams`` and ``WTeams``)."""
    m: dict[int, str] = {}
    for df in team_frames:
        if df is None:
            continue
        for r in df.itertuples(index=False):
            m[int(r.TeamID)] = str(r.TeamName)
    return m
