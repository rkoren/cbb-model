"""Self-computed as-of adjusted efficiency snapshots (WM-002 — the independence play).

Productionizes the reverse-KenPom idea end-to-end. Instead of leaning on KenPom's within-season
``archive`` ratings (``kp_*_asof`` — men-only, 2012+), we compute *our own* point-in-time
opponent-adjusted efficiency at weekly cutoffs — for **both genders, every season** — and feed it
into the same as-of merge pipeline as ``adjself_*_asof``. Two payoffs:

  * **Independence** — the reg model's core within-season strength signal no longer comes from
    KenPom, so an ours-vs-FanMatch comparison is genuinely independent (see the M6 dashboard).
  * **Coverage** — it fills KenPom's gaps: women (no KenPom at all) and pre-2012 men.

The spike (``scripts/spike_self_efficiency.py``) showed dropping ``kp_*_asof`` for this is
accuracy-neutral (loto_brier_reg 0.17043 → 0.16926), so the reg pipeline uses these snapshots and
drops the KenPom as-of join.

Leak-freeness: the snapshot at cutoff ``cut`` is built only from games with ``DayNum < cut`` and is
stamped with the calendar date ``DayZero + cut``, so the downstream
``merge_asof(..., allow_exact_matches=False)`` join (in :mod:`cbb.kenpom.asof_features`) gives each
game the latest snapshot *strictly before* its own date — the same as-of guarantee KenPom gets.
"""

from __future__ import annotations

import logging

import pandas as pd

from .efficiency import compute_adj_efficiency

log = logging.getLogger(__name__)

_BASES = ["AdjOE", "AdjDE", "AdjEM", "AdjTempo"]
# adj_eff column -> as-of feature name; any ``*_asof`` column is picked up by add_asof_features.
ADJSELF_COLS: dict[str, str] = {b: f"adjself_{b}_asof" for b in _BASES}
ADJSELF_FEATURES: list[str] = list(ADJSELF_COLS.values())

# Weekly cutoffs across the regular season (DayZero ≈ early Nov, Selection Sunday ≈ DayNum 132).
# Matches the validated spike grid; the last snapshot carries late-season games to the tournament.
FIRST_DAYNUM, LAST_DAYNUM, STEP = 14, 133, 7

_EMPTY = pd.DataFrame(columns=["Season", "TeamID", "ArchiveDate", *ADJSELF_FEATURES])


def compute_adjself_asof_snapshots(
    reg_sym: pd.DataFrame,
    dayzero_by_season: dict[int, "pd.Timestamp"],
    first_daynum: int = FIRST_DAYNUM,
    last_daynum: int = LAST_DAYNUM,
    step: int = STEP,
) -> pd.DataFrame:
    """Weekly point-in-time reverse-KenPom efficiency snapshots (both genders, all seasons).

    For each weekly cutoff, run :func:`compute_adj_efficiency` on the games *strictly before* it and
    stamp the result with the cutoff's calendar date. Returns a long frame in the same shape as the
    KenPom/Torvik as-of snapshots — ``Season, TeamID, ArchiveDate (datetime), adjself_*_asof`` —
    ready for :func:`cbb.kenpom.asof_features.add_asof_features`. TeamIDs are gender-disjoint (Kaggle
    convention men < 2000, women ≥ 3000), so ``(Season, TeamID)`` keys the join without ``men_women``.

    Empty frame if ``dayzero_by_season`` is missing (→ the downstream join is a no-op).
    """
    if not dayzero_by_season:
        return _EMPTY.copy()

    parts: list[pd.DataFrame] = []
    for cut in range(first_daynum, last_daynum + 1, step):
        sub = reg_sym[reg_sym["DayNum"] < cut]
        if not len(sub):
            continue
        adj = compute_adj_efficiency(sub).rename(columns=ADJSELF_COLS)
        adj = adj[["Season", "TeamID", *ADJSELF_FEATURES]].copy()
        dz = adj["Season"].map(dayzero_by_season)
        adj["ArchiveDate"] = pd.to_datetime(dz) + pd.to_timedelta(cut, unit="D")
        parts.append(adj.dropna(subset=["ArchiveDate"]))

    if not parts:
        return _EMPTY.copy()
    out = pd.concat(parts, ignore_index=True)
    log.info(
        "adjself as-of: %d snapshot-rows, %d seasons, %d cutoffs",
        len(out), out["Season"].nunique(), out["ArchiveDate"].dt.strftime("%m-%d").nunique(),
    )
    return out[["Season", "TeamID", "ArchiveDate", *ADJSELF_FEATURES]]
