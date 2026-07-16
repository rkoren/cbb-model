"""Rolling, point-in-time box-score features (GM-003) — the box-score analog of as-of KenPom.

KenPom's as-of ratings (GM-002) are men's-only and start 2012, and KenPom offers no *as-of*
four-factors/tempo at all. So the within-season pace/efficiency signal for **women** (41% of the
data, otherwise 0% as-of), **pre-2012 men**, and **totals** (the persistent gap vs FanMatch,
GM-004) has to be computed from the Kaggle box scores ourselves. This walks each team's games in
order and, *before* each game, emits its season-to-date:

  * ``bs_Tempo_asof`` — possessions per 40 min (OT-adjusted pace)
  * ``bs_OE_asof`` / ``bs_DE_asof`` — points scored / allowed per 100 possessions
  * ``bs_NetEff_asof`` — OE − DE (an unadjusted AdjEM analog)

Raw (not opponent-adjusted) on purpose: pace is largely opponent-independent (SOS adjusts *rates*,
not possessions), so raw as-of tempo directly targets the total gap; the model already has
opponent-adjusted margin signal via ``d_kp_AdjEM_asof`` / ``d_AdjEM_prev``. Leak-free by
construction — emit-before-update, and **within a DayNum all games use the pre-day snapshot**
(Kaggle has no within-day order, so same-day games are treated as simultaneous). The first
``MIN_GAMES`` games of a team emit NaN (too few to be stable); the ``*_prev`` priors cover that
cold-start. No cross-season carryover — rosters turn over and the priors already carry last year.
"""

from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd

_FT = 0.475  # FT-attempt weight in the possessions estimate (matches four_factors.py)
MIN_GAMES = 3  # below this, season-to-date rates are too noisy → emit NaN (priors cover)

BS_METRICS = ["Tempo", "OE", "DE", "NetEff"]
BS_FEATURES = [f"bs_{m}_asof" for m in BS_METRICS]


def _poss(fga: float, orb: float, to: float, fta: float) -> float:
    """Offensive possessions estimate: FGA − OR + TO + 0.475·FTA."""
    return fga - orb + to + _FT * fta


class _State:
    """Per-team season-to-date accumulators (points, possessions, per-40 pace, game count)."""

    __slots__ = ("pf", "pa", "posf", "posa", "pace", "n")

    def __init__(self) -> None:
        self.pf = self.pa = self.posf = self.posa = self.pace = 0.0
        self.n = 0

    def emit(self, min_games: int) -> tuple[float, float, float, float]:
        if self.n < min_games or self.posf <= 0 or self.posa <= 0:
            return (float("nan"),) * 4
        oe = 100.0 * self.pf / self.posf
        de = 100.0 * self.pa / self.posa
        return (self.pace / self.n, oe, de, oe - de)

    def update(self, pts_for: float, pts_against: float, poss_for: float, poss_against: float, adjot: float) -> None:
        self.pf += pts_for
        self.pa += pts_against
        self.posf += poss_for
        self.posa += poss_against
        self.pace += poss_for / adjot  # per-40 pace (OT-normalized); efficiency ratios stay raw
        self.n += 1


def compute_rolling_boxscore(
    reg_raw: pd.DataFrame, men_women_flag: int, min_games: int = MIN_GAMES
) -> pd.DataFrame:
    """Per-game as-of box-score ratings for both teams — leak-free (emit strictly before the game).

    Args:
        reg_raw: Raw Kaggle ``?RegularSeasonDetailedResults`` (needs Season, DayNum, WTeamID,
                 LTeamID, W/L Score, FGA, OR, TO, FTA, and NumOT).
        men_women_flag: 0 men, 1 women.

    Returns:
        One row per game: Season, men_women, DayNum, WTeamID, LTeamID, and W_/L_ ``bs_*_asof``.
    """
    df = reg_raw.sort_values(["Season", "DayNum"])
    rows: list[tuple] = []
    cols = ["Season", "men_women", "DayNum", "WTeamID", "LTeamID"] + \
        [f"W_{f}" for f in BS_FEATURES] + [f"L_{f}" for f in BS_FEATURES]

    for season, grp in df.groupby("Season"):
        state: dict[int, _State] = defaultdict(_State)
        for daynum, day in grp.groupby("DayNum"):
            # 1) Emit for every game in this DayNum from the pre-day snapshot (same-day = simultaneous).
            for r in day.itertuples(index=False):
                w, l = int(r.WTeamID), int(r.LTeamID)
                rows.append((season, men_women_flag, int(daynum), w, l,
                             *state[w].emit(min_games), *state[l].emit(min_games)))
            # 2) Then apply all of this DayNum's games to the running state.
            for r in day.itertuples(index=False):
                w, l = int(r.WTeamID), int(r.LTeamID)
                adjot = (40 + 5 * getattr(r, "NumOT", 0)) / 40
                wposs = _poss(r.WFGA, r.WOR, r.WTO, r.WFTA)
                lposs = _poss(r.LFGA, r.LOR, r.LTO, r.LFTA)
                # Skip games with unknown box-score detail (NaN possessions) — e.g. score-only
                # results (the 2026 tournament holdout) — so they don't corrupt the accumulator;
                # a team's later games then carry its last *known* as-of state forward. No-op on
                # real Kaggle detailed results (always complete → possessions never NaN).
                if math.isnan(wposs) or math.isnan(lposs):
                    continue
                state[w].update(r.WScore, r.LScore, wposs, lposs, adjot)
                state[l].update(r.LScore, r.WScore, lposs, wposs, adjot)

    return pd.DataFrame(rows, columns=cols)
