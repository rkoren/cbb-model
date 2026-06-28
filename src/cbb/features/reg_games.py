"""Regular-season game-level dataset (GM-001) — the "handicap any game" training target.

The tournament matchup builder (:mod:`cbb.features.matchup`) trains on ~3k post-season games
with *full-season* features (leak-free only because the tournament comes after the season). To
handicap an arbitrary regular-season game we need ~210k in-season games, where full-season
features would leak the future. This module builds a symmetric A/B game-level dataset from a
**leak-free, point-in-time** feature basis:

  * **pre-game Elo** (:func:`cbb.features.elo.compute_pregame_elo`) — each team's rating *as of
    just before* the game, with prior-season carryover; the within-season strength backbone.
  * **venue / home-court** — ``A_home`` ∈ {+1 home, −1 away, 0 neutral} from Kaggle ``WLoc``;
    the model learns the home edge (~3.5 pts) rather than us hard-coding it.
  * **prior-season** end-of-season AdjOE/AdjDE/AdjEM/AdjTempo (joined on ``Season − 1``) — fully
    leak-free preseason priors (they decay in relevance as the season progresses).

Targets: ``Margin`` (ScoreA − ScoreB) and ``Total`` (ScoreA + ScoreB), same two-head design as
SC-001. Differentials ``d_*`` feed the margin head; sums ``s_*`` feed the total head. Built
vectorized (merges, not per-row lookups) because 210k games × 2 symmetric rows = ~420k rows.

This is a *parallel* dataset/model: it never touches ``matchups.parquet`` or the Kaggle path,
and Season 2026 is left in the frame but excluded from training as the trusted reg-season holdout.
GM-002 layers as-of-date KenPom snapshots and GM-003 adds pace onto this same scaffold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# adj_eff columns carried as prior-season (Season-1) priors.
_PRIOR_BASES = ["AdjOE", "AdjDE", "AdjEM", "AdjTempo"]


def _home_sign(wloc: pd.Series, a_is_winner: bool) -> np.ndarray:
    """Venue from the winner's WLoc, oriented to team A. +1 A home, -1 A away, 0 neutral."""
    # WLoc is the *winner's* location. If A is the winner, A_home follows WLoc directly;
    # if A is the loser, the sign flips (the loser was away when the winner was home).
    base = np.where(wloc.to_numpy() == "H", 1, np.where(wloc.to_numpy() == "A", -1, 0))
    return base if a_is_winner else -base


def build_reg_game_dataset(
    reg_raw: pd.DataFrame,
    pregame_elo: pd.DataFrame,
    adj_eff: pd.DataFrame,
    men_women: int,
    prior_bases: list[str] | None = None,
) -> pd.DataFrame:
    """Build the symmetric A/B regular-season game-level dataset for one gender.

    Args:
        reg_raw: Raw Kaggle ``?RegularSeasonDetailedResults`` for this gender (needs Season,
                 DayNum, WTeamID, LTeamID, WScore, LScore, WLoc).
        pregame_elo: Output of :func:`cbb.features.elo.compute_pregame_elo` for this gender.
        adj_eff: Output of :func:`compute_adj_efficiency` (all seasons) — used for prior-season
                 priors via a ``Season − 1`` join. Both genders may be passed; it is filtered.
        men_women: 0 men, 1 women.
        prior_bases: adj_eff columns to carry as priors. Defaults to AdjOE/AdjDE/AdjEM/AdjTempo.

    Returns:
        One row per (game, orientation): Season, men_women, DayNum, A_TeamID, B_TeamID, Outcome,
        Margin, Total, A_home, A/B/d ``Elo_pre``, and A/B/d/s ``*_prev`` prior-season features.
    """
    prior_bases = prior_bases or _PRIOR_BASES

    g = reg_raw[["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore", "WLoc"]].copy()
    g = g.merge(
        pregame_elo[["Season", "DayNum", "WTeamID", "LTeamID", "W_Elo_pre", "L_Elo_pre"]],
        on=["Season", "DayNum", "WTeamID", "LTeamID"], how="left",
    )

    # Prior-season (Season-1) team priors: shift adj_eff's Season +1 so it joins as the *previous*
    # season onto the current game, then attach for the winner (W_) and loser (L_) sides.
    prior = adj_eff.loc[adj_eff["men_women"] == men_women, ["Season", "TeamID", *prior_bases]].copy()
    prior["Season"] = prior["Season"] + 1
    for side, tid_col in (("W", "WTeamID"), ("L", "LTeamID")):
        ren = {"TeamID": tid_col, **{b: f"{side}_{b}_prev" for b in prior_bases}}
        g = g.merge(prior.rename(columns=ren), on=["Season", tid_col], how="left")

    margin = (g["WScore"] - g["LScore"]).to_numpy(dtype=float)
    total = (g["WScore"] + g["LScore"]).to_numpy(dtype=float)

    frames = []
    for a_is_winner in (True, False):
        win, los = ("W", "L") if a_is_winner else ("L", "W")
        rec = pd.DataFrame({
            "Season": g["Season"].to_numpy(),
            "men_women": men_women,
            "DayNum": g["DayNum"].to_numpy(),
            "A_TeamID": g[f"{win}TeamID"].to_numpy(),
            "B_TeamID": g[f"{los}TeamID"].to_numpy(),
            "Outcome": 1 if a_is_winner else 0,
            "Margin": margin if a_is_winner else -margin,
            "Total": total,
            "A_home": _home_sign(g["WLoc"], a_is_winner),
            "A_Elo_pre": g[f"{win}_Elo_pre"].to_numpy(),
            "B_Elo_pre": g[f"{los}_Elo_pre"].to_numpy(),
        })
        for b in prior_bases:
            rec[f"A_{b}_prev"] = g[f"{win}_{b}_prev"].to_numpy()
            rec[f"B_{b}_prev"] = g[f"{los}_{b}_prev"].to_numpy()
        frames.append(rec)

    games = pd.concat(frames, ignore_index=True)

    # Differentials (margin head) and sums (total head).
    games["d_Elo_pre"] = games["A_Elo_pre"] - games["B_Elo_pre"]
    for b in prior_bases:
        games[f"d_{b}_prev"] = games[f"A_{b}_prev"] - games[f"B_{b}_prev"]
        games[f"s_{b}_prev"] = games[f"A_{b}_prev"].fillna(0) + games[f"B_{b}_prev"].fillna(0)
    return games


def build_reg_games(
    data: dict[str, pd.DataFrame],
    adj_eff: pd.DataFrame,
) -> pd.DataFrame:
    """Build the combined men's + women's regular-season game-level dataset.

    Args:
        data: The raw-CSV dict (needs ``M_reg_raw``/``W_reg_raw``), as built by the features stage.
        adj_eff: Output of :func:`compute_adj_efficiency` (all seasons, both genders).

    Returns:
        Concatenated symmetric reg-season game dataset (see :func:`build_reg_game_dataset`).
    """
    from .elo import compute_pregame_elo  # noqa: PLC0415

    parts = []
    for key, mw in (("M_reg_raw", 0), ("W_reg_raw", 1)):
        reg_raw = data[key]
        pregame = compute_pregame_elo(reg_raw, men_women_flag=mw)
        parts.append(build_reg_game_dataset(reg_raw, pregame, adj_eff, men_women=mw))
    return pd.concat(parts, ignore_index=True)
