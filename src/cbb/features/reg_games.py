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


def _games_played_before(reg_raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per raw game, the count of prior games each team (W, L) has played that season (WM-004).

    A maturity signal for the prior→as-of blend: early season → few games → lean on the
    prior-season prior; as games accumulate → lean on the within-season rating.
    """
    n = len(reg_raw)
    long = pd.DataFrame({
        "Season": np.concatenate([reg_raw["Season"].to_numpy(), reg_raw["Season"].to_numpy()]),
        "DayNum": np.concatenate([reg_raw["DayNum"].to_numpy(), reg_raw["DayNum"].to_numpy()]),
        "team": np.concatenate([reg_raw["WTeamID"].to_numpy(), reg_raw["LTeamID"].to_numpy()]),
        "gi": np.concatenate([np.arange(n), np.arange(n)]),
        "side": np.array(["W"] * n + ["L"] * n),
    }).sort_values(["Season", "DayNum"], kind="stable")
    long["gp"] = long.groupby(["Season", "team"]).cumcount()
    w = long[long["side"] == "W"].set_index("gi")["gp"].reindex(range(n)).to_numpy()
    l = long[long["side"] == "L"].set_index("gi")["gp"].reindex(range(n)).to_numpy()
    return w, l


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
    rolling: pd.DataFrame | None = None,
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

    # Rolling as-of box-score features (GM-003) — W_/L_ bs_*_asof, no-op when not provided.
    from .rolling import BS_FEATURES  # noqa: PLC0415
    bs_feats = BS_FEATURES if rolling is not None and len(rolling) else []
    if bs_feats:
        keep = ["Season", "DayNum", "WTeamID", "LTeamID"] + \
            [f"W_{f}" for f in bs_feats] + [f"L_{f}" for f in bs_feats]
        g = g.merge(rolling[keep], on=["Season", "DayNum", "WTeamID", "LTeamID"], how="left")

    g["W_games_asof"], g["L_games_asof"] = _games_played_before(reg_raw)  # WM-004 maturity signal

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
            "A_games_asof": g[f"{win}_games_asof"].to_numpy(),
            "B_games_asof": g[f"{los}_games_asof"].to_numpy(),
        })
        for b in prior_bases:
            rec[f"A_{b}_prev"] = g[f"{win}_{b}_prev"].to_numpy()
            rec[f"B_{b}_prev"] = g[f"{los}_{b}_prev"].to_numpy()
        for f in bs_feats:
            rec[f"A_{f}"] = g[f"{win}_{f}"].to_numpy()
            rec[f"B_{f}"] = g[f"{los}_{f}"].to_numpy()
        frames.append(rec)

    games = pd.concat(frames, ignore_index=True)

    # Differentials (margin head) and sums (total head).
    games["d_Elo_pre"] = games["A_Elo_pre"] - games["B_Elo_pre"]
    for b in prior_bases:
        games[f"d_{b}_prev"] = games[f"A_{b}_prev"] - games[f"B_{b}_prev"]
        # NaN propagates when a side has no prior (first-season team): a half-sum would mislead the
        # total head; the trainer zero-fills NaN uniformly, which is the consistent "missing" signal.
        games[f"s_{b}_prev"] = games[f"A_{b}_prev"] + games[f"B_{b}_prev"]
    for f in bs_feats:  # GM-003: d_ margin (net eff), s_ total (pace × scoring level)
        games[f"d_{f}"] = games[f"A_{f}"] - games[f"B_{f}"]
        games[f"s_{f}"] = games[f"A_{f}"] + games[f"B_{f}"]  # NaN propagates (same reason as *_prev)

    # WM-004: maturity-weighted prior→as-of blend — leans on the prior-season prior early (few games)
    # and the within-season rating as games accumulate. Matches WM-002's early-season signal (corr
    # 0.71) at a fraction of the cost. Only when both components are present (rolling built).
    if "bs_NetEff_asof" in bs_feats and "AdjEM" in prior_bases:
        _add_blend(games, asof="bs_NetEff_asof", prior="AdjEM_prev", out="blend_NetEff_asof")
    return games


def _add_blend(games: pd.DataFrame, asof: str, prior: str, out: str, k: float = 8.0) -> None:
    """Add ``d_<out>`` = A/B maturity-weighted blend of within-season ``asof`` and prior-season ``prior``.

    Weight ``w = min(games/k, 1)`` on the as-of; ``w=0`` when the as-of is missing (full prior),
    ``w=1`` when the prior is missing (full as-of, e.g. first-season teams).
    """
    for side in ("A", "B"):
        a = games[f"{side}_{asof}"]
        p = games[f"{side}_{prior}"]
        gp = games[f"{side}_games_asof"].to_numpy(dtype=float)
        w = np.minimum(gp / k, 1.0)
        w = np.where(a.notna().to_numpy(), w, 0.0)
        w = np.where(p.isna().to_numpy() & a.notna().to_numpy(), 1.0, w)
        games[f"{side}_{out}"] = w * a.fillna(0).to_numpy() + (1 - w) * p.fillna(0).to_numpy()
    games[f"d_{out}"] = games[f"A_{out}"] - games[f"B_{out}"]


def build_reg_games(
    data: dict[str, pd.DataFrame],
    adj_eff: pd.DataFrame,
    asof_snapshots: pd.DataFrame | None = None,
    dayzero_by_season: dict | None = None,
    torvik_women_snapshots: pd.DataFrame | None = None,
    adjself_snapshots: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the combined men's + women's regular-season game-level dataset.

    Args:
        data: The raw-CSV dict (needs ``M_reg_raw``/``W_reg_raw``), as built by the features stage.
        adj_eff: Output of :func:`compute_adj_efficiency` (all seasons, both genders).
        asof_snapshots: Optional long frame of as-of-date KenPom snapshots (GM-002), from
            :func:`cbb.kenpom.asof_features.load_all_archive_snapshots`. When given (with
            ``dayzero_by_season``), within-season ``*_kp_*_asof`` features are joined in. As of
            WM-002 the reg pipeline passes ``None`` here (KenPom dropped from the model in favour of
            the self-computed ``adjself_*_asof``); the plumbing stays for ad-hoc comparison.
        dayzero_by_season: ``Season → DayZero`` (from ``MSeasons``), to turn ``DayNum`` into a
            calendar date for the as-of join. Required alongside any as-of snapshots.
        adjself_snapshots: Optional long frame of *self-computed* weekly as-of efficiency snapshots
            (WM-002), from :func:`cbb.features.adjself_asof.compute_adjself_asof_snapshots`. Joined
            as ``*_adjself_*_asof`` — the reg model's within-season strength signal (both genders).

    Returns:
        Concatenated symmetric reg-season game dataset (see :func:`build_reg_game_dataset`).
    """
    from .elo import compute_pregame_elo  # noqa: PLC0415
    from .rolling import compute_rolling_boxscore  # noqa: PLC0415

    parts = []
    for key, mw in (("M_reg_raw", 0), ("W_reg_raw", 1)):
        reg_raw = data[key]
        # k=60/carry=0.9 tuned for reg-season pre-game Elo (vs the tournament Elo's k=100/carry=0.75):
        # less-aggressive updates + higher season carryover (college teams are consistent year-to-year)
        # → best corr(margin) for both genders.
        pregame = compute_pregame_elo(reg_raw, men_women_flag=mw, k_factor=60.0, carry=0.9)
        rolling = compute_rolling_boxscore(reg_raw, men_women_flag=mw)  # GM-003: as-of box-score, both genders
        parts.append(build_reg_game_dataset(reg_raw, pregame, adj_eff, men_women=mw, rolling=rolling))
    games = pd.concat(parts, ignore_index=True)

    # As-of-date ratings — shared merge_asof pipeline, additive, no-op when snapshots absent:
    # KenPom kp_*_asof (GM-002, men) and BartTorvik tv_*_asof (WM-003, women 2025+).
    if dayzero_by_season:
        import logging  # noqa: PLC0415

        from cbb.kenpom.asof_features import add_asof_features  # noqa: PLC0415
        log = logging.getLogger(__name__)
        for label, snaps in (("KenPom", asof_snapshots), ("Torvik-women", torvik_women_snapshots),
                             ("adjself", adjself_snapshots)):
            if snaps is not None and len(snaps):
                games, added = add_asof_features(games, snaps, dayzero_by_season)
                if added:
                    log.info("As-of %s features joined into reg_games: %d cols", label, len(added))

    _add_exptotal(games)  # totals are multiplicative (pace × efficiency), not additive sums
    return games


# Rating sources → (OE, DE, Tempo) as-of column bases for the pace×efficiency total.
_EXPTOT_SOURCES = {
    "bs": ("bs_OE_asof", "bs_DE_asof", "bs_Tempo_asof"),
    "kp": ("kp_AdjOE_asof", "kp_AdjDE_asof", "kp_AdjTempo_asof"),
    "tv": ("tv_AdjOE_asof", "tv_AdjDE_asof", "tv_AdjTempo_asof"),
    "adjself": ("adjself_AdjOE_asof", "adjself_AdjDE_asof", "adjself_AdjTempo_asof"),
}


def _add_exptotal(games: pd.DataFrame) -> None:
    """Add ``s_exptot_<src>_asof`` — an explicit **pace × opponent-adjusted efficiency** game total.

    The total head otherwise sees only additive sums (``s_OE + s_DE + s_Tempo``), but a game's total
    is multiplicative: ``pace × (A's PPP vs B's D + B's PPP vs A's D)``. Giving the head this product
    directly closes the total gap to FanMatch (men 14.0→13.8 ≈ FanMatch's 13.8). One per rating
    source; the head picks the best available (kp men / tv women / bs everywhere).
    """
    for src, (o, d, t) in _EXPTOT_SOURCES.items():
        if f"A_{o}" not in games.columns:
            continue
        aO, bO = games[f"A_{o}"], games[f"B_{o}"]
        aD, bD = games[f"A_{d}"], games[f"B_{d}"]
        pace = (games[f"A_{t}"] + games[f"B_{t}"]) / 2.0
        lg_oe = games.groupby(["Season", "men_women"])[f"A_{o}"].transform("mean")
        games[f"s_exptot_{src}_asof"] = pace / 100.0 * (aO * bD + bO * aD) / lg_oe
