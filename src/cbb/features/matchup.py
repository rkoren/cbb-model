"""Build symmetric A/B matchup dataset from pre-computed feature frames."""

import numpy as np
import pandas as pd
from tqdm import tqdm

_AE_COLS = ["AdjOE", "AdjDE", "AdjEM", "AdjTempo"]

_AVG_COLS = [
    "avg_Score", "avg_PointDiff", "avg_FGM", "avg_FGA", "avg_FGM3", "avg_FGA3",
    "avg_OR", "avg_Ast", "avg_TO", "avg_Stl", "avg_Blk", "avg_opp_Score",
    "eFGpct", "TOpct", "ORpct", "FTrate", "TSpct",
    "opp_eFGpct", "opp_TOpct", "opp_ORpct", "opp_FTrate", "opp_TSpct",
    "FG3pct", "opp_FG3pct", "FG2pct", "opp_FG2pct", "FTpct", "opp_FTpct",
    "3PArate", "opp_3PArate", "ASTrate", "opp_ASTrate",
    "Blkpct", "Stlpct", "NonStlTOpct", "opp_NonStlTOpct",
    "pct_pts_3", "pct_pts_2", "pct_pts_ft",
    "opp_pct_pts_3", "opp_pct_pts_2", "opp_pct_pts_ft",
]

_SCALAR_FEATS = ["Elo", "Quality", "Form", "Seed", "MasseyRank"]

FEAT_COLS = _AE_COLS + _AVG_COLS + _SCALAR_FEATS


def _safe_get(lookup: dict, key, default=np.nan):
    v = lookup.get(key, default)
    return v if not isinstance(v, (pd.Series, pd.DataFrame)) else default


def compute_quality_wtd_margin(
    reg_sym: pd.DataFrame,
    adj_eff: pd.DataFrame,
) -> pd.DataFrame:
    """Quality-weighted scoring margin: each game weighted by opponent AdjEM percentile.

    Avoids hard top-N schedule strength cutoffs. Teams on weak schedules are naturally
    down-weighted; blowout wins over cupcakes contribute almost nothing.

    Args:
        reg_sym: Symmetric regular-season game log (men_women, Season, T1_TeamID,
                 T2_TeamID, PointDiff).
        adj_eff: Output of compute_adj_efficiency() (needs AdjEM column).

    Returns:
        DataFrame with columns: Season, men_women, TeamID, quality_wtd_margin.
    """
    ae_pct = adj_eff[["Season", "men_women", "TeamID", "AdjEM"]].copy()
    ae_pct["AdjEM_pct"] = ae_pct.groupby(["Season", "men_women"])["AdjEM"].rank(pct=True)

    qwm = reg_sym[["Season", "men_women", "T1_TeamID", "T2_TeamID", "PointDiff"]].copy()
    qwm = qwm.merge(
        ae_pct[["Season", "men_women", "TeamID", "AdjEM_pct"]].rename(
            columns={"TeamID": "T2_TeamID", "AdjEM_pct": "opp_AdjEM_pct"}
        ),
        on=["Season", "men_women", "T2_TeamID"],
        how="left",
    )
    qwm["opp_AdjEM_pct"] = qwm["opp_AdjEM_pct"].fillna(0.5)
    qwm = qwm.rename(columns={"T1_TeamID": "TeamID"})

    def _wtd_mean(grp: pd.DataFrame) -> float:
        w = grp["opp_AdjEM_pct"].values
        m = grp["PointDiff"].values
        return float(np.dot(w, m) / (w.sum() + 1e-9))

    return (
        qwm.groupby(["Season", "men_women", "TeamID"])
        .apply(_wtd_mean, include_groups=False)
        .reset_index(name="quality_wtd_margin")
    )


def compute_massey_ranks(
    massey_raw: pd.DataFrame,
    systems: list[str] | None = None,
    sel_daynum: int = 133,
) -> dict[tuple[int, int], float]:
    """Consensus Massey ordinal ranking (z-scored, averaged across systems).

    Uses ratings at or before Selection Sunday (DayNum 133) to avoid leakage.
    Men's only — Women's teams receive 0.0 at matchup build time.

    Args:
        massey_raw: MMasseyOrdinals.csv loaded as DataFrame.
        systems: System names to include. Defaults to ['POM', 'MOR'].
        sel_daynum: Last DayNum to include (Selection Sunday).

    Returns:
        Dict mapping (Season, TeamID) → mean z-scored rank (higher = better).
    """
    systems = systems or ["POM", "MOR"]
    sel = massey_raw[massey_raw["SystemName"].isin(systems)].copy()
    pre = sel[sel["RankingDayNum"] <= sel_daynum]
    last = (
        pre.sort_values("RankingDayNum")
        .groupby(["Season", "SystemName", "TeamID"])
        .last()
        .reset_index()
    )
    last["neg_rank"] = -last["OrdinalRank"]
    last["rank_z"] = last.groupby(["Season", "SystemName"])["neg_rank"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0)
    )
    team_massey = (
        last.groupby(["Season", "TeamID"])["rank_z"]
        .mean()
        .reset_index()
        .rename(columns={"rank_z": "MasseyRank"})
    )
    return team_massey.set_index(["Season", "TeamID"])["MasseyRank"].to_dict()


def build_matchup_dataset(
    tourn_sym: pd.DataFrame,
    reg_sym: pd.DataFrame,
    adj_eff: pd.DataFrame,
    season_avgs: pd.DataFrame,
    elo_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    form_df: pd.DataFrame,
    seed_lookup: dict[tuple[int, int], int],
    massey_lookup: dict[tuple[int, int], float] | None = None,
    nil_year: int = 2022,
    nil_multiplier: float = 1.5,
    apply_nil_weights: bool = False,
) -> pd.DataFrame:
    """Build symmetric A/B matchup dataset for model training.

    For each historical tournament game, produces two rows (winner-as-A and loser-as-A)
    with absolute features (A_*, B_*) and differentials (d_* = A - B).

    Args:
        tourn_sym: Symmetric tournament game log (same format as reg_sym).
        reg_sym: Symmetric regular-season game log (for quality_wtd_margin).
        adj_eff: Output of compute_adj_efficiency().
        season_avgs: Output of add_four_factors(compute_season_averages()).
        elo_df: Output of compute_elo() for both genders concatenated.
        quality_df: Output of compute_glm_quality().
        form_df: Output of compute_recent_form().
        seed_lookup: Dict mapping (Season, TeamID) → seed number.
        massey_lookup: Dict mapping (Season, TeamID) → Massey z-score. Men's only.
        nil_year: First season with NIL rules (for optional recency weighting).
        nil_multiplier: Sample weight multiplier for post-NIL seasons.
        apply_nil_weights: If True, ramp sample weights from nil_multiplier → 2x
                           for post-NIL seasons. Default False (uniform weights).

    Returns:
        DataFrame with A_*, B_*, d_* feature columns, Outcome, PointDiff,
        sample_weight, and quality_wtd_margin differentials.
    """
    ae_lkp = adj_eff.set_index(["men_women", "Season", "TeamID"])
    avg_lkp = season_avgs.set_index(["men_women", "Season", "TeamID"])
    elo_lkp = elo_df.set_index(["men_women", "Season", "TeamID"])["Elo"].to_dict()
    qual_lkp = quality_df.set_index(["men_women", "Season", "TeamID"])["Quality"].to_dict()
    form_lkp = form_df.set_index(["men_women", "Season", "TeamID"])["RecentWinPct"].to_dict()
    massey_lookup = massey_lookup or {}

    def _get_features(mw: int, season: int, team: int) -> dict:
        key = (mw, season, team)
        feats: dict = {}
        try:
            ae = ae_lkp.loc[key]
            for c in _AE_COLS:
                feats[c] = float(ae[c]) if c in ae.index else np.nan
        except KeyError:
            for c in _AE_COLS:
                feats[c] = np.nan
        try:
            av = avg_lkp.loc[key]
            for c in _AVG_COLS:
                feats[c] = float(av[c]) if c in av.index else np.nan
        except KeyError:
            for c in _AVG_COLS:
                feats[c] = np.nan
        feats["Elo"] = _safe_get(elo_lkp, key)
        feats["Quality"] = _safe_get(qual_lkp, key)
        feats["Form"] = _safe_get(form_lkp, key)
        feats["Seed"] = _safe_get(seed_lookup, (season, team))
        # Massey is Men's only; Women's receive 0 (neutral) so the feature stays useful
        feats["MasseyRank"] = _safe_get(massey_lookup, (season, team), 0.0) if mw == 0 else 0.0
        return feats

    records = []
    for _, row in tqdm(tourn_sym.iterrows(), total=len(tourn_sym), desc="Matchups"):
        mw, season = int(row["men_women"]), int(row["Season"])
        ta, tb = int(row["T1_TeamID"]), int(row["T2_TeamID"])
        fa = _get_features(mw, season, ta)
        fb = _get_features(mw, season, tb)

        rec: dict = {
            "Season": season,
            "men_women": mw,
            "A_TeamID": ta,
            "B_TeamID": tb,
            "Outcome": int(row["Outcome"]),
            "PointDiff": float(row["PointDiff"]),
        }
        for c in FEAT_COLS:
            rec[f"A_{c}"] = fa.get(c, np.nan)
            rec[f"B_{c}"] = fb.get(c, np.nan)
            rec[f"d_{c}"] = fa.get(c, np.nan) - fb.get(c, np.nan)
        records.append(rec)

    matchups = pd.DataFrame(records)

    # Sample weights
    seasons = matchups["Season"].values
    weights = np.ones(len(seasons))
    if apply_nil_weights:
        max_s = seasons.max()
        mask = seasons >= nil_year
        if mask.sum() > 0:
            recency = (seasons[mask] - nil_year) / max(1, max_s - nil_year)
            weights[mask] = nil_multiplier + recency * nil_multiplier
    matchups["sample_weight"] = weights

    # Quality-weighted margin differential
    qwm = compute_quality_wtd_margin(reg_sym, adj_eff)
    qwm_lkp_a = qwm.rename(columns={"TeamID": "A_TeamID", "quality_wtd_margin": "A_quality_wtd_margin"})
    qwm_lkp_b = qwm.rename(columns={"TeamID": "B_TeamID", "quality_wtd_margin": "B_quality_wtd_margin"})
    matchups = matchups.merge(qwm_lkp_a, on=["Season", "men_women", "A_TeamID"], how="left")
    matchups = matchups.merge(qwm_lkp_b, on=["Season", "men_women", "B_TeamID"], how="left")
    matchups["d_quality_wtd_margin"] = matchups["A_quality_wtd_margin"] - matchups["B_quality_wtd_margin"]

    return matchups
