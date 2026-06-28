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

# Sum (level) features for the total head (SC-001). Game totals track the SUM of both teams'
# pace/efficiency, not the differentials the margin head uses — so the total head gets its own
# s_<base> = A_<base> + B_<base> columns (added wherever A_/B_ exist, so train/predict stay in parity).
TOTAL_SUM_BASES = ["AdjTempo", "AdjOE", "AdjDE", "AdjEM", "avg_Score", "avg_opp_Score"]


def add_total_sum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``s_<base> = A_<base> + B_<base>`` columns in place for the total head."""
    for base in TOTAL_SUM_BASES:
        a, b = f"A_{base}", f"B_{base}"
        if a in df.columns and b in df.columns:
            df[f"s_{base}"] = df[a].fillna(0) + df[b].fillna(0)
    return df


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


def _build_lookups(
    adj_eff: pd.DataFrame,
    season_avgs: pd.DataFrame,
    elo_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    form_df: pd.DataFrame,
) -> tuple:
    ae_lkp = adj_eff.set_index(["men_women", "Season", "TeamID"])
    avg_lkp = season_avgs.set_index(["men_women", "Season", "TeamID"])
    elo_lkp = elo_df.set_index(["men_women", "Season", "TeamID"])["Elo"].to_dict()
    qual_lkp = quality_df.set_index(["men_women", "Season", "TeamID"])["Quality"].to_dict()
    form_lkp = form_df.set_index(["men_women", "Season", "TeamID"])["RecentWinPct"].to_dict()
    return ae_lkp, avg_lkp, elo_lkp, qual_lkp, form_lkp


def get_team_features(
    mw: int,
    season: int,
    team: int,
    ae_lkp,
    avg_lkp,
    elo_lkp: dict,
    qual_lkp: dict,
    form_lkp: dict,
    seed_lookup: dict,
    massey_lookup: dict | None = None,
) -> dict:
    """Look up all features for a single team-season. Returns dict of raw values."""
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
    massey_lookup = massey_lookup or {}
    # Massey is Men's only; Women's receive 0 (neutral) so the feature stays useful
    feats["MasseyRank"] = _safe_get(massey_lookup, (season, team), 0.0) if mw == 0 else 0.0
    return feats


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
    ae_lkp, avg_lkp, elo_lkp, qual_lkp, form_lkp = _build_lookups(
        adj_eff, season_avgs, elo_df, quality_df, form_df
    )
    massey_lookup = massey_lookup or {}

    records = []
    for _, row in tqdm(tourn_sym.iterrows(), total=len(tourn_sym), desc="Matchups"):
        mw, season = int(row["men_women"]), int(row["Season"])
        ta, tb = int(row["T1_TeamID"]), int(row["T2_TeamID"])
        fa = get_team_features(mw, season, ta, ae_lkp, avg_lkp, elo_lkp, qual_lkp, form_lkp, seed_lookup, massey_lookup)
        fb = get_team_features(mw, season, tb, ae_lkp, avg_lkp, elo_lkp, qual_lkp, form_lkp, seed_lookup, massey_lookup)

        rec: dict = {
            "Season": season,
            "men_women": mw,
            "A_TeamID": ta,
            "B_TeamID": tb,
            "Outcome": int(row["Outcome"]),
            "PointDiff": float(row["PointDiff"]),       # margin = ScoreA - ScoreB (A=T1)
            "Total": float(row["T1_Score"]) + float(row["T2_Score"]),  # ScoreA + ScoreB (SC-001)
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

    add_total_sum_features(matchups)
    return matchups


def build_prediction_features(
    pairs: pd.DataFrame,
    adj_eff: pd.DataFrame,
    season_avgs: pd.DataFrame,
    elo_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    form_df: pd.DataFrame,
    seed_lookup: dict[tuple[int, int], int],
    massey_lookup: dict[tuple[int, int], float] | None = None,
    reg_sym: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build A/B matchup features for prediction (no outcome columns).

    Same feature extraction as build_matchup_dataset but for unseen matchups.
    Typically used to generate Kaggle submission predictions.

    Args:
        pairs: DataFrame with Season, men_women, A_TeamID, B_TeamID.
               For Kaggle format, A_TeamID < B_TeamID (lower ID first).
        adj_eff: Output of compute_adj_efficiency() (optionally KenPom-merged).
        season_avgs: Output of add_four_factors(compute_season_averages()).
        elo_df: Output of compute_elo() for both genders concatenated.
        quality_df: Output of compute_glm_quality().
        form_df: Output of compute_recent_form().
        seed_lookup: Dict mapping (Season, TeamID) → seed number.
        massey_lookup: Dict mapping (Season, TeamID) → Massey z-score.
        reg_sym: If provided, compute quality_wtd_margin differentials.

    Returns:
        DataFrame with A_*, B_*, d_* feature columns (no Outcome/PointDiff).
    """
    ae_lkp, avg_lkp, elo_lkp, qual_lkp, form_lkp = _build_lookups(
        adj_eff, season_avgs, elo_df, quality_df, form_df
    )
    massey_lookup = massey_lookup or {}

    records = []
    for _, row in pairs.iterrows():
        mw, season = int(row["men_women"]), int(row["Season"])
        ta, tb = int(row["A_TeamID"]), int(row["B_TeamID"])
        fa = get_team_features(mw, season, ta, ae_lkp, avg_lkp, elo_lkp, qual_lkp, form_lkp, seed_lookup, massey_lookup)
        fb = get_team_features(mw, season, tb, ae_lkp, avg_lkp, elo_lkp, qual_lkp, form_lkp, seed_lookup, massey_lookup)
        rec: dict = {"Season": season, "men_women": mw, "A_TeamID": ta, "B_TeamID": tb}
        for c in FEAT_COLS:
            rec[f"A_{c}"] = fa.get(c, np.nan)
            rec[f"B_{c}"] = fb.get(c, np.nan)
            rec[f"d_{c}"] = fa.get(c, np.nan) - fb.get(c, np.nan)
        records.append(rec)

    pred_df = pd.DataFrame(records)

    if reg_sym is not None:
        qwm = compute_quality_wtd_margin(reg_sym, adj_eff)
        qwm_a = qwm.rename(columns={"TeamID": "A_TeamID", "quality_wtd_margin": "A_quality_wtd_margin"})
        qwm_b = qwm.rename(columns={"TeamID": "B_TeamID", "quality_wtd_margin": "B_quality_wtd_margin"})
        pred_df = pred_df.merge(qwm_a, on=["Season", "men_women", "A_TeamID"], how="left")
        pred_df = pred_df.merge(qwm_b, on=["Season", "men_women", "B_TeamID"], how="left")
        pred_df["d_quality_wtd_margin"] = pred_df["A_quality_wtd_margin"] - pred_df["B_quality_wtd_margin"]

    add_total_sum_features(pred_df)
    return pred_df
