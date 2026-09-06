"""Kaggle competition submissions from the tuned ensemble recipe (HAND-008).

Scores every pair in the season's tournament field with BOTH models via their exact production feature
paths, blends them, and writes the two Kaggle submissions:
  * main comp   → P(lower-ID team wins)      = w·tournament_prob + (1−w)·reg_prob
  * logistic    → spread for lower-ID team   = w·(α_t·tournament_margin) + (1−w)·(α_r·reg_margin)
Recipe (22-tournament LOTO CV, 2026-07-19): ensemble beats both models on both metrics (Brier 0.1648→0.1641,
logistic 0.0609→0.0603); optimum flat over w∈[0.4,0.7], default 0.6 = the CV Brier optimum (0.1640). NOTE: an earlier hedge toward
reg came from a mid-tournament-updated 2026 read; under the honest pre-tournament construction below the
tournament model is slightly better (esp. women), and 0.6 agrees; α-shrinks nested-CV
(tournament 0.685, reg 0.82). Non-field pairs (never scored) default to 0.5 / 0.0.

    python scripts/competition_submission.py 2026            # build (+ validates vs actuals if present)
    python scripts/competition_submission.py 2027 --w 0.6
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from cbb.data import normalize_games  # noqa: E402
from cbb.dashboard.wiring import dedupe_symmetric  # noqa: E402
from cbb.features import build_prediction_features  # noqa: E402
from cbb.features.reg_games import build_reg_games  # noqa: E402
from cbb.features.torvik_asof import load_all_torvik_women  # noqa: E402
from cbb.kenpom.features import build_team_name_map  # noqa: E402
from cbb.kenpom.rich_features import build_kenpom_rich_features, join_kenpom_rich  # noqa: E402

RAW, PROC, KP, LIVE, OUT = Path("data/raw"), Path("data/processed"), Path("data/kenpom"), Path("data/live"), Path("data/submissions")
ALPHA_TOURN, ALPHA_REG, TOURN_DAYNUM, SCOPE = 0.685, 0.82, 134, 5


def _rd(n, e=None):
    return pd.read_csv(RAW / f"{n}.csv", encoding=e)


def field_pairs(season: int) -> pd.DataFrame:
    seeds = pd.concat([_rd("MNCAATourneySeeds").assign(men_women=0), _rd("WNCAATourneySeeds").assign(men_women=1)])
    seeds = seeds[seeds.Season == season]
    rows = []
    for gg, grp in seeds.groupby("men_women"):
        ids = sorted(grp.TeamID.unique())
        rows += [(season, gg, a, b) for i, a in enumerate(ids) for b in ids[i + 1:]]
    return pd.DataFrame(rows, columns=["Season", "men_women", "A_TeamID", "B_TeamID"])


def tournament_preds(pairs: pd.DataFrame, season: int) -> pd.DataFrame:
    """Prod tournament model on the pairs via its exact submission feature path (offline)."""
    sl = pd.read_parquet(PROC / "seed_lookup.parquet"); ml = pd.read_parquet(PROC / "massey_lookup.parquet")
    reg_sym = pd.concat([normalize_games(_rd("MRegularSeasonDetailedResults"), men_women=0),
                         normalize_games(_rd("WRegularSeasonDetailedResults"), men_women=1)], ignore_index=True)
    pred = build_prediction_features(
        pairs, pd.read_parquet(PROC / "adj_eff.parquet"), pd.read_parquet(PROC / "season_avgs.parquet"),
        pd.read_parquet(PROC / "elo_df.parquet"), pd.read_parquet(PROC / "quality_df.parquet"),
        pd.read_parquet(PROC / "form_df.parquet"),
        {(int(r.Season), int(r.TeamID)): int(r.SeedNum) for r in sl.itertuples()},
        {(int(r.Season), int(r.TeamID)): float(r.MasseyRank) for r in ml.itertuples()}, reg_sym)
    meta = json.load(open(PROC / "rating_meta.json")); z_cols = meta["z_cols"]; z_feats = [c[:-2] for c in z_cols]
    scaler = pickle.load(open(PROC / "scaler.pkl", "rb"))
    pred[z_cols] = scaler.transform(pred[z_feats].fillna(0))
    pred["d_Rating"] = (pred[z_cols] * np.asarray(meta["opt_weights"])).sum(axis=1)
    arch = pd.read_parquet(KP / "archive" / f"kenpom_archive_{season}.parquet")[["TeamName"]].drop_duplicates()
    tmap = build_team_name_map(_rd("MTeams"), arch, _rd("MTeamSpellings", "latin-1"))
    hp = KP / f"kenpom_height_{season}.parquet"
    kp_rich = build_kenpom_rich_features(season, tmap, None, pd.read_parquet(hp) if hp.exists() else None)
    pred, _ = join_kenpom_rich(pred, kp_rich)
    cbb = pickle.load(open(PROC / "cbb_model.pkl", "rb"))
    for f in cbb.features:
        if f not in pred.columns:
            pred[f] = 0.0
    out = pairs.copy()
    out["tourn_wp"] = cbb.predict_batch(pred)                     # logistic + temperature (prod calibration)
    out["tourn_margin"] = cbb.predict_scores(pred).pred_margin.to_numpy()
    return out


def reg_preds(pairs: pd.DataFrame, season: int) -> pd.DataFrame:
    """Reg v11 on the pairs via the result-less synthetic-row build (neutral, Selection-Sunday-as-of)."""
    data = {"M_reg_raw": _rd("MRegularSeasonDetailedResults"), "W_reg_raw": _rd("WRegularSeasonDetailedResults"),
            "M_teams": _rd("MTeams"), "W_teams": _rd("WTeams")}
    ms = _rd("MSeasons"); dz = dict(zip(ms.Season, pd.to_datetime(ms.DayZero)))
    lo = season - SCOPE + 1
    live = LIVE / f"mreg_live_{season}.csv"
    mreg = data["M_reg_raw"]
    if live.exists():
        mreg = pd.concat([mreg, pd.read_csv(live)], ignore_index=True).drop_duplicates(["Season", "DayNum", "WTeamID", "LTeamID"], keep="first")
    mreg = mreg[(mreg.Season >= lo) & ~((mreg.Season == season) & (mreg.DayNum >= TOURN_DAYNUM))]
    wreg = data["W_reg_raw"]; wreg = wreg[(wreg.Season >= lo) & ~((wreg.Season == season) & (wreg.DayNum >= TOURN_DAYNUM))]
    syn = pairs.rename(columns={"A_TeamID": "WTeamID", "B_TeamID": "LTeamID"}).assign(
        DayNum=TOURN_DAYNUM, WScore=100, LScore=99, WLoc="N", NumOT=0)
    comb = dict(data)
    comb["M_reg_raw"] = pd.concat([mreg, syn[syn.men_women == 0].drop(columns="men_women")], ignore_index=True)
    comb["W_reg_raw"] = pd.concat([wreg, syn[syn.men_women == 1].drop(columns="men_women")], ignore_index=True)
    seasons = list(range(lo, season + 1))
    wspell = _rd("WTeamSpellings", "latin-1"); w_maps = {}
    for s in seasons:
        p = Path("data/torvik") / f"torvik_women_{s}.parquet"
        if p.exists():
            w_maps[s] = build_team_name_map(data["W_teams"], pd.read_parquet(p)[["team"]].drop_duplicates().rename(columns={"team": "TeamName"}), wspell)
    tv = load_all_torvik_women(seasons, Path("data/torvik"), w_maps)
    games = build_reg_games(comb, pd.read_parquet(PROC / "adj_eff.parquet"), asof_snapshots=None, dayzero_by_season=dz,
                            torvik_women_snapshots=tv, adjself_snapshots=pd.read_parquet(PROC / "adjself_asof.parquet"))
    t = dedupe_symmetric(games[(games.Season == season) & (games.DayNum == TOURN_DAYNUM)]).copy()
    model = pickle.load(open(PROC / "reg_model.pkl", "rb")); feats = model.margin_features + model.total_features
    for f in feats:
        if f not in t.columns:
            t[f] = 0.0
    t[feats] = t[feats].fillna(0)
    t["reg_margin"] = model.predict_scores(t).pred_margin.to_numpy(); t["reg_wp"] = model.predict_batch(t)
    return t[["Season", "men_women", "A_TeamID", "B_TeamID", "reg_margin", "reg_wp"]]


def build(season: int, w: float) -> pd.DataFrame:
    pairs = field_pairs(season)
    p = tournament_preds(pairs, season).merge(reg_preds(pairs, season), on=["Season", "men_women", "A_TeamID", "B_TeamID"], how="left")
    p["prob"] = w * p.tourn_wp + (1 - w) * p.reg_wp.fillna(p.tourn_wp)
    p["spread"] = w * ALPHA_TOURN * p.tourn_margin + (1 - w) * ALPHA_REG * p.reg_margin.fillna(p.tourn_margin / ALPHA_REG * ALPHA_TOURN)
    p["ID"] = p.Season.astype(str) + "_" + p.A_TeamID.astype(str) + "_" + p.B_TeamID.astype(str)
    return p


def write(p: pd.DataFrame, season: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sample = RAW / "SampleSubmissionStage2.csv"
    ids = pd.read_csv(sample)[["ID"]] if sample.exists() and pd.read_csv(sample, nrows=1).ID.iloc[0].startswith(str(season)) else p[["ID"]]
    for name, col, default in (("main", "prob", 0.5), ("logistic", "spread", 0.0)):
        sub = ids.merge(p[["ID", col]], on="ID", how="left").rename(columns={col: "Pred"})
        sub["Pred"] = sub.Pred.fillna(default).round(5)
        sub.to_csv(OUT / f"submission_{name}_{season}.csv", index=False)
        print(f"wrote {OUT / f'submission_{name}_{season}.csv'}: {len(sub):,} rows ({sub.Pred.ne(default).sum():,} field pairs scored)")


def validate(p: pd.DataFrame, season: int) -> None:
    """Score against actual results if present (2026: hand-sourced men + women)."""
    frames = []
    for f, gg in ((f"tourney_results_{season}.csv", 0), (f"tourney_results_{season}_women.csv", 1)):
        fp = Path("data/holdout") / f
        if fp.exists():
            r = pd.read_csv(fp); r["men_women"] = gg; frames.append(r)
    if not frames:
        print("no actual results on disk — skipping validation"); return
    r = pd.concat(frames); r["A_TeamID"] = np.minimum(r.WTeamID, r.LTeamID); r["B_TeamID"] = np.maximum(r.WTeamID, r.LTeamID)
    r["y"] = (r.WTeamID == r.A_TeamID).astype(int); r["z"] = np.where(r.WTeamID == r.A_TeamID, r.WScore - r.LScore, r.LScore - r.WScore)
    j = r.merge(p, on=["men_women", "A_TeamID", "B_TeamID"], how="inner")
    c = 7.0; L = lambda x: 1 / (1 + np.exp(-x / c))  # noqa: E731
    print(f"\n=== VALIDATION vs actual {season} bracket ({len(j)} games) — win-Brier / logistic-spread ===")
    for lbl, mk in (("men", j.men_women == 0), ("women", j.men_women == 1), ("ALL", j.men_women >= 0)):
        d = j[mk]
        if len(d):
            print(f"  {lbl:6} n={len(d):3}  ensemble  Brier {np.mean((d.prob - d.y) ** 2):.4f}  logistic {np.mean((L(d.spread) - L(d.z)) ** 2):.4f}"
                  f"   | tourn-only {np.mean((d.tourn_wp - d.y) ** 2):.4f}/{np.mean((L(ALPHA_TOURN * d.tourn_margin) - L(d.z)) ** 2):.4f}"
                  f"   reg-only {np.mean((d.reg_wp - d.y) ** 2):.4f}/{np.mean((L(ALPHA_REG * d.reg_margin) - L(d.z)) ** 2):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("season", type=int); ap.add_argument("--w", type=float, default=0.6)
    a = ap.parse_args()
    p = build(a.season, a.w)
    print(f"{a.season}: {len(p):,} field pairs scored (w_tourn={a.w}); reg coverage {p.reg_wp.notna().mean():.0%}")
    write(p, a.season); validate(p, a.season)


if __name__ == "__main__":
    main()
