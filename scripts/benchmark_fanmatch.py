"""GM-004: benchmark our reg-season model against KenPom FanMatch on 2026 men's games.

Both predictors are as-of-date, so this is a fair head-to-head on score prediction. Reports
FanMatch vs our model on the **same** matched games (margin MAE / total MAE / Brier), plus two
diagnostics the benchmark design called for: home-or-away vs neutral, and our error by snapshot
staleness (our as-of is weekly; FanMatch is exact-as-of, so staleness is our handicap).

    python scripts/benchmark_fanmatch.py
"""
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import sys  # noqa: E402

sys.path.insert(0, "src")
from cbb.benchmark.fanmatch_bench import match_fanmatch_to_results, score_predictions  # noqa: E402
from cbb.kenpom import KenPomClient  # noqa: E402
from cbb.kenpom.features import build_team_name_map  # noqa: E402

SEASON = 2026
RAW, PROC, KP = Path("data/raw"), Path("data/processed"), Path("data/kenpom")


def _our_predictions(matched: pd.DataFrame) -> pd.DataFrame:
    """Inner-join the model's home-as-A reg_games row for each matched game, score it."""
    rg = pd.read_parquet(PROC / "reg_games.parquet")
    rg = rg[(rg.Season == SEASON) & (rg.men_women == 0)]
    key = ["DayNum", "A_TeamID", "B_TeamID"]
    m = matched.merge(
        rg, left_on=["DayNum", "home_id", "vis_id"], right_on=key, how="inner", suffixes=("", "_rg")
    )
    model = pickle.load(open(PROC / "reg_model.pkl", "rb"))
    feats = model.margin_features + model.total_features
    m[feats] = m[feats].fillna(0)
    ps = model.predict_scores(m)
    m["our_margin"] = ps["pred_margin"].to_numpy()
    m["our_total"] = ps["pred_total"].to_numpy()
    m["our_home_wp"] = model.predict_batch(m)
    return m


def _staleness(m: pd.DataFrame, dayzero: pd.Timestamp) -> pd.Series:
    """Days between each game and the most recent archive snapshot strictly before it."""
    arch = pd.read_parquet(KP / "archive" / f"kenpom_archive_{SEASON}.parquet")
    snap_days = np.sort(((pd.to_datetime(arch["ArchiveDate"].unique()) - dayzero).days))
    def stale(daynum):
        prior = snap_days[snap_days < daynum]
        return int(daynum - prior.max()) if len(prior) else np.nan
    return m["DayNum"].map(stale)


def _line(label, s):
    print(f"  {label:16} margin_MAE {s['margin_mae']:6.3f}   total_MAE {s['total_mae']:6.3f}   "
          f"Brier {s['brier']:.4f}   logistic {s['logistic']:.4f}   win% {s['acc']:.3f}   (n={s['n']})")


def main() -> None:
    fanmatch = pd.read_parquet(KP / "fanmatch" / f"fanmatch_{SEASON}.parquet")
    results = pd.read_csv(RAW / "MRegularSeasonDetailedResults.csv")
    results = results[results.Season == SEASON]
    ms = pd.read_csv(RAW / "MSeasons.csv")
    dayzero = pd.to_datetime(ms[ms.Season == SEASON].DayZero.iloc[0])

    spell = pd.read_csv(RAW / "MTeamSpellings.csv", encoding="latin-1")
    tmap = build_team_name_map(pd.read_csv(RAW / "MTeams.csv"), KenPomClient().teams(year=SEASON), spell)

    matched, counts = match_fanmatch_to_results(fanmatch, results, tmap, dayzero)
    m = _our_predictions(matched)
    print("Per-stage N (same-games discipline):")
    for k, v in counts.items():
        print(f"  {k:18} {v}")
    print(f"  {'has_model_row':18} {len(m)}   ← the benchmark set both predictors score on\n")

    def both(df, label):
        print(label)
        _line("FanMatch", score_predictions(df, df.fm_margin, df.fm_total, df.fm_home_wp))
        _line("Ours (reg model)", score_predictions(df, df.our_margin, df.our_total, df.our_home_wp))

    both(m, "=== 2026 men's reg season — head to head ===")
    print()
    both(m[~m.neutral], "--- home/away games only ---")
    both(m[m.neutral], "--- neutral-site games only ---")

    print("\n--- error by snapshot staleness: ours vs FanMatch (gap = ours − fanmatch) ---")
    m["stale"] = _staleness(m, dayzero)
    buckets = [("0–2d", m.stale.between(0, 2)), ("3–5d", m.stale.between(3, 5)),
               ("6–8d", m.stale.between(6, 8)), ("pre-snapshot", m.stale.isna())]
    print(f"  {'bucket':14} {'n':>5}   {'margin gap':>10}   {'total gap':>10}   {'brier gap':>10}")
    for name, mask in buckets:
        d = m[mask]
        if len(d):
            mgap = np.abs(d.our_margin - d.act_margin).mean() - np.abs(d.fm_margin - d.act_margin).mean()
            tgap = np.abs(d.our_total - d.act_total).mean() - np.abs(d.fm_total - d.act_total).mean()
            bgap = ((d.our_home_wp - d.home_won) ** 2).mean() - ((d.fm_home_wp - d.home_won) ** 2).mean()
            print(f"  {name:14} {len(d):5d}   {mgap:>+10.2f}   {tgap:>+10.2f}   {bgap:>+10.4f}")


if __name__ == "__main__":
    main()
