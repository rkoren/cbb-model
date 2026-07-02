"""GM-007: does the all-season as-of handicapper assess *tournament* matchups? (spread + moneyline)

The north-star exam ([[handicapper-north-star]]). Construction (leak-free): append tournament games
to the raw reg-season log, rebuild reg-games, and take rows with DayNum≥134 — their as-of features
are then computed on the full regular season (Selection-Sunday, no tournament results in the ratings),
and each row carries the actual Margin/Total/Outcome. We then score the reg model (`cbb-reg-model`)
as a tournament handicapper.

HONESTY RAILS (the point of this exercise):
  * **2003–2025** (Kaggle results): the reg model's OOS tournament performance — margin MAE + Brier.
    It never trained on ANY tournament game, but it *did* train on those seasons' reg games, so this
    is NOT a clean head-to-head with the LOTO tournament model — it's context.
  * **Round-1 only** is the strictly-fixed subset (no earlier tournament rounds in the features);
    later rounds carry mild mid-tournament rating updates (a small reg-model advantage) — reported
    separately so the fixed number is visible.
  * Women's tournament games are NOT all neutral (A/H/N) — split neutral vs home/away.

    python scripts/benchmark_tournament.py
"""
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import sys  # noqa: E402

sys.path.insert(0, "src")
from cbb.features.reg_games import build_reg_games  # noqa: E402
from cbb.features.torvik_asof import load_all_torvik_women  # noqa: E402
from cbb.kenpom import KenPomClient  # noqa: E402
from cbb.kenpom.asof_features import load_all_archive_snapshots  # noqa: E402
from cbb.kenpom.features import build_team_name_map  # noqa: E402
from cbb.train.model import _brier  # noqa: E402

RAW, PROC, KP, TV = Path("data/raw"), Path("data/processed"), Path("data/kenpom/archive"), Path("data/torvik")
TOURN_DAYNUM = 134  # tournament games start after Selection Sunday (~DayNum 133)


def _asof_inputs(data, seasons):
    ms = pd.read_csv(RAW / "MSeasons.csv")
    dayzero = dict(zip(ms["Season"], pd.to_datetime(ms["DayZero"])))
    client = KenPomClient()
    spell = pd.read_csv(RAW / "MTeamSpellings.csv", encoding="latin-1")
    m_maps = {s: build_team_name_map(data["M_teams"], client.teams(year=s), spell)
              for s in seasons if (KP / f"kenpom_archive_{s}.parquet").exists()}
    kp = load_all_archive_snapshots(seasons, KP, m_maps)
    wspell = pd.read_csv(RAW / "WTeamSpellings.csv", encoding="latin-1")
    w_maps = {}
    for s in seasons:
        p = TV / f"torvik_women_{s}.parquet"
        if p.exists():
            names = pd.read_parquet(p)[["team"]].drop_duplicates().rename(columns={"team": "TeamName"})
            w_maps[s] = build_team_name_map(data["W_teams"], names, wspell)
    tv = load_all_torvik_women(seasons, TV, w_maps)
    return kp, tv, dayzero


def main() -> None:
    data = {
        "M_reg_raw": pd.read_csv(RAW / "MRegularSeasonDetailedResults.csv"),
        "W_reg_raw": pd.read_csv(RAW / "WRegularSeasonDetailedResults.csv"),
        "M_teams": pd.read_csv(RAW / "MTeams.csv"), "W_teams": pd.read_csv(RAW / "WTeams.csv"),
    }
    m_t = pd.read_csv(RAW / "MNCAATourneyDetailedResults.csv")
    w_t = pd.read_csv(RAW / "WNCAATourneyDetailedResults.csv")
    # Combined raw log: reg + tournament, so build_reg_games computes end-of-reg-season as-of for
    # the tournament rows (their as-of uses all reg games; round 1 has no prior tournament games).
    combined = dict(data)
    combined["M_reg_raw"] = pd.concat([data["M_reg_raw"], m_t], ignore_index=True)
    combined["W_reg_raw"] = pd.concat([data["W_reg_raw"], w_t], ignore_index=True)

    seasons = sorted(set(combined["M_reg_raw"].Season) | set(combined["W_reg_raw"].Season))
    # The exact adj_eff the model trained on (leak-free KenPom Selection-Sunday overlay post-KP-005);
    # feeds the *_prev priors via the Season−1 join, so it must match training.
    adj_eff = pd.read_parquet(PROC / "adj_eff.parquet")
    kp, tv, dayzero = _asof_inputs(data, seasons)
    games = build_reg_games(combined, adj_eff, asof_snapshots=kp, dayzero_by_season=dayzero,
                            torvik_women_snapshots=tv)

    t = games[games.DayNum >= TOURN_DAYNUM].copy()
    model = pickle.load(open(PROC / "reg_model.pkl", "rb"))
    feats = model.margin_features + model.total_features
    t[feats] = t[feats].fillna(0)
    ps = model.predict_scores(t); wp = model.predict_batch(t)
    t["pred_margin"] = ps["pred_margin"].to_numpy(); t["pred_total"] = ps["pred_total"].to_numpy(); t["wp"] = wp
    t["round1"] = t.DayNum <= 136

    def report(df, label):
        if not len(df):
            return
        mae = np.abs(df.pred_margin - df.Margin).mean()
        tmae = np.abs(df.pred_total - df.Total).mean()
        br = _brier(df.Outcome.to_numpy(), df.wp.to_numpy())
        acc = ((df.wp >= 0.5).astype(int) == df.Outcome).mean()
        print(f"  {label:28} margin_MAE {mae:6.3f}  total_MAE {tmae:6.3f}  Brier {br:.4f}  acc {acc:.3f}  (n={len(df)})")

    print("=== GM-007: reg handicapper on 2003–2025 tournaments (OOS context — see header caveats) ===")
    report(t, "ALL tournament games")
    report(t[t.round1], "round-1 only (fixed as-of)")
    report(t[~t.round1], "later rounds (mid-tourn upd.)")
    print("men:"); report(t[t.men_women == 0], "  men (all neutral)")
    print("women:")
    report(t[(t.men_women == 1) & (t.A_home == 0)], "  women neutral")
    report(t[(t.men_women == 1) & (t.A_home != 0)], "  women home/away")


if __name__ == "__main__":
    main()
