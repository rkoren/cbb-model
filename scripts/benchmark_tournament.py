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
from cbb.data import normalize_games  # noqa: E402
from cbb.features.adjself_asof import compute_adjself_asof_snapshots  # noqa: E402
from cbb.features.reg_games import build_reg_games  # noqa: E402
from cbb.features.torvik_asof import load_all_torvik_women  # noqa: E402
from cbb.kenpom.features import build_team_name_map  # noqa: E402
from cbb.train.model import _brier  # noqa: E402

RAW, PROC, KP, TV = Path("data/raw"), Path("data/processed"), Path("data/kenpom/archive"), Path("data/torvik")
TOURN_DAYNUM = 134  # tournament games start after Selection Sunday (~DayNum 133)
C_SPREAD = 7.0  # logistic-spread transform scale (the competition's fixed constant)


def _logistic_spread(pred_margin, actual_margin, c: float = C_SPREAD) -> float:
    """Competition logistic-spread Brier: mean (L(pred) − L(actual))², L(x)=1/(1+e^(−x/c)).

    Orientation-invariant (flip A/B → both spreads negate, and L(−x)=1−L(x)), so it's valid on the
    symmetric dataset — the duplicate B-as-A row carries the identical per-game loss.
    """
    def L(x):
        return 1.0 / (1.0 + np.exp(-x / c))
    return float(np.mean((L(np.asarray(pred_margin)) - L(np.asarray(actual_margin))) ** 2))


def _asof_inputs(data, seasons):
    ms = pd.read_csv(RAW / "MSeasons.csv")
    dayzero = dict(zip(ms["Season"], pd.to_datetime(ms["DayZero"])))
    # WM-002: the reg model's within-season strength is self-computed (adjself_*_asof), not KenPom.
    # Compute weekly reverse-KenPom snapshots from the REG-ONLY games (LAST_DAYNUM=133 → Selection-
    # Sunday cutoff, so no tournament game leaks in); fully offline, no KenPom API. Mirrors run.py.
    reg_sym = pd.concat([normalize_games(data["M_reg_raw"], men_women=0),
                         normalize_games(data["W_reg_raw"], men_women=1)], ignore_index=True)
    adjself = compute_adjself_asof_snapshots(reg_sym, dayzero)
    wspell = pd.read_csv(RAW / "WTeamSpellings.csv", encoding="latin-1")
    w_maps = {}
    for s in seasons:
        p = TV / f"torvik_women_{s}.parquet"
        if p.exists():
            names = pd.read_parquet(p)[["team"]].drop_duplicates().rename(columns={"team": "TeamName"})
            w_maps[s] = build_team_name_map(data["W_teams"], names, wspell)
    tv = load_all_torvik_women(seasons, TV, w_maps)
    return adjself, tv, dayzero


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
    adjself, tv, dayzero = _asof_inputs(data, seasons)
    games = build_reg_games(combined, adj_eff, asof_snapshots=None, dayzero_by_season=dayzero,
                            torvik_women_snapshots=tv, adjself_snapshots=adjself)

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
        lsb = _logistic_spread(df.pred_margin, df.Margin)  # the competition's spread metric
        acc = ((df.wp >= 0.5).astype(int) == df.Outcome).mean()
        print(f"  {label:28} margin_MAE {mae:6.3f}  total_MAE {tmae:6.3f}  Brier {br:.4f}  "
              f"logistic {lsb:.4f}  acc {acc:.3f}  (n={len(df)})")

    print("=== GM-007: reg handicapper on 2003–2025 tournaments (OOS context — see header caveats) ===")
    report(t, "ALL tournament games")
    report(t[t.round1], "round-1 only (fixed as-of)")
    report(t[~t.round1], "later rounds (mid-tourn upd.)")
    print("men:"); report(t[t.men_women == 0], "  men (all neutral)")
    print("women:")
    report(t[(t.men_women == 1) & (t.A_home == 0)], "  women neutral")
    report(t[(t.men_women == 1) & (t.A_home != 0)], "  women home/away")

    # ── Phase-1 diagnostic: single-parameter spread scaling to the c=7 transform ──────────────────
    # Is the model globally MIS-SCALED to the logistic (spreads too wide/narrow), or already well
    # scaled but locally imprecise? Fit one α minimizing mean (L(α·s) − L(z))² and see (a) how far α
    # is from 1 and (b) how much the metric moves. α≈1 & Δ<0.001 → well-scaled → a custom objective is
    # real surgery; α off 1 & Δ≥0.002 → a free calibration win first. One param over ~2.4k OOS-for-
    # tournament predictions → negligible overfit (fit == eval is fine for a diagnostic).
    def _best_alpha(df):
        s, z = df.pred_margin.to_numpy(), df.Margin.to_numpy()
        grid = np.linspace(0.50, 1.80, 261)  # 0.005 steps
        losses = [_logistic_spread(a * s, z) for a in grid]
        i = int(np.argmin(losses))
        return grid[i], _logistic_spread(s, z), losses[i]

    print("\n=== Phase-1 diagnostic: optimal 1-param spread scaling α (fit to the c=7 metric) ===")
    for lbl, df in [("all", t), ("men", t[t.men_women == 0]), ("women", t[t.men_women == 1])]:
        a, base, scaled = _best_alpha(df)
        print(f"  {lbl:6} α*={a:.3f}  logistic {base:.4f} → {scaled:.4f}  (Δ {scaled - base:+.4f})")


if __name__ == "__main__":
    main()
