"""WM-001: women's model benchmark — the baseline the women's epic (WM-002/003/004) measures against.

Scores the current reg-season champion on the 2026 holdout, broken down by season phase, margin
size, and conference vs non-conference, for women (the focus) with men and naive baselines as
reference. Re-run after each women's story to see where the needle moved.

    python scripts/benchmark_women.py
"""
import pickle
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

import sys  # noqa: E402

sys.path.insert(0, "src")
from cbb.benchmark.women_bench import add_dimensions, holdout_metrics, naive_metrics, report_by  # noqa: E402

PROC, RAW = Path("data/processed"), Path("data/raw")
SEASON = 2026


def _scored(games: pd.DataFrame, model) -> pd.DataFrame:
    feats = model.margin_features + model.total_features
    g = games.copy()
    g[feats] = g[feats].fillna(0)
    ps = model.predict_scores(g)
    g["pred_margin"] = ps["pred_margin"].to_numpy()
    g["pred_total"] = ps["pred_total"].to_numpy()
    g["pred_wp"] = model.predict_batch(g)
    return g


def _fmt(label: str, m: dict) -> str:
    return f"  {label:20} margin_MAE {m['margin_mae']:6.3f}   total_MAE {m['total_mae']:6.3f}   Brier {m['brier']:.4f}   (n={m['n']})"


def main() -> None:
    games = pd.read_parquet(PROC / "reg_games.parquet")
    model = pickle.load(open(PROC / "reg_model.pkl", "rb"))
    conf = pd.read_csv(RAW / "WTeamConferences.csv")
    mconf = pd.read_csv(RAW / "MTeamConferences.csv")
    conf_lookup = {(int(s), int(t)): c for s, t, c in
                   pd.concat([conf, mconf]).itertuples(index=False)}

    hold = games[games.Season == SEASON]
    women = add_dimensions(_scored(hold[hold.men_women == 1], model), conf_lookup)
    men = _scored(hold[hold.men_women == 0], model)

    print(f"=== WM-001 baseline — {SEASON} holdout (cbb-reg-model current champion) ===\n")
    wm = holdout_metrics(women, women.pred_margin, women.pred_total, women.pred_wp)
    wn = naive_metrics(women)
    mm = holdout_metrics(men, men.pred_margin, men.pred_total, men.pred_wp)
    mn = naive_metrics(men)
    print("WOMEN (the focus):")
    print(_fmt("overall", wm))
    print(_fmt("  vs naive floor", wn))
    print("MEN (reference target):")
    print(_fmt("overall", mm))
    print(_fmt("  vs naive floor", mn))
    # Variance-normalized skill (1 − model/naive): women's games are higher-variance, so raw
    # margin_MAE overstates the gap — skill is the honest cross-gender comparison.
    print("\nMargin skill (1 − model_MAE/naive_MAE — higher = better, variance-normalized):")
    print(f"  women {1 - wm['margin_mae']/wn['margin_mae']:.1%}   men {1 - mm['margin_mae']/mn['margin_mae']:.1%}"
          f"   (women floor {wn['margin_mae']:.1f} vs men {mn['margin_mae']:.1f})")

    for dim in ["phase", "margin_bucket", "conf_game"]:
        print(f"\nWomen by {dim}:")
        rep = report_by(women, dim)
        for r in rep.itertuples(index=False):
            print(_fmt(str(getattr(r, dim)), r._asdict()))


if __name__ == "__main__":
    main()
