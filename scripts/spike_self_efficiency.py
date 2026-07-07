import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import sys  # noqa: E402

sys.path.insert(0, "src")
from cbb.data import build_symmetric_games  # noqa: E402
from cbb.features import compute_adj_efficiency  # noqa: E402
from cbb.train.model import _brier  # noqa: E402
from cbb.train.reg_model import train_reg_loto  # noqa: E402

RAW, PROC = Path("data/raw"), Path("data/processed")
BASES = ["AdjOE", "AdjDE", "AdjEM", "AdjTempo"]


def main() -> None:
    data = {
        "M_reg_raw": pd.read_csv(RAW / "MRegularSeasonDetailedResults.csv"),
        "W_reg_raw": pd.read_csv(RAW / "WRegularSeasonDetailedResults.csv"),
        "M_tourn_raw": pd.read_csv(RAW / "MNCAATourneyDetailedResults.csv"),
        "W_tourn_raw": pd.read_csv(RAW / "WNCAATourneyDetailedResults.csv"),
    }
    reg_sym, _ = build_symmetric_games(data)

    # 1. as-of adjusted efficiency at weekly cutoffs (games strictly before each cutoff → leak-free)
    print("computing weekly as-of reverse-KenPom snapshots...")
    snaps = []
    for cut in range(14, 133, 7):
        sub = reg_sym[reg_sym.DayNum < cut]
        if not len(sub):
            continue
        adj = compute_adj_efficiency(sub)
        adj["snap"] = cut
        snaps.append(adj[["Season", "men_women", "TeamID", "snap", *BASES]])
    asof = pd.concat(snaps, ignore_index=True)

    # 2. DayNum-based as-of join onto reg_games (latest snapshot strictly before each game)
    g = pd.read_parquet(PROC / "reg_games.parquet").copy()

    def side(gg, s):
        tid = f"{s}_TeamID"
        left = gg[["Season", "men_women", tid, "DayNum"]].reset_index().sort_values("DayNum")
        right = asof.rename(columns={"TeamID": tid}).sort_values("snap")
        m = pd.merge_asof(left, right, left_on="DayNum", right_on="snap",
                          by=["Season", "men_women", tid], direction="backward", allow_exact_matches=False)
        return m.set_index("index")[BASES]

    A, B = side(g, "A"), side(g, "B")
    for b in BASES:
        g[f"d_adjself_{b}_asof"] = A[b] - B[b]
        g[f"s_adjself_{b}_asof"] = A[b] + B[b]

    kp = [c for c in g.columns if "kp_" in c and c.startswith(("d_", "s_"))]
    selfc = [c for c in g.columns if "adjself_" in c and c.startswith(("d_", "s_"))]
    print(f"joined {len(selfc)} self-efficiency features; kp features: {len(kp)}\n")

    def evalm(games, label):
        res = train_reg_loto(games)
        h = games[games.Season == 2026].copy()
        mf, tf = res.model.margin_features, res.model.total_features
        h[mf + tf] = h[mf + tf].fillna(0)
        men = h[h.men_women == 0]
        ps = res.model.predict_scores(men); pb = res.model.predict_batch(men)
        print(f"  {label:26} loto_brier_reg {res.metrics['loto_brier_reg']:.5f} | men-2026 "
              f"margin {np.abs(ps.pred_margin - men.Margin).mean():.3f}  "
              f"total {np.abs(ps.pred_total - men.Total).mean():.3f}  "
              f"Brier {_brier(men.Outcome.to_numpy(), pb):.4f}")

    print("=== independence test (men 2026 is where KenPom matters most) ===")
    evalm(g.drop(columns=selfc), "A: current (KenPom)")
    evalm(g.drop(columns=kp), "B: drop KenPom, use ours")
    evalm(g, "C: both")


if __name__ == "__main__":
    main()
