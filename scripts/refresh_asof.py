"""Stage-3 weekly refresh: recompute self-computed as-of ratings including live results.

Elo + rolling box-score ratings are rebuilt on the fly by ``daily_slate.py`` from the raw log, so they
need no refresh. The ``adjself`` weekly snapshots (opponent-adjusted reverse-KenPom efficiency) are the
one cached model input — this recomputes ``data/processed/adjself_asof.parquet`` from Kaggle history +
the ESPN live logs so the current season's snapshots stay current. Run WEEKLY (daily cadence is a
proven wash, WM/daily-adjself); ``adj_eff`` priors are prev-season and only change at a season boundary.

    python scripts/refresh_asof.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from cbb.data import normalize_games  # noqa: E402
from cbb.features.adjself_asof import compute_adjself_asof_snapshots  # noqa: E402

RAW, PROC, LIVE = Path("data/raw"), Path("data/processed"), Path("data/live")


def main() -> None:
    ms = pd.read_csv(RAW / "MSeasons.csv")
    dz = dict(zip(ms.Season, pd.to_datetime(ms.DayZero)))
    m = pd.read_csv(RAW / "MRegularSeasonDetailedResults.csv")
    w = pd.read_csv(RAW / "WRegularSeasonDetailedResults.csv")
    live = [pd.read_csv(p) for p in sorted(glob.glob(str(LIVE / "mreg_live_*.csv")))]
    if live:
        m = pd.concat([m, *live], ignore_index=True).drop_duplicates(["Season", "DayNum", "WTeamID", "LTeamID"])
    reg_sym = pd.concat([normalize_games(m, men_women=0), normalize_games(w, men_women=1)], ignore_index=True)

    adjself = compute_adjself_asof_snapshots(reg_sym, dz)
    PROC.mkdir(parents=True, exist_ok=True)
    adjself.to_parquet(PROC / "adjself_asof.parquet", index=False)
    ng = sum(len(x) for x in live)
    print(f"refreshed adjself_asof.parquet: {len(adjself):,} snapshot-rows "
          f"(incl. {ng} live men's games across {len(live)} season log(s))")


if __name__ == "__main__":
    main()
