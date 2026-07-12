"""Fetch + cache KenPom FanMatch game predictions per season (GM-004).

FanMatch is KenPom's own per-game forecast (predicted scores, win prob, tempo) made *as of the
game date* — the natural opponent for our as-of-date reg-season model (GM-002). This fetches every
game date in a season and stacks the predictions into
``data/kenpom/fanmatch/fanmatch_{season}.parquet`` (one row per game). Predictions only — no actual
results (those come from Kaggle); see [[kenpom-fanmatch-predictions-only]].

Iterates daily across the regular season (~130 calls/season), skips dates already cached, so it's
safe to re-run. DVC-track the output afterward.

    python scripts/fetch_kenpom_fanmatch.py 2026          # one season (the benchmark target)
    python scripts/fetch_kenpom_fanmatch.py               # all seasons 2012..2026
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from cbb.kenpom import KenPomClient  # noqa: E402

FANMATCH_DIR = Path("data/kenpom/fanmatch")
KEEP = ["DateOfGame", "Visitor", "Home", "HomePred", "VisitorPred", "HomeWP", "PredTempo"]
# DayZero (early Nov) → through the NCAA tournament (~DayNum 154). The KenPom `fanmatch` endpoint
# takes a past date, so tournament FanMatch fetches exactly like the regular season — this range
# just extends past Selection Sunday (DayNum ~132) so the dashboard's tournament view has a KenPom
# comparator. Re-run for a season to backfill its bracket; already-cached dates are skipped.
FIRST_DAYNUM, LAST_DAYNUM = 0, 154


def main() -> None:
    seasons = [int(a) for a in sys.argv[1:]] or list(range(2012, 2027))
    ms = pd.read_csv("data/raw/MSeasons.csv")
    dayzero = dict(zip(ms["Season"], pd.to_datetime(ms["DayZero"])))
    today = pd.Timestamp.today().normalize()

    client = KenPomClient()
    FANMATCH_DIR.mkdir(parents=True, exist_ok=True)
    for s in seasons:
        out = FANMATCH_DIR / f"fanmatch_{s}.parquet"
        existing = pd.read_parquet(out) if out.exists() else None
        have = set(existing["DateOfGame"].astype(str)) if existing is not None else set()

        new_parts = []
        for d in range(FIRST_DAYNUM, LAST_DAYNUM + 1):
            dt = dayzero[s] + pd.Timedelta(days=d)
            if dt > today:
                break
            ds = dt.strftime("%Y-%m-%d")
            if ds in have:
                continue
            try:
                fm = client.fanmatch(ds)
                if len(fm):
                    fm = fm[[c for c in KEEP if c in fm.columns]].copy()
                    fm["DateOfGame"] = ds
                    new_parts.append(fm)
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).split("\n")[0]
                if "404" not in msg:  # 404 = no games that date (off-day) — silent
                    print(f"  {s} {ds}: FAILED ({msg})")

        if not new_parts:
            print(f"  {s}: up to date ({len(have)} dates cached)" if existing is not None else f"  {s}: nothing fetched")
            continue
        combined = pd.concat(([existing] if existing is not None else []) + new_parts, ignore_index=True)
        combined.to_parquet(out, index=False)
        print(f"  {s}: +{len(new_parts)} dates → {len(combined)} games ({out.name})")


if __name__ == "__main__":
    main()
