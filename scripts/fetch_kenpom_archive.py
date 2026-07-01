"""Fetch + cache as-of-date KenPom archive snapshots per season (GM-002).

For each season (archive coverage starts 2012, men's only) this fetches **weekly** snapshots from
the ``archive`` endpoint — each a point-in-time ratings table — and stacks them into
``data/kenpom/archive/kenpom_archive_{season}.parquet`` (one ``ArchiveDate`` per snapshot). Only
the leak-free as-of columns are kept (``AdjEM/AdjOE/AdjDE/AdjTempo`` + ``TeamName``/``ArchiveDate``);
the ``*Final`` / ``Seed`` / ``Event`` columns are dropped because they leak end-of-season info.

The reg-season features pipeline picks these up automatically (→ ``*_kp_*_asof`` features). This
makes ~20 API calls per season; it skips snapshot dates already cached, so it's safe to re-run.
DVC-track the outputs afterward so CI doesn't re-fetch.

    python scripts/fetch_kenpom_archive.py                 # all seasons 2012..latest
    python scripts/fetch_kenpom_archive.py 2025 2026       # just these seasons (validation slice)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from cbb.kenpom import KenPomClient  # noqa: E402
from cbb.kenpom.asof_features import ARCHIVE_FIRST_SEASON, ASOF_COLS  # noqa: E402

ARCHIVE_DIR = Path("data/kenpom/archive")
KEEP = ["ArchiveDate", "TeamName", *ASOF_COLS]   # non-Final only — the rest leak

# Weekly snapshots across the regular season: DayZero is early November, Selection Sunday ~DayNum 133.
FIRST_DAYNUM, LAST_DAYNUM, STEP = 14, 133, 7


def _snapshot_dates(season: int, dayzero: pd.Timestamp) -> list[str]:
    today = pd.Timestamp.today().normalize()
    dates = []
    for d in range(FIRST_DAYNUM, LAST_DAYNUM + 1, STEP):
        dt = dayzero + pd.Timedelta(days=d)
        if dt <= today:                       # can't fetch future snapshots
            dates.append(dt.strftime("%Y-%m-%d"))
    return dates


def main() -> None:
    seasons = [int(a) for a in sys.argv[1:]] or list(range(ARCHIVE_FIRST_SEASON, 2027))
    mseasons = pd.read_csv("data/raw/MSeasons.csv")
    dayzero = dict(zip(mseasons["Season"], pd.to_datetime(mseasons["DayZero"])))

    client = KenPomClient()
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    for s in seasons:
        if s < ARCHIVE_FIRST_SEASON:
            print(f"  {s}: before archive coverage ({ARCHIVE_FIRST_SEASON}) — skipping")
            continue
        out = ARCHIVE_DIR / f"kenpom_archive_{s}.parquet"
        existing = pd.read_parquet(out) if out.exists() else None
        have = set(existing["ArchiveDate"].astype(str)) if existing is not None else set()

        new_parts = []
        # GM-005: the preseason snapshot — KenPom's roster-aware "game 0" projection — stamped just
        # before DayZero so the as-of merge seeds it for pre-first-weekly-snapshot early-season games.
        preseason_date = (dayzero[s] - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        for d in [preseason_date, *_snapshot_dates(s, dayzero[s])]:
            if d in have:
                continue
            try:
                if d == preseason_date:
                    snap = client.ratings_archive(preseason=True, year=s)
                else:
                    snap = client.ratings_archive(date=d)
                snap = snap[[c for c in KEEP if c in snap.columns]].copy()
                snap["ArchiveDate"] = d
                new_parts.append(snap)
            except Exception as exc:  # noqa: BLE001
                print(f"  {s} {d}: FAILED ({str(exc).splitlines()[0]})")

        if not new_parts and existing is not None:
            print(f"  {s}: up to date ({len(have)} snapshots)")
            continue
        if not new_parts:
            print(f"  {s}: no snapshots fetched")
            continue
        combined = pd.concat(([existing] if existing is not None else []) + new_parts, ignore_index=True)
        combined.to_parquet(out, index=False)
        print(f"  {s}: +{len(new_parts)} snapshots → {len(combined)} rows ({out.name})")


if __name__ == "__main__":
    main()
