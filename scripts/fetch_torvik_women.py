"""Fetch + cache BartTorvik women's as-of ratings (WM-003) — the free women's KenPom-analog.

KenPom is men-only and Kaggle has no women's ratings, so women's within-season opponent-adjusted
signal comes from BartTorvik (barttorvik.com/ncaaw), which publishes free daily as-of snapshots via
its Time-Machine JSON. This fetches **weekly** women's snapshots per season (matching the KenPom
archive cadence) and stacks the opponent-adjusted ratings into
``data/torvik/torvik_women_{season}.parquet`` (columns: ArchiveDate, TeamName, tv_AdjOE, tv_AdjDE,
tv_AdjTempo). Verified schema: JSON array idx 1=team, 4=AdjOE, 6=AdjDE, 44=AdjTempo (corr 0.996–0.998
vs KenPom for men). Coverage: women 2021–2026. Skip-if-cached; DVC-track the output.

    python scripts/fetch_torvik_women.py                # all seasons 2021–2026
    python scripts/fetch_torvik_women.py 2026           # one season
"""

from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

TORVIK_DIR = Path("data/torvik")
FIRST_SEASON = 2021
UA = {"User-Agent": "Mozilla/5.0 AppleWebKit/537.36 Chrome/120 Safari/537.36"}
BASE = "https://barttorvik.com/ncaaw/timemachine/team_results/{date}_team_results.json.gz"
# Verified JSON array positions (barttorvik team_results schema).
IDX = {"team": 1, "tv_AdjOE": 4, "tv_AdjDE": 6, "tv_AdjTempo": 44}
FIRST_DAYNUM, LAST_DAYNUM, STEP = 0, 132, 7  # weekly, DayZero (early Nov) → Selection Sunday


def _snapshot(url_date: str, archive_date: str) -> pd.DataFrame | None:
    # Time-Machine filenames are YYYYMMDD (no dashes); ArchiveDate stored as YYYY-MM-DD.
    r = httpx.get(BASE.format(date=url_date), headers=UA, timeout=30, follow_redirects=True)
    if r.status_code != 200:
        return None
    body = r.content
    try:
        body = gzip.decompress(body)  # raw .gz when the server doesn't set Content-Encoding
    except (OSError, gzip.BadGzipFile):
        pass  # httpx already decoded it (Content-Encoding: gzip) → plain JSON
    rows = json.loads(body)
    if not rows:
        return None
    df = pd.DataFrame({k: [row[i] for row in rows] for k, i in IDX.items()})
    df["ArchiveDate"] = archive_date
    return df


def main() -> None:
    seasons = [int(a) for a in sys.argv[1:]] or list(range(FIRST_SEASON, 2027))
    ws = pd.read_csv("data/raw/WSeasons.csv")
    dayzero = dict(zip(ws["Season"], pd.to_datetime(ws["DayZero"])))
    today = pd.Timestamp.today().normalize()

    TORVIK_DIR.mkdir(parents=True, exist_ok=True)
    for s in seasons:
        if s < FIRST_SEASON:
            print(f"  {s}: before Torvik women coverage ({FIRST_SEASON}) — skipping")
            continue
        out = TORVIK_DIR / f"torvik_women_{s}.parquet"
        existing = pd.read_parquet(out) if out.exists() else None
        have = set(existing["ArchiveDate"].astype(str)) if existing is not None else set()

        new_parts = []
        for d in range(FIRST_DAYNUM, LAST_DAYNUM + 1, STEP):
            dt = dayzero[s] + pd.Timedelta(days=d)
            if dt > today:
                break
            ds = dt.strftime("%Y-%m-%d")
            if ds in have:
                continue
            try:
                snap = _snapshot(dt.strftime("%Y%m%d"), ds)
                if snap is not None:
                    new_parts.append(snap)
                time.sleep(0.3)  # be polite to a free site
            except Exception as exc:  # noqa: BLE001
                print(f"  {s} {ds}: FAILED ({str(exc).splitlines()[0]})")

        if not new_parts:
            print(f"  {s}: up to date ({len(have)} snapshots)" if existing is not None else f"  {s}: nothing fetched")
            continue
        combined = pd.concat(([existing] if existing is not None else []) + new_parts, ignore_index=True)
        combined.to_parquet(out, index=False)
        print(f"  {s}: +{len(new_parts)} snapshots → {len(combined)} rows ({out.name})")


if __name__ == "__main__":
    main()
