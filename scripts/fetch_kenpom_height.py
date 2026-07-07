"""Fetch + cache KenPom height data per season (KP-002).

Writes data/processed/kenpom_height_{season}.parquet for each season that has cached
kenpom_ratings (the features pipeline's `_apply_kenpom` picks them up automatically → the
leak-free kp_AvgHgt/HgtEff features; the endpoint's minutes-weighted Exp/Bench/Continuity
are cached but left unmapped per KP-006). Best-effort: skips a season whose fetch fails or
is already cached. DVC-track the outputs afterward so CI doesn't re-fetch.

    python scripts/fetch_kenpom_height.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

sys.path.insert(0, "src")
from cbb.kenpom import KenPomClient

PROC = Path("data/processed")
KENPOM = Path("data/kenpom")


def main() -> None:
    seasons = sorted(
        int(Path(p).stem.split("_")[-1]) for p in glob.glob(str(KENPOM / "kenpom_ratings_*.parquet"))
    )
    print(f"Seasons with cached ratings: {seasons}")
    client = KenPomClient()
    fetched = skipped = failed = 0
    for s in seasons:
        KENPOM.mkdir(parents=True, exist_ok=True)
        out = KENPOM / f"kenpom_height_{s}.parquet"
        if out.exists():
            print(f"  {s}: already cached")
            skipped += 1
            continue
        try:
            df = client.height(year=s)
            df.to_parquet(out, index=False)
            print(f"  {s}: {len(df)} teams → {out.name}")
            fetched += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  {s}: FAILED ({exc})")
            failed += 1
    print(f"\nfetched={fetched} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
