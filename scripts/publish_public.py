#!/usr/bin/env python
"""Publish the browsable season archive the website reads.

    python scripts/publish_public.py                      # season 2026 -> S3
    python scripts/publish_public.py --season 2026 2025    # several seasons
    python scripts/publish_public.py --out /tmp/pub        # dry run to a directory

Reads data/processed/reg_predictions_log.parquet, which build_dashboard.py writes. It
does not train, refit or fetch anything, so it is safe to run any time -- including in
the off-season, which is exactly when the archive is the only thing worth showing.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb.dashboard.publish import dumps, pointer_file, season_files  # noqa: E402
from cbb.dashboard.wiring import build_name_map  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LOG = DATA / "processed" / "reg_predictions_log.parquet"
BUCKET = "reilly-cbb-model-data"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, nargs="+", default=[2026], help="Seasons to publish (default 2026)")
    ap.add_argument("--out", type=Path, help="Write to this directory instead of S3")
    ap.add_argument("--bucket", default=BUCKET, help=f"S3 bucket (default {BUCKET})")
    args = ap.parse_args()

    if not LOG.exists():
        raise SystemExit(
            f"missing {LOG}\n"
            "It is DVC-tracked, not in git. Fetch it with:  dvc pull data/processed\n"
            "Or rebuild from scratch with:  python scripts/build_dashboard.py"
        )

    log = pd.read_parquet(LOG)
    name_map = build_name_map(pd.read_csv(DATA / "raw/MTeams.csv"), pd.read_csv(DATA / "raw/WTeams.csv"))
    generated = date.today().isoformat()

    files: dict = {}
    published: list[int] = []
    for season in args.season:
        season_map = season_files(log, name_map, season=season, generated=generated)
        if not season_map:
            print(f"season {season}: no games in the predictions log, skipping")
            continue
        files.update(season_map)
        published.append(season)
        n_dates = sum(1 for k in season_map if "/slate/" in k)
        print(f"season {season}: {n_dates} dates")

    if not published:
        raise SystemExit("nothing to publish")
    # Written last: the pointer must never advertise a season whose files are missing.
    files.update(pointer_file(published, generated))

    if args.out:
        for key, (payload, _) in files.items():
            path = args.out / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(dumps(payload))
        print(f"wrote {len(files)} files -> {args.out}")
        return

    import boto3

    s3 = boto3.client("s3")
    for i, (key, (payload, cache)) in enumerate(files.items(), 1):
        s3.put_object(
            Bucket=args.bucket, Key=key, Body=dumps(payload),
            ContentType="application/json", CacheControl=cache,
        )
        if i % 25 == 0 or i == len(files):
            print(f"  uploaded {i}/{len(files)}")
    print(f"published {len(files)} files -> s3://{args.bucket}/public/ (seasons {published})")


if __name__ == "__main__":
    main()
