"""Split the dashboard payload into files a static website can fetch one at a time.

``build_payload`` produces one dict covering a whole season -- 2.9 MB for 2026. That is
fine for the self-contained HTML dashboard, which embeds it, but a web page should not
download a season to show one night's games. So the same data is written as:

  public/latest.json                       which seasons exist, and which is newest
  public/season=2026/index.json            the date list + season accuracy table
  public/season=2026/slate/2026-01-17.json one date's games (18 KB median, 78 KB worst)

The per-game shape is exactly ``payload.build_slate``'s, so the page renders whatever the
dashboard renders. Nothing here retrains or refits: it reads the predictions log that
``scripts/build_dashboard.py`` already wrote.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from cbb.dashboard.payload import build_metrics, build_slate

PUBLIC = "public"

# The date list changes whenever a night's games finalize; the per-date files do not
# change once a night is complete.
CACHE_POINTER = "no-cache"
CACHE_INDEX = "public, max-age=300"
CACHE_SLATE = "public, max-age=3600"


def dumps(payload: dict[str, Any]) -> bytes:
    """Compact JSON, refusing NaN so a malformed number fails here and not in a browser."""
    return json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()


def season_files(
    predictions_log: pd.DataFrame,
    name_map: dict[int, str],
    *,
    season: int,
    generated: str,
) -> dict[str, tuple[dict[str, Any], str]]:
    """Return ``{key: (payload, cache_control)}`` for one season. Empty if the season has no games."""
    rows = predictions_log[predictions_log["Season"] == season]
    if rows.empty:
        return {}

    slate = build_slate(rows, name_map)
    metrics = build_metrics(rows)
    dates = sorted(slate)
    base = f"{PUBLIC}/season={season}"

    files: dict[str, tuple[dict[str, Any], str]] = {
        f"{base}/index.json": (
            {
                "generatedAt": generated,
                "season": season,
                "dates": dates,
                "metrics": metrics,
                "meta": {
                    "n_games": int(len(rows)),
                    # Only games with a KenPom line can be scored against one; the rest
                    # still render, they just have no comparison column.
                    "n_compared": int(rows["cmp_margin"].notna().sum())
                    if "cmp_margin" in rows.columns
                    else 0,
                },
            },
            CACHE_INDEX,
        )
    }
    for d in dates:
        files[f"{base}/slate/{d}.json"] = (
            {"generatedAt": generated, "season": season, "date": d, "games": slate[d]},
            CACHE_SLATE,
        )
    return files


def pointer_file(seasons: list[int], generated: str) -> dict[str, tuple[dict[str, Any], str]]:
    """The file a client reads first. Written last, so it never advertises missing seasons."""
    return {
        f"{PUBLIC}/latest.json": (
            {
                "generatedAt": generated,
                "seasons": sorted(seasons),
                "season": max(seasons),
            },
            CACHE_POINTER,
        )
    }
