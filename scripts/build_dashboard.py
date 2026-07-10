"""Build the handicapper dashboard HTML from on-disk data (DASH-002 wiring).

Assembles the walk-forward predictions log for a season and renders the self-contained dashboard
(`cbb.dashboard`) to ``monitoring/handicapper.html``. Our side is fully offline: LOTO out-of-fold
predictions for past seasons + the 2026 holdout, each leak-free as-of its game (the walk-forward
discipline that keeps "we beat KenPom by X" fair). Actuals ride on the log, so the FanMatch page
shows us vs actual today; the KenPom column fills in once the FanMatch cache is present (a small
insert at the marked hook), and the ratings page (DASH-003) is a follow-up.

Run from the repo root after the features pipeline has written ``reg_games.parquet``::

    python scripts/build_dashboard.py --season 2026
"""

import argparse
import warnings
from datetime import date
from pathlib import Path

import pandas as pd

from cbb.dashboard import build_payload, render_html
from cbb.dashboard.wiring import (
    add_game_date,
    build_name_map,
    dayzero_by_gender_season,
    dedupe_symmetric,
)
from cbb.train.reg_model import RegConfig, build_reg_predictions_log, train_reg_loto

warnings.filterwarnings("ignore")

DATA = Path("data/processed")
RAW = Path("data/raw")
OUT = Path("monitoring/handicapper.html")
LOG_OUT = DATA / "reg_predictions_log.parquet"


def _read_csv(name: str) -> pd.DataFrame | None:
    path = RAW / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


def build(season: int) -> None:
    reg_games = pd.read_parquet(DATA / "reg_games.parquet")
    print(f"reg_games: {len(reg_games):,} rows, seasons {reg_games['Season'].min()}–{reg_games['Season'].max()}")

    # Walk-forward predictions: LOTO OOF over past seasons + the holdout season, never in-sample.
    result = train_reg_loto(reg_games, RegConfig())
    log = dedupe_symmetric(build_reg_predictions_log(result, reg_games))

    dayzero = dayzero_by_gender_season(_read_csv("MSeasons"), _read_csv("WSeasons"))
    log = add_game_date(log, dayzero)
    log = log.dropna(subset=["game_date"])
    LOG_OUT.parent.mkdir(parents=True, exist_ok=True)
    log.to_parquet(LOG_OUT, index=False)
    print(f"predictions log: {len(log):,} rows → {LOG_OUT}")

    # ── DASH-002 KenPom comparator hook ──────────────────────────────────────────────
    # When data/processed/fanmatch_{season}.parquet is cached, adapt it with
    # cbb.benchmark.slate.fanmatch_to_comparator + match_comparator_to_log here to add the
    # cmp_* columns; until then the slate renders us-vs-actual and the KenPom column is blank.

    name_map = build_name_map(_read_csv("MTeams"), _read_csv("WTeams"))
    slate_log = log[log["Season"] == season]
    if slate_log.empty:
        raise SystemExit(f"no games for season {season} in the predictions log")

    payload = build_payload(
        slate_log, ratings_log=pd.DataFrame(), name_map=name_map, generated=date.today().isoformat()
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render_html(payload))
    print(f"dashboard: {payload['meta']['n_games']:,} games across {len(payload['slate_dates'])} dates → {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the handicapper dashboard HTML")
    ap.add_argument("--season", type=int, default=2026, help="Season to render on the slate (default 2026)")
    build(ap.parse_args().season)


if __name__ == "__main__":
    main()
