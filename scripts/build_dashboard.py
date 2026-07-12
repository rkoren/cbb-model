"""Build the handicapper dashboard HTML from on-disk data (DASH-002 wiring).

Assembles the walk-forward predictions log for a season and renders the self-contained dashboard
(`cbb.dashboard`) to ``monitoring/handicapper.html``. Our side is fully offline: LOTO out-of-fold
predictions for past seasons + the 2026 holdout, each leak-free as-of its game (the walk-forward
discipline that keeps "we beat KenPom by X" fair). Both pages render from cached data with no live
API call: the FanMatch page (DASH-002) shows us vs KenPom vs actual, and the ratings page
(DASH-003) shows our as-of AdjEM/OE/DE/Tempo vs KenPom's (men) / Torvik's (women) with rank deltas.

Run from the repo root after the features pipeline has written ``reg_games.parquet``::

    python scripts/build_dashboard.py --season 2026
"""

import argparse
import warnings
from datetime import date
from pathlib import Path

import pandas as pd

from cbb.benchmark.ratings_log import adjself_to_ours
from cbb.benchmark.slate import fanmatch_to_comparator
from cbb.data import normalize_games
from cbb.dashboard import build_payload, render_html
from cbb.dashboard.wiring import (
    add_conf_game,
    add_game_date,
    attach_kenpom_slate,
    build_gendered_ratings_log,
    build_name_map,
    dayzero_by_gender_season,
    dedupe_symmetric,
    ratings_frame_to_comparator,
)
from cbb.features.adjself_asof import compute_adjself_asof_snapshots
from cbb.kenpom.features import build_team_name_map
from cbb.train.reg_model import RegConfig, build_reg_predictions_log, train_reg_loto

warnings.filterwarnings("ignore")

DATA = Path("data/processed")
RAW = Path("data/raw")
KP = Path("data/kenpom")
OUT = Path("monitoring/handicapper.html")
LOG_OUT = DATA / "reg_predictions_log.parquet"


def _read_csv(name: str) -> pd.DataFrame | None:
    path = RAW / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


def _kenpom_comparator(season: int, dayzero: dict) -> pd.DataFrame | None:
    """KenPom FanMatch predictions as a comparator (men only), from the cached parquets.

    The archive frame supplies the KenPom team names for the KenPom→Kaggle-TeamID map, so no live
    API call is needed. Returns ``None`` when the FanMatch or archive cache for the season is absent.
    """
    fm_path = KP / "fanmatch" / f"fanmatch_{season}.parquet"
    arch_path = KP / "archive" / f"kenpom_archive_{season}.parquet"
    if not (fm_path.exists() and arch_path.exists()):
        return None
    kp_teams = pd.read_parquet(arch_path)[["TeamName"]].drop_duplicates()
    tmap = build_team_name_map(_read_csv("MTeams"), kp_teams, _read_csv("MTeamSpellings"))
    return fanmatch_to_comparator(pd.read_parquet(fm_path), tmap, dayzero[(season, 0)], season)


def _our_ratings(season: int, dayzero: pd.Timestamp) -> pd.DataFrame:
    """Our self-computed as-of ratings snapshots for the season (weekly reverse-KenPom)."""
    reg_sym = pd.concat([
        normalize_games(_read_csv("MRegularSeasonDetailedResults"), men_women=0),
        normalize_games(_read_csv("WRegularSeasonDetailedResults"), men_women=1),
    ], ignore_index=True)
    reg_sym = reg_sym[reg_sym["Season"] == season]
    snaps = compute_adjself_asof_snapshots(reg_sym, {season: dayzero})
    return adjself_to_ours(snaps)


def _ratings_comparator(season: int) -> pd.DataFrame:
    """KenPom archive (men) + Torvik (women) as-of ratings, mapped to Kaggle TeamIDs.

    Empty frame when neither cache exists; each source contributes only if present. Torvik has no
    efficiency-margin column, so ``cmp_AdjEM`` is derived as OE − DE.
    """
    parts = []
    arch_path = KP / "archive" / f"kenpom_archive_{season}.parquet"
    if arch_path.exists():
        arch = pd.read_parquet(arch_path)
        nm = build_team_name_map(_read_csv("MTeams"), arch[["TeamName"]].drop_duplicates(),
                                 _read_csv("MTeamSpellings"))
        parts.append(ratings_frame_to_comparator(
            arch, nm, {"AdjEM": "AdjEM", "AdjOE": "AdjOE", "AdjDE": "AdjDE", "AdjTempo": "AdjTempo"}, season))
    tv_path = Path("data/torvik") / f"torvik_women_{season}.parquet"
    if tv_path.exists():
        tv = pd.read_parquet(tv_path).rename(columns={"team": "TeamName"})
        nm = build_team_name_map(_read_csv("WTeams"), tv[["TeamName"]].drop_duplicates(),
                                 _read_csv("WTeamSpellings"))
        parts.append(ratings_frame_to_comparator(
            tv, nm, {"tv_AdjOE": "AdjOE", "tv_AdjDE": "AdjDE", "tv_AdjTempo": "AdjTempo"},
            season, derive_em=True))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build(season: int) -> None:
    reg_games = pd.read_parquet(DATA / "reg_games.parquet")
    print(f"reg_games: {len(reg_games):,} rows, seasons {reg_games['Season'].min()}–{reg_games['Season'].max()}")

    # Walk-forward predictions: LOTO OOF over past seasons + the holdout season, never in-sample.
    result = train_reg_loto(reg_games, RegConfig())
    log = dedupe_symmetric(build_reg_predictions_log(result, reg_games))

    dayzero = dayzero_by_gender_season(_read_csv("MSeasons"), _read_csv("WSeasons"))
    log = add_game_date(log, dayzero)
    log = log.dropna(subset=["game_date"])

    # KenPom FanMatch comparator (men-only where cached) → us vs KenPom vs actual on the slate.
    comparator = _kenpom_comparator(season, dayzero)
    if comparator is not None:
        log = attach_kenpom_slate(log, comparator)
        print(f"KenPom comparator: {int(log['cmp_margin'].notna().sum()):,} games matched")
    else:
        print("KenPom comparator: no FanMatch cache — slate renders us vs actual only")

    # In/non-conference flag for the metrics segments (DASH-005).
    log = add_conf_game(log, _read_csv("MTeamConferences"), _read_csv("WTeamConferences"))

    LOG_OUT.parent.mkdir(parents=True, exist_ok=True)
    log.to_parquet(LOG_OUT, index=False)
    print(f"predictions log: {len(log):,} rows → {LOG_OUT}")

    # Ratings page (DASH-003): our as-of ratings vs KenPom (men) / Torvik (women), per snapshot.
    ours = _our_ratings(season, dayzero[(season, 0)])
    ratings_log = build_gendered_ratings_log(ours, _ratings_comparator(season))
    print(f"ratings log: {len(ratings_log):,} team-snapshots across {ratings_log['ArchiveDate'].nunique()} dates")

    name_map = build_name_map(_read_csv("MTeams"), _read_csv("WTeams"))
    slate_log = log[log["Season"] == season]
    if slate_log.empty:
        raise SystemExit(f"no games for season {season} in the predictions log")

    payload = build_payload(
        slate_log, ratings_log=ratings_log, name_map=name_map, generated=date.today().isoformat()
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render_html(payload))
    print(f"dashboard: {payload['meta']['n_games']:,} games, "
          f"{payload['meta']['n_snapshots']} rating snapshots → {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the handicapper dashboard HTML")
    ap.add_argument("--season", type=int, default=2026, help="Season to render on the slate (default 2026)")
    build(ap.parse_args().season)


if __name__ == "__main__":
    main()
