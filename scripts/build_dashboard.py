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

import numpy as np
import pandas as pd
import xgboost as xgb

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
from cbb.features.reg_games import build_reg_games
from cbb.features.torvik_asof import load_all_torvik_women
from cbb.kenpom.asof_features import load_all_archive_snapshots
from cbb.kenpom.features import build_team_name_map
from cbb.train.reg_model import RegConfig, build_reg_predictions_log, train_reg_loto

warnings.filterwarnings("ignore")

# Readable labels for the reg model's margin features (DASH-006 game drivers).
FEATURE_LABELS = {
    "A_home": "Home court", "d_Elo_pre": "Elo edge",
    "d_AdjEM_prev": "AdjEM (prev yr)", "d_AdjOE_prev": "Offense (prev yr)",
    "d_AdjDE_prev": "Defense (prev yr)", "d_AdjTempo_prev": "Tempo (prev yr)",
    "d_bs_NetEff_asof": "Net eff (as-of)", "d_bs_OE_asof": "Offense (as-of)",
    "d_bs_DE_asof": "Defense (as-of)", "d_bs_Tempo_asof": "Tempo (as-of)",
    "d_blend_NetEff_asof": "Net eff (as-of, blended)",
    "d_kp_AdjEM_asof": "KenPom AdjEM (as-of)", "d_kp_AdjOE_asof": "KenPom off (as-of)",
    "d_kp_AdjDE_asof": "KenPom def (as-of)", "d_kp_AdjTempo_asof": "KenPom tempo (as-of)",
    "d_tv_AdjEM_asof": "Torvik AdjEM (as-of)", "d_tv_AdjOE_asof": "Torvik off (as-of)",
    "d_tv_AdjDE_asof": "Torvik def (as-of)", "d_tv_AdjTempo_asof": "Torvik tempo (as-of)",
}

DATA = Path("data/processed")
RAW = Path("data/raw")
KP = Path("data/kenpom")
TOURN_DAYNUM = 134  # tournament games start after Selection Sunday (~DayNum 133)
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


def _driver_list(frame: pd.DataFrame, model, top: int = 5) -> list:
    """Top XGBoost margin contributions per row (SHAP-style pred_contribs), aligned to ``frame``.

    + leans A, − leans B. The frame is scored by the same production model, so the drivers explain
    exactly the predictions shown. Returns one ``[{l, v}, ...]`` list per row.
    """
    feats = model.margin_features
    contribs = model.margin_booster.predict(xgb.DMatrix(frame[feats].fillna(0).to_numpy()), pred_contribs=True)
    labels = [FEATURE_LABELS.get(f, f) for f in feats]
    out = []
    for c in contribs:
        cc = c[:-1]   # drop the bias term
        idx = np.argsort(-np.abs(cc))[:top]
        out.append([{"l": labels[j], "v": round(float(cc[j]), 1)} for j in idx if abs(cc[j]) >= 0.1])
    return out


def _feature_drivers(reg_games: pd.DataFrame, model, season: int, top: int = 5) -> dict:
    """DASH-006 drivers keyed by game identity (for the reg-season slate, whose row order differs)."""
    g = dedupe_symmetric(reg_games[reg_games["Season"] == season]).reset_index(drop=True)
    dl = _driver_list(g, model, top)
    return {(int(r.Season), int(r.DayNum), int(r.A_TeamID), int(r.B_TeamID)): dl[i]
            for i, r in enumerate(g.itertuples(index=False))}


def _build_tournament_games(season: int) -> pd.DataFrame:
    """Season's tournament games with Selection-Sunday-as-of features + actuals (offline, no API).

    Mirrors scripts/benchmark_tournament.py (GM-007): append the tournament results to the raw reg
    log, rebuild reg-games, and take rows DayNum ≥ 134 — their as-of features are then computed on
    the full regular season (no tournament results leak into the ratings), and each row carries the
    real Margin/Total/Outcome. Team→ID maps are built from the cached archive/Torvik names (no API).
    """
    def _rd(name, enc=None):
        return pd.read_csv(RAW / f"{name}.csv", encoding=enc)

    data = {"M_reg_raw": _rd("MRegularSeasonDetailedResults"), "W_reg_raw": _rd("WRegularSeasonDetailedResults"),
            "M_teams": _rd("MTeams"), "W_teams": _rd("WTeams")}
    combined = dict(data)
    combined["M_reg_raw"] = pd.concat([data["M_reg_raw"], _rd("MNCAATourneyDetailedResults")], ignore_index=True)
    combined["W_reg_raw"] = pd.concat([data["W_reg_raw"], _rd("WNCAATourneyDetailedResults")], ignore_index=True)
    seasons = sorted(set(combined["M_reg_raw"].Season) | set(combined["W_reg_raw"].Season))

    mspell, wspell = _rd("MTeamSpellings", "latin-1"), _rd("WTeamSpellings", "latin-1")
    m_maps = {s: build_team_name_map(
        data["M_teams"], pd.read_parquet(KP / "archive" / f"kenpom_archive_{s}.parquet")[["TeamName"]].drop_duplicates(), mspell)
        for s in seasons if (KP / "archive" / f"kenpom_archive_{s}.parquet").exists()}
    w_maps = {s: build_team_name_map(
        data["W_teams"], pd.read_parquet(Path("data/torvik") / f"torvik_women_{s}.parquet").rename(
            columns={"team": "TeamName"})[["TeamName"]].drop_duplicates(), wspell)
        for s in seasons if (Path("data/torvik") / f"torvik_women_{s}.parquet").exists()}
    dayzero_s = dict(zip(_rd("MSeasons").Season, pd.to_datetime(_rd("MSeasons").DayZero)))

    kp = load_all_archive_snapshots(seasons, KP / "archive", m_maps)
    tv = load_all_torvik_women(seasons, Path("data/torvik"), w_maps)
    games = build_reg_games(combined, pd.read_parquet(DATA / "adj_eff.parquet"),
                            asof_snapshots=kp, dayzero_by_season=dayzero_s, torvik_women_snapshots=tv)
    return dedupe_symmetric(games[(games["DayNum"] >= TOURN_DAYNUM) & (games["Season"] == season)]).reset_index(drop=True)


def build_tournament(season: int) -> None:
    """Render a season's NCAA-tournament bracket to monitoring/handicapper_<season>_tourney.html."""
    model = train_reg_loto(pd.read_parquet(DATA / "reg_games.parquet"), RegConfig()).model
    t = _build_tournament_games(season)
    if t.empty:
        raise SystemExit(f"no tournament games for {season} — need {season} rows in "
                         "M/W NCAATourneyDetailedResults (2026 results aren't in Kaggle yet)")
    feats = model.margin_features + model.total_features
    t[feats] = t[feats].fillna(0)
    ps = model.predict_scores(t)
    slate = pd.DataFrame({
        "Season": t["Season"].to_numpy(), "DayNum": t["DayNum"].to_numpy(), "men_women": t["men_women"].to_numpy(),
        "A_TeamID": t["A_TeamID"].to_numpy(), "B_TeamID": t["B_TeamID"].to_numpy(),
        "pred_margin": ps["pred_margin"].to_numpy(), "pred_total": ps["pred_total"].to_numpy(),
        "pred_prob": model.predict_batch(t),
        "Margin": t["Margin"].to_numpy(), "Total": t["Total"].to_numpy(), "Outcome": t["Outcome"].to_numpy(),
    })
    dayzero = dayzero_by_gender_season(_read_csv("MSeasons"), _read_csv("WSeasons"))
    slate = add_game_date(slate, dayzero)

    comparator = _kenpom_comparator(season, dayzero)   # only matches if the FanMatch cache covers tourney dates
    if comparator is not None and attach_kenpom_slate(slate, comparator)["cmp_margin"].notna().any():
        slate = attach_kenpom_slate(slate, comparator)
        print(f"KenPom comparator: {int(slate['cmp_margin'].notna().sum()):,} tournament games matched")
    else:
        print("KenPom comparator: FanMatch cache doesn't cover the tournament — us vs actual only "
              "(extend fetch_kenpom_fanmatch.py past Selection Sunday to add it)")
    slate = add_conf_game(slate, _read_csv("MTeamConferences"), _read_csv("WTeamConferences"))
    slate["drivers"] = _driver_list(t, model)
    print(f"tournament games: {len(slate)} ({int((slate['men_women'] == 0).sum())} men "
          f"+ {int((slate['men_women'] == 1).sum())} women)")

    ours = _our_ratings(season, dayzero[(season, 0)])
    ratings_log = build_gendered_ratings_log(ours, _ratings_comparator(season))
    name_map = build_name_map(_read_csv("MTeams"), _read_csv("WTeams"))
    payload = build_payload(slate, ratings_log, name_map, generated=date.today().isoformat())
    out = Path(f"monitoring/handicapper_{season}_tourney.html")
    out.write_text(render_html(payload))
    print(f"dashboard: {payload['meta']['n_games']} tournament games, "
          f"{payload['meta']['n_snapshots']} rating snapshots → {out}")


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
    slate_log = log[log["Season"] == season].copy()
    if slate_log.empty:
        raise SystemExit(f"no games for season {season} in the predictions log")

    # Per-game feature drivers (DASH-006) — top XGBoost margin contributions, attached by identity.
    drivers = _feature_drivers(reg_games, result.model, season)
    slate_log["drivers"] = [
        drivers.get((int(r.Season), int(r.DayNum), int(r.A_TeamID), int(r.B_TeamID)), [])
        for r in slate_log.itertuples(index=False)]
    print(f"feature drivers: computed for {sum(bool(d) for d in slate_log['drivers']):,} games")

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
    ap.add_argument("--tournament", action="store_true",
                    help="Render the season's NCAA tournament bracket instead of the regular season")
    args = ap.parse_args()
    if args.tournament:
        build_tournament(args.season)
    else:
        build(args.season)


if __name__ == "__main__":
    main()
