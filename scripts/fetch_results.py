"""Stage-3 nightly ingest: pull a date's final men's D1 results from ESPN → the live results log.

Kaggle's ``MRegularSeasonDetailedResults`` is a season-end dump; during the live season we need last
night's box scores to keep the as-of ratings current. ESPN's public scoreboard + summary endpoints
give final scores + the full box (FGM/FGA/3P/FT/OR/DR/AST/TO/STL/BLK/PF). We parse to the Kaggle
detailed-results schema, crosswalk ESPN team locations → Kaggle TeamIDs, and append to
``data/live/mreg_live_{season}.csv`` (skip-if-cached, so it's safe to re-run).

    python scripts/fetch_results.py 2026-01-17    # one date
    python scripts/fetch_results.py               # yesterday (the nightly call)

`daily_slate.py` reads Kaggle + this live log as the raw log, so Elo/box-score ratings are recomputed
current on the fly. Scores/FGA match Kaggle exactly; OR/TO differ by ~1–3 (ESPN vs Kaggle counting) —
a minor noise on *current-season* box detail only (history stays exact-Kaggle).
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

RAW, LIVE = Path("data/raw"), Path("data/live")
SB = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?dates={d}&groups=50&limit=400"
SUM = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary?event={e}"
# ESPN location → Kaggle name, for the handful the normalized lookup misses (verified, not fuzzy).
OVERRIDES = {"app state": "Appalachian St", "ul monroe": "ULM", "ut rio grande valley": "UTRGV",
             "san josé state": "San Jose St", "san jose state": "San Jose St", "miami": "Miami FL",
             "saint francis": "St Francis PA", "louisiana": "Louisiana"}
COLS = ["Season", "DayNum", "WTeamID", "WScore", "LTeamID", "LScore", "WLoc", "NumOT",
        "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WOR", "WDR", "WAst", "WTO", "WStl", "WBlk", "WPF",
        "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA", "LOR", "LDR", "LAst", "LTO", "LStl", "LBlk", "LPF"]


def _get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def _crosswalk():
    teams = pd.read_csv(RAW / "MTeams.csv")
    spell = pd.read_csv(RAW / "MTeamSpellings.csv", encoding="latin-1")
    norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())  # noqa: E731
    look: dict[str, int] = {}
    for _, r in teams.iterrows():
        look[norm(r.TeamName)] = int(r.TeamID)
    for _, r in spell.iterrows():
        look.setdefault(norm(r.TeamNameSpelling), int(r.TeamID))
    ov = {norm(k): look.get(norm(v)) for k, v in OVERRIDES.items()}

    def resolve(location):
        k = norm(location)
        return ov.get(k) or look.get(k)
    return resolve


def _box(stats: list[dict]) -> dict:
    """ESPN team boxscore stats → Kaggle box fields (made/att split from 'M-A' strings)."""
    by = {s.get("name"): s.get("displayValue") for s in stats}

    def ma(name):
        v = by.get(name, "")
        p = str(v).split("-")
        return (int(p[0]), int(p[1])) if len(p) == 2 and p[0].lstrip("-").isdigit() else (None, None)

    def v(name):
        x = by.get(name)
        return int(x) if x is not None and str(x).lstrip("-").isdigit() else None
    fgm, fga = ma("fieldGoalsMade-fieldGoalsAttempted")
    fgm3, fga3 = ma("threePointFieldGoalsMade-threePointFieldGoalsAttempted")
    ftm, fta = ma("freeThrowsMade-freeThrowsAttempted")
    return {"FGM": fgm, "FGA": fga, "FGM3": fgm3, "FGA3": fga3, "FTM": ftm, "FTA": fta,
            "OR": v("offensiveRebounds"), "DR": v("defensiveRebounds"), "Ast": v("assists"),
            "TO": v("turnovers"), "Stl": v("steals"), "Blk": v("blocks"), "PF": v("fouls")}


def _game_row(event, resolve, dayzero) -> dict | None:
    comp = event["competitions"][0]
    if comp["status"]["type"]["name"] != "STATUS_FINAL":
        return None
    sides = {c["homeAway"]: c for c in comp["competitors"]}
    h, a = sides.get("home"), sides.get("away")
    if not h or not a:
        return None
    hs, as_ = int(h["score"]), int(a["score"])
    (w, ws), (loser, ls) = ((h, hs), (a, as_)) if hs > as_ else ((a, as_), (h, hs))
    wid, lid = resolve(w["team"].get("location")), resolve(loser["team"].get("location"))
    if wid is None or lid is None or wid == lid:
        return None
    try:
        box = _get(SUM.format(e=event["id"]))["boxscore"]["teams"]
    except Exception:  # noqa: BLE001
        return None
    bx = {t["homeAway"]: _box(t["statistics"]) for t in box}
    wb, lb = bx.get(w["homeAway"], {}), bx.get(loser["homeAway"], {})
    neutral = bool(comp.get("neutralSite"))
    wloc = "N" if neutral else ("H" if w["homeAway"] == "home" else "A")
    period = int(comp["status"].get("period", 2))
    date = pd.to_datetime(event["date"]).tz_convert("US/Eastern").normalize().tz_localize(None)
    row = {"Season": None, "DayNum": (date - dayzero).days if dayzero is not None else None,
           "WTeamID": wid, "WScore": ws, "LTeamID": lid, "LScore": ls, "WLoc": wloc,
           "NumOT": max(0, period - 2)}
    for pre, b in (("W", wb), ("L", lb)):
        for k, kk in [("FGM", "FGM"), ("FGA", "FGA"), ("FGM3", "FGM3"), ("FGA3", "FGA3"), ("FTM", "FTM"),
                      ("FTA", "FTA"), ("OR", "OR"), ("DR", "DR"), ("Ast", "Ast"), ("TO", "TO"),
                      ("Stl", "Stl"), ("Blk", "Blk"), ("PF", "PF")]:
            row[f"{pre}{kk}"] = b.get(k)
    return row


def fetch_date(date: str) -> pd.DataFrame:
    season = pd.to_datetime(date).year + (1 if pd.to_datetime(date).month >= 8 else 0)
    ms = pd.read_csv(RAW / "MSeasons.csv")
    dz = dict(zip(ms.Season, pd.to_datetime(ms.DayZero))).get(season)
    resolve = _crosswalk()
    events = _get(SB.format(d=date.replace("-", "")))["events"]
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(lambda e: _game_row(e, resolve, dz), events))
    df = pd.DataFrame([r for r in rows if r]).assign(Season=season)
    return df[COLS] if len(df) else pd.DataFrame(columns=COLS)


def main() -> None:
    date = sys.argv[1] if len(sys.argv) > 1 else (pd.Timestamp.today() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = fetch_date(date)
    season = pd.to_datetime(date).year + (1 if pd.to_datetime(date).month >= 8 else 0)
    LIVE.mkdir(parents=True, exist_ok=True)
    out = LIVE / f"mreg_live_{season}.csv"
    prior = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=COLS)
    combined = pd.concat([prior, df], ignore_index=True).drop_duplicates(["Season", "DayNum", "WTeamID", "LTeamID"])
    combined.to_csv(out, index=False)
    print(f"{date}: fetched {len(df)} final games → {out} ({len(combined)} total in {season} live log)")


if __name__ == "__main__":
    main()
