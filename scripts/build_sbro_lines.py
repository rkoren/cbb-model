"""Build a tidy men's closing-lines table from the SBRO archive (MK-001).

Source: the Sportsbook Reviews Online NCAAB archive, mirrored on Kaggle as
``ameyerk/ncaab-vegas`` (14 season .xlsx, 2007-08 → 2020-21 = Kaggle Seasons 2008–2021).
The SBRO layout is two rows per game (visitor then home); ``Close`` holds the point
*spread* on the favorite's row and the game *total* on the other row ("NL"=no line,
"pk"=pick'em). We decode favorite by moneyline, orient the spread to the home team,
crosswalk SBRO's concatenated team names to Kaggle TeamIDs, and stamp Season/DayNum.

    python scripts/build_sbro_lines.py   → data/betting/sbro_lines.parquet
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

SBRO_DIR = Path("data/betting/sbro")
RAW = Path("data/raw")
OUT = Path("data/betting/sbro_lines.parquet")


def _num(x):
    s = str(x).strip().lower()
    if s in ("nl", "nan", ""):
        return None
    if s == "pk":
        return 0.0
    try:
        return float(s)
    except ValueError:
        return None


def _crosswalk():
    """SBRO team name → Kaggle TeamID. norm-lookup + strip-'U' + verified overrides."""
    teams = pd.read_csv(RAW / "MTeams.csv")
    spell = pd.read_csv(RAW / "MTeamSpellings.csv", encoding="latin-1")
    norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())  # noqa: E731
    lookup: dict[str, int] = {}
    for _, r in teams.iterrows():
        lookup[norm(r.TeamName)] = int(r.TeamID)
    for _, r in spell.iterrows():
        lookup.setdefault(norm(r.TeamNameSpelling), int(r.TeamID))
    # Verified overrides (fuzzy false-positive-prone → mapped by hand to the correct TeamID).
    ov_id = {"arkansaslr": 1114, "etennessest": 1190, "easttennstate": 1190, "utriograndevalley": 1410,
             "utriograndvalley": 1410, "stephenaustin": 1372, "towsonstate": 1406, "northernarz": 1319,
             "zzzzndakotast": 1295}
    ov_name = {"wiscmilwaukee": "WI Milwaukee", "wiscgreenbay": "WI Green Bay", "soillinois": "S Illinois",
               "noillinois": "N Illinois", "nocolorado": "N Colorado", "collcharleston": "Col Charleston",
               "collofcharleston": "Col Charleston", "geowashington": "G Washington", "ullafayette": "Louisiana",
               "ulmonroe": "ULM", "calirvine": "UC Irvine", "calsantabarb": "UC Santa Barbara",
               "calsantabarbara": "UC Santa Barbara", "texsanantonio": "UT San Antonio", "flagulfcoast": "FL Gulf Coast",
               "charlestonsou": "Charleston So", "fairdickinson": "F Dickinson", "scarupstate": "SC Upstate",
               "socarolinast": "S Carolina St", "somississippi": "Southern Miss", "fullertonst": "CS Fullerton",
               "njtech": "NJIT"}
    for k, v in ov_name.items():
        if norm(v) in lookup:
            ov_id.setdefault(k, lookup[norm(v)])

    def resolve(name):
        k = norm(name)
        if k in ov_id:
            return ov_id[k]
        if k in lookup:
            return lookup[k]
        if k.endswith("u") and k[:-1] in lookup:  # SBRO "U"-suffix schools (WashingtonU → Washington)
            return lookup[k[:-1]]
        return None
    return resolve


def _parse_season(path: str, end_year: int) -> pd.DataFrame:
    df = pd.read_excel(path).reset_index(drop=True)
    rows = []
    for i in range(0, len(df) - 1, 2):
        a, b = df.iloc[i], df.iloc[i + 1]  # a = visitor row, b = home row
        try:
            d = str(int(a["Date"])).zfill(4)
        except (ValueError, TypeError):
            continue
        mm, dd = int(d[:2]), int(d[2:])
        year = end_year - 1 if mm >= 8 else end_year  # Nov–Dec → prior calendar year
        vml, hml = _num(a["ML"]), _num(b["ML"])
        home_fav = (hml is not None and vml is not None and hml < vml) or \
                   (hml is not None and vml is None and hml < 0)

        def _split(col):
            av, bv = _num(a[col]), _num(b[col])
            vals = [v for v in (av, bv) if v is not None]
            total = next((v for v in vals if abs(v) > 80), None)       # totals are 100–200 in CBB
            spr = next((v for v in vals if abs(v) <= 80), None)         # spread magnitude
            return (None if spr is None else (-abs(spr) if home_fav else abs(spr)), total)
        hs_c, tot_c = _split("Close")
        hs_o, tot_o = _split("Open")
        rows.append({"Season": end_year, "year": year, "month": mm, "day": dd,
                     "v_name": str(a["Team"]).strip(), "h_name": str(b["Team"]).strip(),
                     "neutral": int("N" in (str(a["VH"]), str(b["VH"]))),
                     "home_spread_close": hs_c, "total_close": tot_c,
                     "home_spread_open": hs_o, "total_open": tot_o,
                     "v_ml": vml, "h_ml": hml, "v_score": _num(a["Final"]), "h_score": _num(b["Final"])})
    return pd.DataFrame(rows)


def main() -> None:
    resolve = _crosswalk()
    parts = [_parse_season(p, int(p.split("-")[-1].split(".")[0]) + 2000)
             for p in sorted(glob.glob(str(SBRO_DIR / "ncaa-basketball-*.xlsx")))]
    g = pd.concat(parts, ignore_index=True)

    g["visitor_id"] = g.v_name.map(resolve)
    g["home_id"] = g.h_name.map(resolve)
    g["game_date"] = pd.to_datetime(dict(year=g.year, month=g.month, day=g.day), errors="coerce")
    ms = pd.read_csv(RAW / "MSeasons.csv")
    dz = dict(zip(ms.Season, pd.to_datetime(ms.DayZero)))
    g["DayNum"] = [(d - dz[s]).days if s in dz and pd.notna(d) else np.nan
                   for s, d in zip(g.Season, g.game_date)]
    # flag garbled lines (rare SBRO row errors) rather than drop the game
    bad = (g.home_spread_close.abs() > 40) | (g.total_close < 100) | (g.total_close > 250)
    g.loc[bad, ["home_spread_close", "home_spread_open", "total_close", "total_open"]] = np.nan

    d1 = g.dropna(subset=["visitor_id", "home_id"]).copy()
    d1[["visitor_id", "home_id", "DayNum"]] = d1[["visitor_id", "home_id", "DayNum"]].astype("Int64")
    SBRO_DIR.parent.mkdir(parents=True, exist_ok=True)
    d1.drop(columns=["year", "month", "day"]).to_parquet(OUT, index=False)

    rg = pd.read_parquet("data/processed/reg_games.parquet")
    rg = rg[(rg.men_women == 0) & (rg.Season.between(2008, 2021))]
    rg_pairs = set(zip(rg.Season, np.minimum(rg.A_TeamID, rg.B_TeamID), np.maximum(rg.A_TeamID, rg.B_TeamID)))
    d1["pair"] = list(zip(d1.Season, np.minimum(d1.visitor_id, d1.home_id).astype(int),
                          np.maximum(d1.visitor_id, d1.home_id).astype(int)))
    cov = len(rg_pairs & set(d1.pair))
    print(f"parsed {len(g):,} SBRO games → {len(d1):,} with both teams mapped to Kaggle (D1-vs-D1)")
    print(f"name match: {g.home_id.notna().mean():.1%} of home slots; garbled lines flagged: {int(bad.sum())}")
    print(f"our reg_games (men 2008–2021): {len(rg_pairs):,} unique games; "
          f"HAVE a market line: {cov:,} ({cov / len(rg_pairs):.1%})")
    print(f"wrote {OUT}  ({len(d1):,} rows, cols: {list(d1.drop(columns=['pair']).columns)})")


if __name__ == "__main__":
    main()
