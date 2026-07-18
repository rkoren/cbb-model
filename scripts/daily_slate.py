"""Stage-1 handicapper: a day's men's slate — us (v11 reg model) vs KenPom FanMatch (M4).

For a date, pull FanMatch (the day's scheduled games + KenPom's predicted scores/win%), score each
game with our reg model, and render a us-vs-FanMatch board sorted by where we most disagree with the
market. Our prediction uses the same result-less "upcoming game" construction as the tournament build:
append the day's matchups to the season log with no result, rebuild reg-games (as-of features come only
from games played *before* the date — leak-free), and predict.

    python scripts/daily_slate.py                # today
    python scripts/daily_slate.py 2026-01-17      # a specific date (historical replay)

Reads cached artifacts (data/processed/{reg_model.pkl, adjself_asof.parquet, adj_eff.parquet}). During
the live season these need a nightly refresh — recompute as-of ratings from the previous night's box
scores — for predictions to reflect the latest games (the Stage-3 job). Offseason/replay uses the cache.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from cbb.features.reg_games import build_reg_games  # noqa: E402
from cbb.kenpom.features import build_team_name_map  # noqa: E402

RAW, PROC, KP = Path("data/raw"), Path("data/processed"), Path("data/kenpom")


def _rd(name, enc=None):
    return pd.read_csv(RAW / f"{name}.csv", encoding=enc)


def _prob_to_ml(p: float) -> int:
    """Win probability → American moneyline (favorite negative, dog positive)."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def handicap_slate(date: str, season: int) -> pd.DataFrame:
    """Score a date's FanMatch men's slate with the reg model → us-vs-FanMatch board."""
    fm = pd.read_parquet(KP / "fanmatch" / f"fanmatch_{season}.parquet")
    fm = fm[fm.DateOfGame == date].copy()
    if fm.empty:
        raise SystemExit(f"no FanMatch games cached for {date} (season {season})")
    arch = pd.read_parquet(KP / "archive" / f"kenpom_archive_{season}.parquet")[["TeamName"]].drop_duplicates()
    tmap = build_team_name_map(_rd("MTeams"), arch, _rd("MTeamSpellings", "latin-1"))
    fm["home_id"], fm["vis_id"] = fm.Home.map(tmap.get), fm.Visitor.map(tmap.get)
    dropped = fm.home_id.isna().sum() + fm.vis_id.isna().sum()
    fm = fm.dropna(subset=["home_id", "vis_id"]).astype({"home_id": int, "vis_id": int})

    ms = _rd("MSeasons")
    dz = dict(zip(ms.Season, pd.to_datetime(ms.DayZero)))
    daynum = (pd.to_datetime(date) - dz[season]).days

    # "Morning of D": the season log as it would look before today's games — drop this season's
    # games from D onward, append the day's matchups with no result. As-of features then use only
    # games played before D (leak-free), and there's no collision with a real same-day game on replay.
    data = {"M_reg_raw": _rd("MRegularSeasonDetailedResults"), "W_reg_raw": _rd("WRegularSeasonDetailedResults"),
            "M_teams": _rd("MTeams"), "W_teams": _rd("WTeams")}
    # Stage-3: fold in the live results log (ESPN-sourced via fetch_results.py) so Elo + box-score
    # ratings recompute current for the in-season game (Kaggle CSVs lag a year). No-op offseason.
    live_path = Path("data/live") / f"mreg_live_{season}.csv"
    mreg = data["M_reg_raw"]
    if live_path.exists():  # keep='first' → Kaggle's exact row wins any overlap; live only adds new games
        mreg = pd.concat([mreg, pd.read_csv(live_path)], ignore_index=True).drop_duplicates(
            ["Season", "DayNum", "WTeamID", "LTeamID"], keep="first")
    mreg = mreg[~((mreg.Season == season) & (mreg.DayNum >= daynum))]
    syn = pd.DataFrame({"Season": season, "DayNum": daynum, "WTeamID": fm.home_id, "LTeamID": fm.vis_id,
                        "WScore": 100, "LScore": 99, "WLoc": "H", "NumOT": 0})  # scores are placeholders (as-of ignores them)
    combined = dict(data)
    combined["M_reg_raw"] = pd.concat([mreg, syn], ignore_index=True)

    games = build_reg_games(combined, pd.read_parquet(PROC / "adj_eff.parquet"), asof_snapshots=None,
                            dayzero_by_season=dz, adjself_snapshots=pd.read_parquet(PROC / "adjself_asof.parquet"))
    t = games[(games.Season == season) & (games.DayNum == daynum) & (games.A_TeamID.isin(fm.home_id))].copy()

    model = pickle.load(open(PROC / "reg_model.pkl", "rb"))
    feats = model.margin_features + model.total_features
    for f in feats:  # men's slate won't generate women-only cols (d_tv_*); 0 matches training's fill
        if f not in t.columns:
            t[f] = 0.0
    t[feats] = t[feats].fillna(0)
    ps = model.predict_scores(t)
    t["our_margin"], t["our_total"], t["our_wp"] = ps.pred_margin.to_numpy(), ps.pred_total.to_numpy(), model.predict_batch(t)

    t = t.merge(fm, left_on="A_TeamID", right_on="home_id")
    nm = _rd("MTeams").set_index("TeamID").TeamName.to_dict()
    out = pd.DataFrame({
        "matchup": [f"{nm.get(v)} @ {nm.get(h)}" for v, h in zip(t.vis_id, t.home_id)],
        "our_spread": (-t.our_margin).round(1), "fm_spread": (-(t.HomePred - t.VisitorPred)).round(1),
        "our_total": t.our_total.round(0), "fm_total": (t.HomePred + t.VisitorPred).round(0),
        "our_wp": t.our_wp.round(3), "fm_wp": (t.HomeWP / 100).round(3),
    })
    out["our_ml"] = [_prob_to_ml(p) for p in t.our_wp]
    out["spread_gap"] = (out.our_spread - out.fm_spread).round(1)
    out["total_gap"] = (out.our_total - out.fm_total).round(0)
    out.attrs["dropped"] = int(dropped)
    out = out.iloc[out.spread_gap.abs().sort_values(ascending=False).index].reset_index(drop=True)
    return out


_HTML = """<title>Daily slate — us vs FanMatch ({date})</title>
<style>
:root{{color-scheme:light dark}} body{{font:14px/1.5 system-ui;margin:24px;max-width:1000px}}
h1{{font-size:18px}} table{{border-collapse:collapse;width:100%}} th,td{{padding:6px 10px;text-align:right;border-bottom:1px solid #8884}}
td:first-child,th:first-child{{text-align:left}} .big{{font-weight:700;color:#e0603a}} caption{{text-align:left;color:#888;padding-bottom:8px}}
</style>
<h1>🏀 {date} — {n} games · us (v11) vs KenPom FanMatch</h1>
<caption>Sorted by spread disagreement. Positive gap = we favor the home team more than FanMatch. Spread is the home line.</caption>
<table><tr><th>Matchup (vis @ home)</th><th>Our spread</th><th>FM spread</th><th>Δ</th>
<th>Our total</th><th>FM total</th><th>Δ</th><th>Our ML</th><th>Our WP</th><th>FM WP</th></tr>
{rows}
</table>"""


def _html(df: pd.DataFrame, date: str) -> str:
    def cell(g):
        return f'<td class="big">{g:+.1f}</td>' if abs(g) >= 4 else f"<td>{g:+.1f}</td>"
    rows = "".join(
        f"<tr><td>{r.matchup}</td><td>{r.our_spread:+.1f}</td><td>{r.fm_spread:+.1f}</td>{cell(r.spread_gap)}"
        f"<td>{r.our_total:.0f}</td><td>{r.fm_total:.0f}</td><td>{r.total_gap:+.0f}</td>"
        f"<td>{r.our_ml:+d}</td><td>{r.our_wp:.2f}</td><td>{r.fm_wp:.2f}</td></tr>"
        for r in df.itertuples()
    )
    return _HTML.format(date=date, n=len(df), rows=rows)


def main() -> None:
    date = sys.argv[1] if len(sys.argv) > 1 else pd.Timestamp.today().strftime("%Y-%m-%d")
    season = pd.to_datetime(date).year + (1 if pd.to_datetime(date).month >= 8 else 0)
    df = handicap_slate(date, season)
    out = Path(f"monitoring/slate_{date}.html")
    out.write_text(_html(df, date))
    pd.set_option("display.max_rows", None, "display.width", 160)
    print(f"{date}: {len(df)} games ({df.attrs['dropped']} unmapped dropped) → {out}\n")
    print(df[["matchup", "our_spread", "fm_spread", "spread_gap", "our_total", "fm_total", "our_wp", "fm_wp", "our_ml"]].to_string(index=False))
    print(f"\nagreement: spread corr {np.corrcoef(df.our_spread, df.fm_spread)[0, 1]:.3f}  "
          f"total corr {np.corrcoef(df.our_total, df.fm_total)[0, 1]:.3f}  wp corr {np.corrcoef(df.our_wp, df.fm_wp)[0, 1]:.3f}")


if __name__ == "__main__":
    main()
