"""Score a single men's matchup with the v11 reg model → full handicap (spread/total/moneyline).

Shared by the FastAPI ``/predict`` endpoint and any ad-hoc query. Uses the same result-less
"upcoming game" construction as ``scripts/daily_slate.py`` (append the matchup with no result, rebuild
reg-games so as-of features come only from games before the date — leak-free), but scopes the raw log
to the last few seasons so it's fast enough for a request while Elo stays converged.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cbb.features.reg_games import build_reg_games

RAW, PROC, LIVE = Path("data/raw"), Path("data/processed"), Path("data/live")
SCOPE_SEASONS = 5  # current + 4 prior — Elo has converged; keeps the build fast


def _prob_to_ml(p: float) -> int:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


@dataclass
class Engine:
    """Everything loaded once at startup for fast per-request scoring."""
    model: object
    adj_eff: pd.DataFrame
    adjself: pd.DataFrame
    dayzero: dict
    mreg: pd.DataFrame            # Kaggle history + live results log (men)
    wreg: pd.DataFrame
    wteams: pd.DataFrame
    mteams: pd.DataFrame
    name_to_id: dict


def load_engine() -> Engine:
    def rd(n, e=None):
        return pd.read_csv(RAW / f"{n}.csv", encoding=e)
    mreg = rd("MRegularSeasonDetailedResults")
    live = sorted((LIVE).glob("mreg_live_*.csv")) if LIVE.exists() else []
    if live:
        mreg = pd.concat([mreg, *[pd.read_csv(p) for p in live]], ignore_index=True).drop_duplicates(
            ["Season", "DayNum", "WTeamID", "LTeamID"], keep="first")
    ms = rd("MSeasons")
    mteams = rd("MTeams")
    name_to_id = {str(r.TeamName).lower(): int(r.TeamID) for r in mteams.itertuples()}
    for r in rd("MTeamSpellings", "latin-1").itertuples():
        name_to_id.setdefault(str(r.TeamNameSpelling).lower(), int(r.TeamID))
    return Engine(
        model=pickle.load(open(PROC / "reg_model.pkl", "rb")),
        adj_eff=pd.read_parquet(PROC / "adj_eff.parquet"),
        adjself=pd.read_parquet(PROC / "adjself_asof.parquet"),
        dayzero=dict(zip(ms.Season, pd.to_datetime(ms.DayZero))),
        mreg=mreg, wreg=rd("WRegularSeasonDetailedResults"), wteams=rd("WTeams"), mteams=mteams,
        name_to_id=name_to_id,
    )


def resolve_team(eng: Engine, team) -> int:
    """Accept a Kaggle TeamID (int/str-int) or a team name/spelling → TeamID."""
    if isinstance(team, int) or (isinstance(team, str) and team.isdigit()):
        return int(team)
    tid = eng.name_to_id.get(str(team).lower())
    if tid is None:
        raise KeyError(f"unknown team '{team}'")
    return tid


def handicap_matchup(eng: Engine, team_a, team_b, date: str, venue: str = "home_a") -> dict:
    """venue: 'home_a' (a hosts), 'home_b' (b hosts), or 'neutral'."""
    a, b = resolve_team(eng, team_a), resolve_team(eng, team_b)
    season = pd.to_datetime(date).year + (1 if pd.to_datetime(date).month >= 8 else 0)
    daynum = (pd.to_datetime(date) - eng.dayzero[season]).days

    mreg = eng.mreg[eng.mreg.Season >= season - SCOPE_SEASONS + 1]
    mreg = mreg[~((mreg.Season == season) & (mreg.DayNum >= daynum))]
    wloc = {"home_a": "H", "home_b": "A", "neutral": "N"}[venue]  # WLoc is the winner's (=A=team_a) venue
    syn = pd.DataFrame([{"Season": season, "DayNum": daynum, "WTeamID": a, "LTeamID": b,
                         "WScore": 100, "LScore": 99, "WLoc": wloc, "NumOT": 0}])
    data = {"M_reg_raw": pd.concat([mreg, syn], ignore_index=True),
            "W_reg_raw": eng.wreg, "M_teams": eng.mteams, "W_teams": eng.wteams}
    games = build_reg_games(data, eng.adj_eff, asof_snapshots=None,
                            dayzero_by_season=eng.dayzero, adjself_snapshots=eng.adjself)
    t = games[(games.Season == season) & (games.DayNum == daynum) & (games.A_TeamID == a) & (games.B_TeamID == b)].copy()
    if t.empty:
        raise ValueError("matchup row not built (check team IDs / date)")
    feats = eng.model.margin_features + eng.model.total_features
    for f in feats:
        if f not in t.columns:
            t[f] = 0.0
    t[feats] = t[feats].fillna(0)
    ps = eng.model.predict_scores(t)
    margin = float(ps.pred_margin.iloc[0])   # A − B (team_a − team_b)
    total = float(ps.pred_total.iloc[0])
    wp_a = float(eng.model.predict_batch(t)[0])
    nm = eng.mteams.set_index("TeamID").TeamName.to_dict()
    return {
        "matchup": f"{nm.get(a)} vs {nm.get(b)}", "date": date, "venue": venue,
        "predicted_score": {nm.get(a): round((total + margin) / 2, 1), nm.get(b): round((total - margin) / 2, 1)},
        "spread": {"favorite": nm.get(a) if margin > 0 else nm.get(b), "line": -round(abs(margin), 1)},
        "total": round(total, 1),
        "win_prob": {nm.get(a): round(wp_a, 3), nm.get(b): round(1 - wp_a, 3)},
        "moneyline": {nm.get(a): _prob_to_ml(wp_a), nm.get(b): _prob_to_ml(1 - wp_a)},
    }
