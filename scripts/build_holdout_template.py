"""Build the 2026 men's holdout template from KenPom fanmatch (KP-001).

fanmatch returns KenPom's *predictions* (no actual outcomes), so this can't produce final
results — but it gives the exact 2026 tournament game list + KenPom's predicted scores/WP
(a benchmark to beat). We write a template with one row per game (mapped to Kaggle TeamIDs,
filtered to the 68-team field) and a blank ``Winner`` column for you to fill with the winning
TeamID. Then ``cbb.holdout.finalize_template`` converts it to the
``data/holdout/tourney_results_2026.csv`` contract the holdout reads.

    python scripts/build_holdout_template.py    # writes data/holdout/2026_template.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from cbb.kenpom import KenPomClient
from cbb.kenpom.features import build_team_name_map

SEASON = 2026
# 2026 bracket dates (Selection Sunday 2026-03-15): First Four → Championship.
BRACKET_DATES = [
    "2026-03-17", "2026-03-18",                # First Four
    "2026-03-19", "2026-03-20",                # Round 1
    "2026-03-21", "2026-03-22",                # Round 2
    "2026-03-26", "2026-03-27",                # Sweet 16
    "2026-03-28", "2026-03-29",                # Elite 8
    "2026-04-04",                              # Final Four
    "2026-04-06",                              # Championship
]
OUT = Path("data/holdout/2026_template.csv")


def main() -> None:
    client = KenPomClient()
    m_teams = pd.read_csv("data/raw/MTeams.csv")
    name_to_id = build_team_name_map(m_teams, client.teams(year=SEASON))

    seeds = pd.read_csv("data/raw/MNCAATourneySeeds.csv")
    field = set(seeds.loc[seeds["Season"] == SEASON, "TeamID"].astype(int))
    print(f"2026 men's field: {len(field)} teams")

    rows: list[dict] = []
    seen: set[tuple[int, int]] = set()
    unmapped: set[str] = set()

    for date in BRACKET_DATES:
        try:
            fm = client.fanmatch(date)
        except Exception as exc:  # noqa: BLE001
            print(f"  {date}: fanmatch failed ({exc})")
            continue
        n_field = 0
        for g in fm.itertuples(index=False):
            home_id = name_to_id.get(g.Home)
            vis_id = name_to_id.get(g.Visitor)
            if home_id is None:
                unmapped.add(g.Home)
            if vis_id is None:
                unmapped.add(g.Visitor)
            if home_id is None or vis_id is None:
                continue
            if home_id not in field or vis_id not in field:
                continue  # not an NCAA tournament game (NIT / other postseason)
            a_id, b_id = (home_id, vis_id) if home_id < vis_id else (vis_id, home_id)
            if (a_id, b_id) in seen:
                continue
            seen.add((a_id, b_id))
            a_is_home = a_id == home_id
            rows.append({
                "Season": SEASON,
                "DateOfGame": g.DateOfGame,
                "A_TeamID": a_id,
                "A_Name": g.Home if a_is_home else g.Visitor,
                "B_TeamID": b_id,
                "B_Name": g.Visitor if a_is_home else g.Home,
                "KP_A_Pred": g.HomePred if a_is_home else g.VisitorPred,
                "KP_B_Pred": g.VisitorPred if a_is_home else g.HomePred,
                "KP_A_WP": (g.HomeWP if a_is_home else 100 - g.HomeWP) / 100.0,
                "Winner": "",  # ← fill with the winning TeamID (A_TeamID or B_TeamID)
            })
            n_field += 1
        print(f"  {date}: {len(fm)} games, {n_field} in field")

    template = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(OUT, index=False)
    print(f"\nWrote {OUT} — {len(template)} games (single-elim 68-team bracket = 67 expected)")
    if unmapped:
        print(f"Unmapped fanmatch names (check if any are tournament teams): {sorted(unmapped)}")


if __name__ == "__main__":
    main()
