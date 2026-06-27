"""Validate a filled holdout template before finalizing (catches data-entry mistakes).

Single-elimination has strong invariants we can check against the filled `Winner` column:
  - every Winner is one of that game's two teams;
  - all 68 field teams appear, and only field teams;
  - processing games in date order, no team plays after it has lost (no double-loss);
  - exactly one undefeated team — the champion — and it wins the last game.

It also FLAGS (not errors) games the winner was a big KenPom underdog (low `KP_*_WP`), since a
1-seed "losing" round 1 is either a historic upset or a typo worth a second look.

    python scripts/validate_holdout_template.py    # reads data/holdout/2026_template.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

TEMPLATE = Path("data/holdout/2026_template.csv")
SEEDS = Path("data/raw/MNCAATourneySeeds.csv")
SEASON = 2026
UPSET_WP = 0.15  # flag wins by a team KenPom gave < this win prob


def main() -> int:
    t = pd.read_csv(TEMPLATE)
    played = t[t["Winner"].notna() & (t["Winner"].astype(str).str.strip() != "")].copy()
    played["Winner"] = played["Winner"].astype(int)
    name = {}
    for r in t.itertuples(index=False):
        name[int(r.A_TeamID)] = r.A_Name
        name[int(r.B_TeamID)] = r.B_Name

    errors: list[str] = []
    flags: list[str] = []

    # 1. Winner ∈ {A, B}
    for r in played.itertuples(index=False):
        if r.Winner not in (int(r.A_TeamID), int(r.B_TeamID)):
            errors.append(f"{r.DateOfGame}: Winner {r.Winner} not in {r.A_Name} vs {r.B_Name}")

    # 2. field coverage
    seeds = pd.read_csv(SEEDS)
    field = set(seeds.loc[seeds.Season == SEASON, "TeamID"].astype(int))
    teams = set(played["A_TeamID"]) | set(played["B_TeamID"])
    if field - teams:
        errors.append(f"field teams with no game: {sorted(name.get(i, i) for i in field - teams)}")
    if teams - field:
        errors.append(f"non-field teams present: {sorted(name.get(i, i) for i in teams - field)}")
    if len(played) != 67:
        flags.append(f"{len(played)} games filled (a complete 68-team bracket has 67)")

    # 3. advancement in date order — no team plays after a loss
    eliminated: set[int] = set()
    for r in played.sort_values("DateOfGame").itertuples(index=False):
        a, b, w = int(r.A_TeamID), int(r.B_TeamID), int(r.Winner)
        for tm in (a, b):
            if tm in eliminated:
                errors.append(f"{r.DateOfGame}: {name.get(tm, tm)} plays after already losing")
        eliminated.add(b if w == a else a)

    # 4. exactly one undefeated champion = winner of the last game
    undefeated = teams - eliminated
    if len(undefeated) != 1:
        errors.append(f"expected 1 undefeated champion, got {sorted(name.get(i, i) for i in undefeated)}")
    else:
        champ = next(iter(undefeated))
        last = played.sort_values("DateOfGame").iloc[-1]
        if int(last["Winner"]) != champ:
            errors.append(f"champion {name.get(champ)} isn't the winner of the last game")

    # 5. upset flags (soft)
    for r in played.itertuples(index=False):
        w = int(r.Winner)
        w_wp = r.KP_A_WP if w == int(r.A_TeamID) else 1 - r.KP_A_WP
        if w_wp < UPSET_WP:
            errors_team = name.get(w)
            other = name.get(int(r.B_TeamID) if w == int(r.A_TeamID) else int(r.A_TeamID))
            flags.append(f"{r.DateOfGame}: UPSET — {errors_team} (KP WP {w_wp:.0%}) beat {other}")

    print(f"Validated {len(played)} games.\n")
    if errors:
        print(f"❌ {len(errors)} ERROR(S) — fix before finalizing:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("✓ Bracket is internally consistent (no structural errors).")
    if flags:
        print(f"\n⚠ {len(flags)} flag(s) to eyeball (likely upsets, not necessarily wrong):")
        for f in flags:
            print(f"  - {f}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
