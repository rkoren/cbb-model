"""Raw data loading and symmetric game representation.

Converts Kaggle W/L game logs to a symmetric T1/T2 format used by all
feature-engineering modules. Extracted here so both the Prefect flow
(experiments/baseline.py) and the kitchen-platform adapter (src/features/run.py)
can import it without either one depending on the other.
"""

from __future__ import annotations

import pandas as pd

BOX_COLS = [
    "Score", "FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA",
    "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF",
]


def _parse_seed(s: str | None) -> int | None:
    """Extract seed number from a Kaggle seed string like 'W01' or 'X16a'."""
    digits = "".join(filter(str.isdigit, str(s)[1:3]))
    return int(digits) if digits else None


def normalize_games(df: pd.DataFrame, men_women: int, is_tourn: bool = False) -> pd.DataFrame:
    """Convert a W/L game log to symmetric T1/T2 format, OT-adjusted.

    For every game row in *df*, two rows are emitted:
    one with the winner as T1 (Outcome=1) and one with the loser as T1 (Outcome=0).
    Box-score stats are divided by an OT adjustment factor so 40-minute stats
    are comparable regardless of overtime periods played.

    Args:
        df: Raw Kaggle game DataFrame (MRegularSeasonDetailedResults format).
        men_women: 0 for men's games, 1 for women's games.
        is_tourn: True for tournament games, False for regular-season games.

    Returns:
        Symmetric DataFrame with T1_*/T2_* box-score columns and Outcome/PointDiff.
    """
    records = []
    for _, row in df.iterrows():
        adjot = (40 + 5 * row.get("NumOT", 0)) / 40
        base = {
            "Season": row["Season"],
            "DayNum": row["DayNum"],
            "men_women": men_women,
            "is_tourn": int(is_tourn),
        }
        loc = row.get("WLoc", "N")

        for t1_is_winner in (True, False):
            rec = dict(base)
            if t1_is_winner:
                rec["T1_TeamID"] = row["WTeamID"]
                rec["T2_TeamID"] = row["LTeamID"]
                rec["Outcome"] = 1
                rec["T1_home"] = 1 if loc == "H" else (-1 if loc == "A" else 0)
                for c in BOX_COLS:
                    rec[f"T1_{c}"] = row[f"W{c}"] / adjot
                    rec[f"T2_{c}"] = row[f"L{c}"] / adjot
            else:
                rec["T1_TeamID"] = row["LTeamID"]
                rec["T2_TeamID"] = row["WTeamID"]
                rec["Outcome"] = 0
                rec["T1_home"] = -1 if loc == "H" else (1 if loc == "A" else 0)
                for c in BOX_COLS:
                    rec[f"T1_{c}"] = row[f"L{c}"] / adjot
                    rec[f"T2_{c}"] = row[f"W{c}"] / adjot
            rec["PointDiff"] = rec["T1_Score"] - rec["T2_Score"]
            records.append(rec)
    return pd.DataFrame(records)


def build_symmetric_games(
    data: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build symmetric reg-season and tournament DataFrames from Kaggle raw data.

    Args:
        data: Dict keyed by Kaggle file stem — must include M_reg_raw, W_reg_raw,
              M_tourn_raw, W_tourn_raw (loaded from Kaggle CSVs).

    Returns:
        (reg_sym, tourn_sym) — both in symmetric T1/T2 format.
    """
    reg_sym = pd.concat([
        normalize_games(data["M_reg_raw"], men_women=0),
        normalize_games(data["W_reg_raw"], men_women=1),
    ], ignore_index=True)

    tourn_sym = pd.concat([
        normalize_games(data["M_tourn_raw"], men_women=0, is_tourn=True),
        normalize_games(data["W_tourn_raw"], men_women=1, is_tourn=True),
    ], ignore_index=True)

    return reg_sym, tourn_sym


def build_seed_lookup(
    m_seeds: pd.DataFrame,
    w_seeds: pd.DataFrame,
) -> dict[tuple[int, int], int]:
    """Build a (Season, TeamID) → SeedNum lookup from Kaggle seed DataFrames."""
    seeds = pd.concat([m_seeds, w_seeds], ignore_index=True)
    seeds["SeedNum"] = seeds["Seed"].apply(_parse_seed)
    return (
        seeds.dropna(subset=["SeedNum"])
        .astype({"Season": int, "TeamID": int, "SeedNum": int})
        .set_index(["Season", "TeamID"])["SeedNum"]
        .to_dict()
    )
