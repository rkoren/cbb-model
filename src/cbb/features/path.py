"""Tournament path features: round number, path quality, prior margin.

For each tournament game, captures how difficult each team's path has been
to get there. Round 1 teams have no prior path; later rounds accumulate history.

Path quality = mean AdjEM of opponents beaten so far (0 for round 1).
Prior margin = mean point margin in prior tournament games (0 for round 1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_path_features(
    tourn_sym: pd.DataFrame,
    adj_eff: pd.DataFrame,
) -> pd.DataFrame:
    """Compute tournament path features for each game in symmetric matchup format.

    Processes each season sequentially in DayNum order so cumulative path history
    is available when each game is processed.

    Args:
        tourn_sym: Symmetric tournament game log. Required columns: Season,
                   men_women, DayNum, T1_TeamID, T2_TeamID, Outcome, PointDiff.
                   Outcome=1 means T1 won (used to determine actual game winner).
        adj_eff: Output of compute_adj_efficiency(). Used to look up opponent AdjEM
                 for path quality computation.

    Returns:
        DataFrame with columns: Season, men_women, A_TeamID, B_TeamID, round_num,
        A_path_quality, B_path_quality, d_path_quality, A_prior_margin,
        B_prior_margin, d_prior_margin. One row per (Season, men_women, A_TeamID,
        B_TeamID) pair — same symmetric structure as the matchup dataset.
    """
    adj_em_lkp = adj_eff.set_index(["men_women", "Season", "TeamID"])["AdjEM"].to_dict()

    # One row per physical game (winner perspective, sorted chronologically)
    winners = tourn_sym[tourn_sym["Outcome"] == 1].copy()
    winners = winners.sort_values(["Season", "men_women", "DayNum"])

    # Round number = dense rank of DayNum within (Season, men_women)
    # Play-in games (First Four) appear on earlier DayNums and get round_num=1;
    # their winners then play in round_num=2 games.
    winners["round_num"] = winners.groupby(["Season", "men_women"])["DayNum"].transform(
        lambda x: x.rank(method="dense").astype(int)
    )

    records: list[dict] = []

    for (mw, season), grp in winners.groupby(["men_women", "Season"]):
        grp = grp.sort_values("DayNum")
        # team -> list of (opponent_adj_em, margin) from prior games this tournament
        team_history: dict[int, list[tuple[float, float]]] = {}

        for _, row in grp.iterrows():
            winner = int(row["T1_TeamID"])
            loser = int(row["T2_TeamID"])
            rnd = int(row["round_num"])
            margin = float(row["PointDiff"])

            w_hist = team_history.get(winner, [])
            l_hist = team_history.get(loser, [])

            w_path_q = float(np.mean([x[0] for x in w_hist])) if w_hist else 0.0
            l_path_q = float(np.mean([x[0] for x in l_hist])) if l_hist else 0.0
            w_prior_m = float(np.mean([x[1] for x in w_hist])) if w_hist else 0.0
            l_prior_m = float(np.mean([x[1] for x in l_hist])) if l_hist else 0.0

            loser_em = float(adj_em_lkp.get((mw, season, loser), 0.0))

            base = {"Season": season, "men_women": mw, "round_num": rnd}
            records.append({
                **base,
                "A_TeamID": winner, "B_TeamID": loser,
                "A_path_quality": w_path_q, "B_path_quality": l_path_q,
                "A_prior_margin": w_prior_m, "B_prior_margin": l_prior_m,
            })
            records.append({
                **base,
                "A_TeamID": loser, "B_TeamID": winner,
                "A_path_quality": l_path_q, "B_path_quality": w_path_q,
                "A_prior_margin": l_prior_m, "B_prior_margin": w_prior_m,
            })

            team_history.setdefault(winner, []).append((loser_em, margin))

    df = pd.DataFrame(records)
    df["d_path_quality"] = df["A_path_quality"] - df["B_path_quality"]
    df["d_prior_margin"] = df["A_prior_margin"] - df["B_prior_margin"]
    return df
