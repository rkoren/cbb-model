"""Iterative SOS-adjusted efficiency ratings (reverse-engineered KenPom).

Algorithm: for each team-season, adjust raw per-game OE/DE by the opponent's
current adjusted rating, normalize to league average, repeat until convergence.
Home court is removed before iteration so schedule imbalance doesn't inflate ratings.
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

# Possession-weighted means and iterative AdjTempo were tested and reverted (2026-03-15):
# Brier 0.169167 → 0.169696, no KenPom r improvement. Single-pass tempo is equivalent.
_FT_FACTOR = 0.475


def compute_adj_efficiency(
    reg_sym: pd.DataFrame,
    max_iter: int = 50,
    tol: float = 1e-4,
    hca_oe: float = 1.4,
) -> pd.DataFrame:
    """Compute AdjOE, AdjDE, AdjEM, AdjTempo per (Season, men_women, TeamID).

    Args:
        reg_sym: Symmetric regular-season game log (one row per team per game).
                 Required columns: Season, men_women, T1_TeamID, T2_TeamID,
                 T1_Score, T2_Score, T1_FGA, T1_OR, T1_TO, T1_FTA,
                 T2_FGA, T2_OR, T2_TO, T2_FTA. Optional: T1_home (-1/0/1).
        max_iter: Maximum convergence iterations.
        tol: Stop when max per-team rating change drops below this.
        hca_oe: Points-per-100-poss home court advantage to strip before iteration.

    Returns:
        DataFrame with columns: Season, men_women, TeamID, AdjOE, AdjDE, AdjEM,
        AdjTempo, iters.
    """
    records = []
    groups = reg_sym.groupby(["men_women", "Season"])

    for (mw, season), grp in tqdm(groups, desc="Adj efficiency", total=groups.ngroups):
        grp = grp.reset_index(drop=True)

        t1_poss = grp["T1_FGA"] - grp["T1_OR"] + grp["T1_TO"] + _FT_FACTOR * grp["T1_FTA"]
        t2_poss = grp["T2_FGA"] - grp["T2_OR"] + grp["T2_TO"] + _FT_FACTOR * grp["T2_FTA"]
        poss = ((t1_poss + t2_poss) / 2).clip(lower=40)

        raw_oe = (100 * grp["T1_Score"] / poss).values
        raw_de = (100 * grp["T2_Score"] / poss).values

        # Strip home court so schedule composition doesn't bias ratings
        t1_home = grp["T1_home"].values if "T1_home" in grp.columns else np.zeros(len(grp))
        adj_raw_oe = raw_oe - t1_home * hca_oe
        adj_raw_de = raw_de + t1_home * hca_oe

        t1_ids = grp["T1_TeamID"].values
        t2_ids = grp["T2_TeamID"].values
        teams = sorted(set(t1_ids))

        team_game_idx: dict[int, list[int]] = defaultdict(list)
        for i, t in enumerate(t1_ids):
            team_game_idx[t].append(i)

        adj_oe = {t: float(np.mean(adj_raw_oe[team_game_idx[t]])) if team_game_idx[t] else 100.0 for t in teams}
        adj_de = {t: float(np.mean(adj_raw_de[team_game_idx[t]])) if team_game_idx[t] else 100.0 for t in teams}
        league_avg = float(np.mean(list(adj_oe.values())))

        converged_iter = max_iter
        for it in range(max_iter):
            new_oe, new_de = {}, {}
            for t in teams:
                idx = team_game_idx[t]
                if not idx:
                    new_oe[t], new_de[t] = adj_oe[t], adj_de[t]
                    continue
                opp_ade = np.array([adj_de.get(t2_ids[i], league_avg) for i in idx])
                opp_aoe = np.array([adj_oe.get(t2_ids[i], league_avg) for i in idx])
                new_oe[t] = float(np.mean(adj_raw_oe[idx] / np.maximum(opp_ade, 1) * league_avg))
                new_de[t] = float(np.mean(adj_raw_de[idx] / np.maximum(opp_aoe, 1) * league_avg))

            scale_oe = league_avg / max(float(np.mean(list(new_oe.values()))), 1)
            scale_de = league_avg / max(float(np.mean(list(new_de.values()))), 1)
            new_oe = {t: v * scale_oe for t, v in new_oe.items()}
            new_de = {t: v * scale_de for t, v in new_de.items()}

            max_chg = max(
                max(abs(new_oe[t] - adj_oe.get(t, 0)) for t in teams),
                max(abs(new_de[t] - adj_de.get(t, 0)) for t in teams),
            )
            adj_oe, adj_de = new_oe, new_de
            if max_chg < tol:
                converged_iter = it + 1
                break

        # Single-pass tempo (iterative equivalent, see module docstring)
        tempo_raw = {
            t: float(np.mean(poss.values[team_game_idx[t]])) if team_game_idx[t] else float(poss.mean())
            for t in teams
        }
        lg_tempo = float(np.mean(list(tempo_raw.values())))
        adj_tempo = {}
        for t in teams:
            idx = team_game_idx[t]
            opp_tempos = np.array([tempo_raw.get(t2_ids[i], lg_tempo) for i in idx])
            adj_tempo[t] = float(np.mean(poss.values[idx] / np.maximum(opp_tempos, 1) * lg_tempo))

        for t in teams:
            records.append({
                "Season": season,
                "men_women": mw,
                "TeamID": t,
                "AdjOE": adj_oe[t],
                "AdjDE": adj_de[t],
                "AdjEM": adj_oe[t] - adj_de[t],
                "AdjTempo": adj_tempo[t],
                "iters": converged_iter,
            })

    return pd.DataFrame(records)
