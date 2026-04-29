"""GLM team quality coefficients from a sparse margin regression."""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import statsmodels.api as sm
from tqdm import tqdm


def compute_glm_quality(reg_sym: pd.DataFrame) -> pd.DataFrame:
    """Per-season team quality via OLS: PointDiff ~ T1_dummies - T2_dummies.

    Each team's coefficient is its average contribution to point margin, implicitly
    SOS-adjusted through joint estimation. Serves as a cross-signal alternative to
    iterative AdjEM — useful because it uses a different functional form.

    Args:
        reg_sym: Symmetric regular-season game log. Requires: men_women, Season,
                 T1_TeamID, T2_TeamID, PointDiff.

    Returns:
        DataFrame with columns: Season, men_women, TeamID, Quality.
    """
    records = []
    groups = reg_sym.groupby(["men_women", "Season"])

    for (mw, season), grp in tqdm(groups, desc="GLM quality", total=groups.ngroups):
        grp = grp.reset_index(drop=True)
        all_teams = sorted(set(grp["T1_TeamID"].unique()) | set(grp["T2_TeamID"].unique()))
        team_idx = {t: i for i, t in enumerate(all_teams)}
        n, k = len(grp), len(all_teams)

        rows = list(range(n)) * 2
        cols = [team_idx[t] for t in grp["T1_TeamID"]] + [team_idx[t] for t in grp["T2_TeamID"]]
        vals = [1.0] * n + [-1.0] * n
        X = sp.csr_matrix((vals, (rows, cols)), shape=(n, k))
        y = grp["PointDiff"].values

        try:
            coefs = sm.OLS(y, X.toarray()).fit().params
        except Exception:
            coefs = np.zeros(k)

        for t, i in team_idx.items():
            records.append({"Season": season, "men_women": mw, "TeamID": t, "Quality": coefs[i]})

    return pd.DataFrame(records)
