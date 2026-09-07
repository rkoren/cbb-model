"""KenPom **preseason projection** as a season-constant, leak-free prior for the reg model.

KenPom publishes a roster-aware "game-0" projection each October (``ratings_archive(preseason=True)``),
made *after* the transfer portal / roster confirmation — so it knows *this year's* team, while the
prior-season ``*_prev`` priors describe last year's roster. Feature-first (men 2012–2026, LOTO OLS
margin MAE on games before the first as-of snapshot): preseason ``AdjEM`` diff **9.44 vs 9.97** for
the prior-season diff, and it still helps late in the season (D≥71: 9.19 vs 9.53) — the roster
information never fully fades. Roster ``Continuity``/``Exp`` (the ``height`` endpoint) add ~nothing
on top of the projection (which already embeds them); the full-model LOTO agreed (Brier −0.0001,
men-2026 vs FanMatch slightly *worse*), and the cached height parquets are end-of-season realized
minutes while a new season's October fetch is a projection (train/serve drift) — so ``include_roster``
defaults to **off**; the option stays for experiments.

The projection is cached by ``scripts/fetch_kenpom_archive.py`` inside each season's archive
parquet, stamped ``ArchiveDate = DayZero − 1`` (GM-005); this module lifts exactly that row.
Offline (parquets only, no API); name→TeamID via :func:`cbb.kenpom.features.build_team_name_map`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .features import build_team_name_map

log = logging.getLogger(__name__)

# Archive preseason column -> prior feature name (season-constant; d_ for the margin head, s_ for total).
PRE_COLS: dict[str, str] = {
    "AdjEM": "kp_pre_AdjEM",
    "AdjOE": "kp_pre_AdjOE",
    "AdjDE": "kp_pre_AdjDE",
    "AdjTempo": "kp_pre_AdjTempo",
}
# Roster descriptors from the `height` endpoint (d_ only — a sum of continuity says nothing about a total).
ROSTER_COLS: dict[str, str] = {"Continuity": "kp_Continuity", "Exp": "kp_Exp"}
PRE_FEATURES: list[str] = list(PRE_COLS.values())
ROSTER_FEATURES: list[str] = list(ROSTER_COLS.values())
KEY_COLS = ["Season", "TeamID"]


def load_preseason_priors(
    seasons: list[int],
    m_teams: pd.DataFrame,
    team_spellings: pd.DataFrame | None,
    dayzero_by_season: dict,
    kenpom_dir: Path = Path("data/kenpom"),
    include_roster: bool = False,
) -> pd.DataFrame:
    """Season-constant KenPom priors, keyed ``(Season, TeamID)`` — men only (KenPom has no women).

    Per season: the archive parquet's ``ArchiveDate == DayZero − 1`` rows (the preseason projection)
    → ``kp_pre_*``; with ``include_roster`` the height parquet's ``Continuity``/``Exp`` →
    ``kp_Continuity``/``kp_Exp`` (off by default — see module docstring). Seasons with no cached archive (pre-2012, or next season before the October fetch)
    are simply absent → NaN downstream, the same "missing" signal as every other as-of feature.
    """
    cols = PRE_FEATURES + (ROSTER_FEATURES if include_roster else [])
    parts = []
    for s in seasons:
        arch_path = kenpom_dir / "archive" / f"kenpom_archive_{s}.parquet"
        if not arch_path.exists() or s not in dayzero_by_season:
            continue
        arch = pd.read_parquet(arch_path)
        pre_date = pd.Timestamp(dayzero_by_season[s]) - pd.Timedelta(days=1)
        pre = arch[pd.to_datetime(arch["ArchiveDate"]) == pre_date]
        if pre.empty:
            log.warning("No preseason (DayZero-1) snapshot in %s — season %d gets no kp_pre_* prior", arch_path, s)
            continue
        tmap = build_team_name_map(m_teams, arch[["TeamName"]].drop_duplicates(), team_spellings)
        pre = pre.rename(columns=PRE_COLS).assign(TeamID=pre["TeamName"].map(tmap))
        pre = pre.dropna(subset=["TeamID"]).astype({"TeamID": int})[["TeamID", *PRE_FEATURES]]

        if include_roster:
            h_path = kenpom_dir / f"kenpom_height_{s}.parquet"
            if h_path.exists():
                h = pd.read_parquet(h_path)
                hmap = build_team_name_map(m_teams, h[["TeamName"]].drop_duplicates(), team_spellings)
                h = h.rename(columns=ROSTER_COLS).assign(TeamID=h["TeamName"].map(hmap))
                h = h.dropna(subset=["TeamID"]).astype({"TeamID": int})[["TeamID", *ROSTER_FEATURES]]
                pre = pre.merge(h.drop_duplicates("TeamID"), on="TeamID", how="left")
            else:
                for c in ROSTER_FEATURES:
                    pre[c] = float("nan")
        parts.append(pre.drop_duplicates("TeamID").assign(Season=s)[KEY_COLS + cols])
    if not parts:
        return pd.DataFrame(columns=KEY_COLS + cols)
    out = pd.concat(parts, ignore_index=True)
    log.info("Preseason priors loaded: %d team-seasons over %d seasons", len(out), out["Season"].nunique())
    return out
