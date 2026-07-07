"""As-of ratings log (DASH-001c) — the second of the M6 dashboard's two logs.

Where the predictions log (DASH-001a/b) compares *game* forecasts, this compares *ratings*: our
self-computed as-of AdjEM/OE/DE/Tempo (``cbb.features.adjself_asof``) beside a comparator's —
KenPom's archive for men, Torvik's for women — per ``(Season × weekly snapshot)``, with rank and
the ours-minus-theirs deltas. It's the visual independence proof for DASH-003 ("are our numbers
tracking KenPom's?") and feeds DASH-006's team trajectory.

Two load-bearing pieces, both pure and gender-agnostic (TeamIDs are Kaggle gender-disjoint):
  * **rank** — derived *identically* for both sides as ``rank(-AdjEM)`` (1 = best) within each
    snapshot, since the archive carries no rank column; this is what makes rank-delta meaningful.
  * **as-of alignment** — the comparator is aligned to *our* snapshot grid with a backward
    ``merge_asof`` (latest comparator snapshot ≤ our date), the same contract
    ``cbb.kenpom.asof_features`` uses, so a team with no prior comparator snapshot is NaN, never
    a wrong-date match.

Persistence + the real KenPom/Torvik wiring live in the dashboard assembly, not here.
"""

from __future__ import annotations

import pandas as pd

# The rating bases compared on both sides. Rank is derived from AdjEM.
_BASES = ["AdjEM", "AdjOE", "AdjDE", "AdjTempo"]
_RANK_BASE = "AdjEM"

# Generic comparator contract this builder consumes (KenPom archive / Torvik adapt to it).
COMPARATOR_COLS = ["Season", "TeamID", "snapshot_date"] + [f"cmp_{b}" for b in _BASES]


def adjself_to_ours(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Rename ``cbb.features.adjself_asof`` output (``adjself_<base>_asof``) to ``our_<base>``."""
    ren = {f"adjself_{b}_asof": f"our_{b}" for b in _BASES}
    return snapshots.rename(columns=ren)[["Season", "TeamID", "ArchiveDate", *ren.values()]]


def ratings_to_comparator(
    df: pd.DataFrame, rating_cols: dict[str, str], season: int, snapshot_date: pd.Timestamp
) -> pd.DataFrame:
    """Adapt a raw ratings frame (KenPom archive / Torvik) to :data:`COMPARATOR_COLS`.

    Args:
        df: Raw ratings, one row per team, already keyed to Kaggle ``TeamID``.
        rating_cols: source column → base name, e.g. ``{"AdjEM": "AdjEM", ...}`` (KenPom) — only
            the four bases are kept.
        season: Season stamped on every row.
        snapshot_date: The archive date these ratings are as-of.
    """
    ren = {src: f"cmp_{base}" for src, base in rating_cols.items() if base in _BASES}
    out = df.rename(columns=ren).copy()
    out["Season"] = season
    out["snapshot_date"] = pd.to_datetime(snapshot_date)
    return out[COMPARATOR_COLS].reset_index(drop=True)


def _rank_desc(s: pd.Series) -> pd.Series:
    """rank(-value): 1 = highest (best). NaN stays NaN; ties share the lower rank."""
    return s.rank(ascending=False, method="min")


def build_ratings_log(ours: pd.DataFrame, comparator: pd.DataFrame) -> pd.DataFrame:
    """One row per (Season, ArchiveDate, TeamID): our ratings + rank vs the comparator's + deltas.

    ``ours`` has ``Season, TeamID, ArchiveDate, our_<base>`` (see :func:`adjself_to_ours`);
    ``comparator`` follows :data:`COMPARATOR_COLS`. The comparator is aligned to each of our
    snapshot dates by a backward ``merge_asof`` (latest ≤ our date, per Season+TeamID); teams with
    no prior comparator snapshot get NaN comparator ratings/rank/deltas. Rank is ``rank(-AdjEM)``
    within each of our snapshots, computed the same way for both sides.
    """
    our_cols = [f"our_{b}" for b in _BASES]
    cmp_cols = [f"cmp_{b}" for b in _BASES]

    if comparator.empty:
        merged = ours.copy()
        for c in cmp_cols:
            merged[c] = pd.NA
    else:
        merged = pd.merge_asof(
            ours.sort_values("ArchiveDate"),
            comparator.sort_values("snapshot_date"),
            left_on="ArchiveDate", right_on="snapshot_date",
            by=["Season", "TeamID"], direction="backward",
        )

    # Rank each side within our (Season, ArchiveDate) snapshot — identical derivation both sides.
    grp = merged.groupby(["Season", "ArchiveDate"], observed=True)
    merged["our_rank"] = grp[f"our_{_RANK_BASE}"].transform(_rank_desc)
    merged["cmp_rank"] = grp[f"cmp_{_RANK_BASE}"].transform(_rank_desc)

    for b in _BASES:
        merged[f"d_{b}"] = merged[f"our_{b}"] - merged[f"cmp_{b}"]
    merged["d_rank"] = merged["our_rank"] - merged["cmp_rank"]

    cols = (["Season", "ArchiveDate", "TeamID"] + our_cols + cmp_cols
            + ["our_rank", "cmp_rank"] + [f"d_{b}" for b in _BASES] + ["d_rank"])
    return merged[cols].sort_values(["Season", "ArchiveDate", "our_rank"]).reset_index(drop=True)
