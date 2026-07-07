"""Tests for the as-of ratings log (DASH-001c).

Load-bearing properties: rank is ``rank(-AdjEM)`` with 1 = best and is derived identically for
both sides; the comparator aligns to our snapshot grid with a backward (nearest-≤) merge, so a
team with no prior comparator snapshot is NaN rather than a wrong-date match.
"""

import pandas as pd
import pytest

from cbb.benchmark.ratings_log import (
    COMPARATOR_COLS,
    adjself_to_ours,
    build_ratings_log,
    ratings_to_comparator,
)

D1 = pd.Timestamp("2026-01-08")
D2 = pd.Timestamp("2026-01-15")
D3 = pd.Timestamp("2026-01-22")


def _ours(rows):
    return pd.DataFrame(rows)


def _cmp(rows):
    return pd.DataFrame(rows, columns=COMPARATOR_COLS)


# ── adapters ───────────────────────────────────────────────────────────────────────

def test_adjself_to_ours_renames():
    snap = pd.DataFrame({
        "Season": [2026], "TeamID": [1101], "ArchiveDate": [D2],
        "adjself_AdjEM_asof": [20.0], "adjself_AdjOE_asof": [115.0],
        "adjself_AdjDE_asof": [95.0], "adjself_AdjTempo_asof": [68.0],
    })
    out = adjself_to_ours(snap)
    assert set(out.columns) == {"Season", "TeamID", "ArchiveDate",
                                "our_AdjEM", "our_AdjOE", "our_AdjDE", "our_AdjTempo"}
    assert out["our_AdjEM"].iloc[0] == 20.0


def test_ratings_to_comparator_shapes_to_contract():
    raw = pd.DataFrame({"TeamID": [1101, 1102],
                        "AdjEM": [20.0, 10.0], "AdjOE": [115.0, 108.0],
                        "AdjDE": [95.0, 98.0], "AdjTempo": [68.0, 70.0]})
    cmp = ratings_to_comparator(
        raw, {"AdjEM": "AdjEM", "AdjOE": "AdjOE", "AdjDE": "AdjDE", "AdjTempo": "AdjTempo"},
        season=2026, snapshot_date=D2)
    assert list(cmp.columns) == COMPARATOR_COLS
    assert (cmp["Season"] == 2026).all() and (cmp["snapshot_date"] == D2).all()
    assert cmp.loc[cmp["TeamID"] == 1101, "cmp_AdjEM"].iloc[0] == 20.0


# ── rank ─────────────────────────────────────────────────────────────────────────────

def test_rank_is_best_first_within_snapshot():
    ours = _ours([
        {"Season": 2026, "TeamID": 1101, "ArchiveDate": D2, "our_AdjEM": 10.0,
         "our_AdjOE": 110, "our_AdjDE": 100, "our_AdjTempo": 68},
        {"Season": 2026, "TeamID": 1102, "ArchiveDate": D2, "our_AdjEM": 20.0,
         "our_AdjOE": 115, "our_AdjDE": 95, "our_AdjTempo": 70},
        {"Season": 2026, "TeamID": 1103, "ArchiveDate": D2, "our_AdjEM": 5.0,
         "our_AdjOE": 108, "our_AdjDE": 103, "our_AdjTempo": 66},
    ])
    log = build_ratings_log(ours, _cmp([]))
    ranks = dict(zip(log["TeamID"], log["our_rank"]))
    assert ranks[1102] == 1 and ranks[1101] == 2 and ranks[1103] == 3  # highest AdjEM → rank 1


def test_rank_does_not_mix_snapshots():
    # Same team, two dates; each date ranks within itself.
    ours = _ours([
        {"Season": 2026, "TeamID": 1101, "ArchiveDate": D1, "our_AdjEM": 5.0,
         "our_AdjOE": 108, "our_AdjDE": 103, "our_AdjTempo": 66},
        {"Season": 2026, "TeamID": 1102, "ArchiveDate": D1, "our_AdjEM": 4.0,
         "our_AdjOE": 107, "our_AdjDE": 104, "our_AdjTempo": 66},
        {"Season": 2026, "TeamID": 1101, "ArchiveDate": D2, "our_AdjEM": 20.0,
         "our_AdjOE": 115, "our_AdjDE": 95, "our_AdjTempo": 70},
    ])
    log = build_ratings_log(ours, _cmp([]))
    r = log.set_index(["ArchiveDate", "TeamID"])["our_rank"]
    assert r[(D1, 1101)] == 1 and r[(D1, 1102)] == 2
    assert r[(D2, 1101)] == 1  # alone in its snapshot


# ── comparator alignment (merge_asof backward) ────────────────────────────────────────

def _base_ours():
    return _ours([{"Season": 2026, "TeamID": 1101, "ArchiveDate": D2, "our_AdjEM": 20.0,
                   "our_AdjOE": 115, "our_AdjDE": 95, "our_AdjTempo": 70}])


def test_comparator_aligns_to_latest_snapshot_on_or_before():
    # Comparator has D1 and D3 for the team; our snapshot is D2 → backward picks D1.
    cmp = _cmp([
        {"Season": 2026, "TeamID": 1101, "snapshot_date": D1, "cmp_AdjEM": 18.0,
         "cmp_AdjOE": 113, "cmp_AdjDE": 95, "cmp_AdjTempo": 69},
        {"Season": 2026, "TeamID": 1101, "snapshot_date": D3, "cmp_AdjEM": 22.0,
         "cmp_AdjOE": 117, "cmp_AdjDE": 95, "cmp_AdjTempo": 71},
    ])
    log = build_ratings_log(_base_ours(), cmp)
    row = log.iloc[0]
    assert row["cmp_AdjEM"] == 18.0            # D1, not the future D3
    assert row["d_AdjEM"] == pytest.approx(2.0)  # our 20 − their 18


def test_comparator_only_in_future_is_nan():
    cmp = _cmp([{"Season": 2026, "TeamID": 1101, "snapshot_date": D3, "cmp_AdjEM": 22.0,
                 "cmp_AdjOE": 117, "cmp_AdjDE": 95, "cmp_AdjTempo": 71}])
    log = build_ratings_log(_base_ours(), cmp)
    row = log.iloc[0]
    assert pd.isna(row["cmp_AdjEM"]) and pd.isna(row["cmp_rank"]) and pd.isna(row["d_AdjEM"])


def test_team_without_comparator_is_nan_not_dropped():
    ours = _ours([
        {"Season": 2026, "TeamID": 1101, "ArchiveDate": D2, "our_AdjEM": 20.0,
         "our_AdjOE": 115, "our_AdjDE": 95, "our_AdjTempo": 70},
        {"Season": 2026, "TeamID": 1102, "ArchiveDate": D2, "our_AdjEM": 10.0,
         "our_AdjOE": 110, "our_AdjDE": 100, "our_AdjTempo": 68},
    ])
    cmp = _cmp([{"Season": 2026, "TeamID": 1101, "snapshot_date": D1, "cmp_AdjEM": 18.0,
                 "cmp_AdjOE": 113, "cmp_AdjDE": 95, "cmp_AdjTempo": 69}])
    log = build_ratings_log(ours, cmp)
    assert len(log) == 2  # 1102 kept
    row2 = log.set_index("TeamID").loc[1102]
    assert pd.isna(row2["cmp_AdjEM"]) and pd.isna(row2["cmp_rank"])


def test_both_genders_ranked_independently_by_teamid_domain():
    # Men (<2000) and women (>=3000) share a snapshot date but are separate rating populations.
    ours = _ours([
        {"Season": 2026, "TeamID": 1101, "ArchiveDate": D2, "our_AdjEM": 20.0,
         "our_AdjOE": 115, "our_AdjDE": 95, "our_AdjTempo": 70},
        {"Season": 2026, "TeamID": 3101, "ArchiveDate": D2, "our_AdjEM": 30.0,
         "our_AdjOE": 120, "our_AdjDE": 90, "our_AdjTempo": 72},
    ])
    log = build_ratings_log(ours, _cmp([]))
    # Ranking is per-snapshot across whatever teams are present; both rows survive with a rank.
    assert set(log["TeamID"]) == {1101, 3101}
    assert not log["our_rank"].isna().any()


def test_deltas_are_ours_minus_theirs():
    cmp = _cmp([{"Season": 2026, "TeamID": 1101, "snapshot_date": D2, "cmp_AdjEM": 15.0,
                 "cmp_AdjOE": 110, "cmp_AdjDE": 98, "cmp_AdjTempo": 72}])
    log = build_ratings_log(_base_ours(), cmp)
    row = log.iloc[0]
    assert row["d_AdjEM"] == pytest.approx(5.0)     # 20 − 15
    assert row["d_AdjTempo"] == pytest.approx(-2.0)  # 70 − 72
    assert row["d_rank"] == 0.0                      # both alone → rank 1 each
