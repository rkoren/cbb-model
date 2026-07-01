"""Tests for as-of-date KenPom features (GM-002).

The load-bearing property is **leak-freeness**: a game's as-of rating must come from a snapshot
taken *strictly before* the game (KenPom snapshots are end-of-day, so a same-day snapshot already
contains the game's own result). The rest checks the merge-asof picks the latest prior snapshot,
men's/early-season gaps become NaN, and the A/B differentials + sums are formed correctly.
"""

import numpy as np
import pandas as pd
import pytest

from cbb.kenpom.asof_features import (
    add_asof_features,
    game_dates,
    load_archive_snapshots,
)

DZ = {2025: pd.Timestamp("2024-12-25")}  # DayNum 7→Jan 1, 14→Jan 8, 15→Jan 9


def _snaps():
    rows = []
    for tid, em1, em2 in [(1101, 10.0, 20.0), (1102, 5.0, 8.0)]:
        rows += [
            {"Season": 2025, "TeamID": tid, "ArchiveDate": pd.Timestamp("2025-01-01"),
             "kp_AdjEM_asof": em1, "kp_AdjOE_asof": 110.0, "kp_AdjDE_asof": 100.0, "kp_AdjTempo_asof": 68.0},
            {"Season": 2025, "TeamID": tid, "ArchiveDate": pd.Timestamp("2025-01-08"),
             "kp_AdjEM_asof": em2, "kp_AdjOE_asof": 115.0, "kp_AdjDE_asof": 95.0, "kp_AdjTempo_asof": 67.0},
        ]
    return pd.DataFrame(rows)


def _games(daynums, a=1101, b=1102):
    return pd.DataFrame({
        "Season": 2025, "DayNum": daynums,
        "A_TeamID": a, "B_TeamID": b,
    })


# ── game_dates ──────────────────────────────────────────────────────────────────

def test_game_dates_from_dayzero():
    g = _games([7, 14])
    dts = game_dates(g, DZ)
    assert list(dts) == [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-08")]


# ── leak-free boundary (the critical one) ───────────────────────────────────────

def test_same_day_snapshot_excluded():
    # Game on 2025-01-08 (a snapshot date) must take the 2025-01-01 snapshot (AdjEM 10),
    # NOT the same-day 2025-01-08 one (AdjEM 20) — which would include the game's own result.
    out, _ = add_asof_features(_games([14]), _snaps(), DZ)  # DayNum 14 = 2025-01-08
    assert out["A_kp_AdjEM_asof"].iloc[0] == pytest.approx(10.0)


def test_takes_latest_prior_snapshot():
    out, _ = add_asof_features(_games([15]), _snaps(), DZ)  # DayNum 15 = 2025-01-09
    assert out["A_kp_AdjEM_asof"].iloc[0] == pytest.approx(20.0)  # the 01-08 snapshot now qualifies


def test_before_first_snapshot_is_nan():
    out, _ = add_asof_features(_games([6]), _snaps(), DZ)  # DayNum 6 = 2024-12-31, before any snapshot
    assert pd.isna(out["A_kp_AdjEM_asof"].iloc[0])


# ── differentials / sums / coverage ─────────────────────────────────────────────

def test_differential_and_sum():
    out, added = add_asof_features(_games([15]), _snaps(), DZ)
    row = out.iloc[0]
    assert row["d_kp_AdjEM_asof"] == pytest.approx(20.0 - 8.0)        # A 1101=20, B 1102=8
    assert row["s_kp_AdjTempo_asof"] == pytest.approx(67.0 + 67.0)   # both 01-08 snapshots → 67
    assert "d_kp_AdjEM_asof" in added and "s_kp_AdjTempo_asof" in added


def test_unknown_team_propagates_nan():
    # Women / non-archive team (3101) → no snapshot → NaN. The SUM must also be NaN, not a
    # half-sum (real + 0) — a partial sum would mislead the total head; the trainer zero-fills
    # NaN uniformly instead (the consistent "missing" signal).
    out, _ = add_asof_features(_games([15], a=3101), _snaps(), DZ)
    row = out.iloc[0]
    assert pd.isna(row["A_kp_AdjEM_asof"]) and pd.isna(row["d_kp_AdjEM_asof"])
    assert pd.isna(row["s_kp_AdjEM_asof"])  # one side NaN → sum NaN (no partial half-sum)


def test_noop_when_snapshots_empty():
    g = _games([15])
    out, added = add_asof_features(g, pd.DataFrame(), DZ)
    assert added == [] and "A_kp_AdjEM_asof" not in out.columns


def test_row_order_preserved():
    # Multiple games out of date order → as-of values must map back to the right rows.
    out, _ = add_asof_features(_games([15, 6, 14]), _snaps(), DZ)
    assert out["A_kp_AdjEM_asof"].iloc[0] == pytest.approx(20.0)   # DayNum 15
    assert pd.isna(out["A_kp_AdjEM_asof"].iloc[1])                 # DayNum 6
    assert out["A_kp_AdjEM_asof"].iloc[2] == pytest.approx(10.0)   # DayNum 14 (same-day excluded)


# ── load_archive_snapshots ──────────────────────────────────────────────────────

def test_load_maps_names_and_parses_dates(tmp_path):
    raw = pd.DataFrame({
        "ArchiveDate": ["2025-01-01", "2025-01-08"],
        "TeamName": ["Duke", "Duke"],
        "AdjEM": [30.0, 32.0], "AdjOE": [120.0, 121.0], "AdjDE": [90.0, 89.0], "AdjTempo": [66.0, 67.0],
    })
    raw.to_parquet(tmp_path / "kenpom_archive_2025.parquet", index=False)
    out = load_archive_snapshots(2025, tmp_path, {"Duke": 1181})
    assert list(out["TeamID"].unique()) == [1181]
    assert out["ArchiveDate"].dtype == np.dtype("datetime64[ns]")
    assert "kp_AdjEM_asof" in out.columns and len(out) == 2


def test_load_absent_file_returns_empty(tmp_path):
    out = load_archive_snapshots(2025, tmp_path, {"Duke": 1181})
    assert len(out) == 0
