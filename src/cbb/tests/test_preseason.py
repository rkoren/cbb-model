"""Tests for the KenPom preseason prior loader (season-constant, offline)."""

import pandas as pd

from cbb.kenpom.preseason import PRE_FEATURES, ROSTER_FEATURES, load_preseason_priors

_TEAMS = pd.DataFrame({"TeamID": [1101, 1102], "TeamName": ["Duke", "Kansas"]})
_DZ = {2024: pd.Timestamp("2023-11-06")}


def _write(tmp_path, with_height=True, with_preseason=True):
    (tmp_path / "archive").mkdir()
    dates = (["2023-11-05"] if with_preseason else []) + ["2023-11-20"]
    arch = pd.DataFrame([
        {"ArchiveDate": d, "TeamName": t, "AdjEM": em, "AdjOE": 110.0, "AdjDE": 95.0, "AdjTempo": 68.0}
        for d in dates for t, em in (("Duke", 25.0 if d == "2023-11-05" else 30.0), ("Kansas", 20.0))
    ])
    arch.to_parquet(tmp_path / "archive" / "kenpom_archive_2024.parquet", index=False)
    if with_height:
        pd.DataFrame({"TeamName": ["Duke"], "Continuity": [0.4], "Exp": [1.5]}).to_parquet(
            tmp_path / "kenpom_height_2024.parquet", index=False)


def test_lifts_only_the_preseason_row(tmp_path):
    _write(tmp_path)
    out = load_preseason_priors([2024], _TEAMS, None, _DZ, kenpom_dir=tmp_path, include_roster=True)
    assert list(out.columns) == ["Season", "TeamID", *PRE_FEATURES, *ROSTER_FEATURES]
    assert len(out) == 2 and set(out.TeamID) == {1101, 1102}
    # Duke's preseason AdjEM is 25 — the in-season 30 (Nov 20 snapshot) must NOT be used.
    assert out.set_index("TeamID").loc[1101, "kp_pre_AdjEM"] == 25.0
    assert out.set_index("TeamID").loc[1101, "kp_Continuity"] == 0.4
    assert pd.isna(out.set_index("TeamID").loc[1102, "kp_Exp"])  # Kansas absent from height → NaN


def test_missing_preseason_snapshot_skips_season(tmp_path):
    _write(tmp_path, with_preseason=False)
    out = load_preseason_priors([2024], _TEAMS, None, _DZ, kenpom_dir=tmp_path)
    assert out.empty and "kp_pre_AdjEM" in out.columns


def test_missing_archive_and_roster_off_by_default(tmp_path):
    _write(tmp_path, with_height=False)
    out = load_preseason_priors([2024, 2025], _TEAMS, None, {**_DZ, 2025: pd.Timestamp("2024-11-04")},
                                kenpom_dir=tmp_path)
    assert list(out.columns) == ["Season", "TeamID", *PRE_FEATURES] and set(out.Season) == {2024}
