"""Tests for cbb.kenpom.features."""

import logging
from pathlib import Path

import pandas as pd
import pytest

from cbb.kenpom.features import (
    build_team_name_map,
    load_kenpom_efficiency,
    load_selection_sunday_efficiency,
    merge_kenpom_efficiency,
)


def _kaggle_teams(*names) -> pd.DataFrame:
    return pd.DataFrame({"TeamID": range(1001, 1001 + len(names)), "TeamName": list(names)})


def _kenpom_teams(*names) -> pd.DataFrame:
    return pd.DataFrame({"TeamName": list(names)})


# ── build_team_name_map ───────────────────────────────────────────────────────

def test_exact_match():
    kaggle = _kaggle_teams("Duke", "Kansas")
    kenpom = _kenpom_teams("Duke", "Kansas")
    result = build_team_name_map(kaggle, kenpom)
    assert result["Duke"] == 1001
    assert result["Kansas"] == 1002


def test_fuzzy_match():
    # "Kennesaw State" vs "Kennesaw St" — high sequence ratio, passes 0.85 threshold
    kaggle = _kaggle_teams("Kennesaw State")
    kenpom = _kenpom_teams("Kennesaw St")
    result = build_team_name_map(kaggle, kenpom)
    assert "Kennesaw St" in result
    assert result["Kennesaw St"] == 1001


def test_override_applied():
    kaggle = _kaggle_teams("NC State")
    kenpom = _kenpom_teams("N.C. State")
    result = build_team_name_map(kaggle, kenpom)
    assert result["N.C. State"] == 1001


def test_unmatched_logged_as_count_not_full_list(caplog):
    kaggle = _kaggle_teams("Duke")
    kenpom = _kenpom_teams("NoSuchTeamXYZ")
    with caplog.at_level(logging.DEBUG, logger="cbb.kenpom.features"):
        result = build_team_name_map(kaggle, kenpom)
    assert "NoSuchTeamXYZ" not in result
    # Count at INFO (no full list spam); names only at DEBUG.
    info = [r for r in caplog.records if r.levelno == logging.INFO]
    assert any("1 unmatched" in r.getMessage() for r in info)
    assert any("NoSuchTeamXYZ" in r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG)
    # The unmatched name must NOT appear at WARNING (that was the spam we removed).
    assert not any("NoSuchTeamXYZ" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)


def test_unmatched_team_excluded_from_map():
    kaggle = _kaggle_teams("Duke")
    kenpom = _kenpom_teams("Duke", "NoSuchTeamXYZ")
    result = build_team_name_map(kaggle, kenpom)
    assert len(result) == 1
    assert "NoSuchTeamXYZ" not in result


def _spellings(*pairs) -> pd.DataFrame:
    return pd.DataFrame(
        {"TeamNameSpelling": [p[0] for p in pairs], "TeamID": [p[1] for p in pairs]}
    )


def test_spellings_recovers_abbreviated_mid_major():
    # Kaggle abbreviates "Abilene Chr" — too short for the 0.85 fuzzy. The spellings table
    # (Kaggle's own variant list) bridges it where exact + fuzzy both fail.
    kaggle = _kaggle_teams("Abilene Chr")  # TeamID 1001
    kenpom = _kenpom_teams("Abilene Christian")
    assert "Abilene Christian" not in build_team_name_map(kaggle, kenpom)  # fuzzy-only misses it
    spell = _spellings(("Abilene Christian", 1001))
    assert build_team_name_map(kaggle, kenpom, spell)["Abilene Christian"] == 1001


def test_spellings_normalization_ignores_punctuation():
    # "Cal St. Bakersfield" vs a spelling "cal st bakersfield" — _norm_spelling strips the period.
    kaggle = _kaggle_teams("CS Bakersfield")  # 1001
    kenpom = _kenpom_teams("Cal St. Bakersfield")
    spell = _spellings(("cal st bakersfield", 1001))
    assert build_team_name_map(kaggle, kenpom, spell)["Cal St. Bakersfield"] == 1001


def test_spellings_precedence_exact_beats_spellings():
    # An exact Kaggle name wins before the spellings table is consulted.
    kaggle = _kaggle_teams("Duke")  # 1001
    kenpom = _kenpom_teams("Duke")
    spell = _spellings(("duke", 9999))  # wrong id — must NOT be used since exact matched
    assert build_team_name_map(kaggle, kenpom, spell)["Duke"] == 1001


# ── load_kenpom_efficiency ────────────────────────────────────────────────────

def _write_kenpom_parquet(path: Path, teams: list[str], **cols) -> None:
    data = {"TeamName": teams}
    data.update(cols)
    pd.DataFrame(data).to_parquet(path, index=False)


def test_load_returns_correct_columns(tmp_path):
    p = tmp_path / "kenpom_ratings_2024.parquet"
    _write_kenpom_parquet(p, ["Duke", "Kansas"], AdjOE=[115.0, 118.0], AdjDE=[90.0, 92.0], AdjEM=[25.0, 26.0], AdjTempo=[70.0, 68.0])
    team_map = {"Duke": 1001, "Kansas": 1002}
    result = load_kenpom_efficiency(2024, p, team_map)
    assert set(result.columns) >= {"Season", "men_women", "TeamID", "AdjOE", "AdjDE", "AdjEM", "AdjTempo"}


def test_load_men_women_is_zero(tmp_path):
    p = tmp_path / "kenpom.parquet"
    _write_kenpom_parquet(p, ["Duke"], AdjOE=[115.0], AdjDE=[90.0], AdjEM=[25.0], AdjTempo=[70.0])
    result = load_kenpom_efficiency(2024, p, {"Duke": 1001})
    assert (result["men_women"] == 0).all()


def test_load_sets_season(tmp_path):
    p = tmp_path / "kenpom.parquet"
    _write_kenpom_parquet(p, ["Duke"], AdjOE=[115.0], AdjDE=[90.0], AdjEM=[25.0], AdjTempo=[70.0])
    result = load_kenpom_efficiency(2025, p, {"Duke": 1001})
    assert (result["Season"] == 2025).all()


def test_load_excludes_unmatched_teams(tmp_path):
    p = tmp_path / "kenpom.parquet"
    _write_kenpom_parquet(p, ["Duke", "NoSuchTeam"], AdjOE=[115.0, 100.0], AdjDE=[90.0, 95.0], AdjEM=[25.0, 5.0], AdjTempo=[70.0, 65.0])
    result = load_kenpom_efficiency(2024, p, {"Duke": 1001})
    assert len(result) == 1
    assert result.iloc[0]["TeamID"] == 1001


def test_load_raises_on_missing_efficiency_cols(tmp_path):
    p = tmp_path / "kenpom.parquet"
    pd.DataFrame({"TeamName": ["Duke"], "SomeOtherCol": [1.0]}).to_parquet(p)
    with pytest.raises(ValueError, match="efficiency columns"):
        load_kenpom_efficiency(2024, p, {"Duke": 1001})


# ── load_selection_sunday_efficiency (KP-005 leak fix) ──────────────────────────

def test_selection_sunday_takes_latest_snapshot(tmp_path):
    # Archive with a preseason row (earliest) + two weekly snapshots; must take the LATEST
    # (Selection-Sunday) AdjEM — never the preseason or an early-season value.
    p = tmp_path / "kenpom_archive_2025.parquet"
    pd.DataFrame({
        "ArchiveDate": ["2024-11-02", "2025-01-15", "2025-03-10"],  # preseason, mid, latest
        "TeamName": ["Duke", "Duke", "Duke"],
        "AdjOE": [110.0, 118.0, 122.0], "AdjDE": [95.0, 90.0, 88.0],
        "AdjEM": [15.0, 28.0, 34.0], "AdjTempo": [68.0, 67.0, 67.0],
    }).to_parquet(p, index=False)
    out = load_selection_sunday_efficiency(2025, p, {"Duke": 1001})
    assert len(out) == 1
    assert out.iloc[0]["AdjEM"] == 34.0  # the 2025-03-10 snapshot, not preseason 15 or mid 28


def test_selection_sunday_absent_archive_returns_empty(tmp_path):
    out = load_selection_sunday_efficiency(2011, tmp_path / "kenpom_archive_2011.parquet", {"Duke": 1001})
    assert len(out) == 0  # pre-2012: no archive → caller keeps manual (leak-free) efficiency


# ── merge_kenpom_efficiency ───────────────────────────────────────────────────

def _adj_eff_df(men_women: list[int], team_ids: list[int], season: int = 2024) -> pd.DataFrame:
    n = len(team_ids)
    return pd.DataFrame({
        "Season": season,
        "men_women": men_women,
        "TeamID": team_ids,
        "AdjOE": [100.0] * n,
        "AdjDE": [100.0] * n,
        "AdjEM": [0.0] * n,
        "AdjTempo": [65.0] * n,
        "iters": [30] * n,
    })


def _kenpom_eff_df(team_ids: list[int], season: int = 2024) -> pd.DataFrame:
    n = len(team_ids)
    return pd.DataFrame({
        "Season": season,
        "men_women": 0,
        "TeamID": team_ids,
        "AdjOE": [120.0] * n,
        "AdjDE": [85.0] * n,
        "AdjEM": [35.0] * n,
        "AdjTempo": [72.0] * n,
    })


def test_merge_replaces_mens_efficiency():
    adj_eff = _adj_eff_df(men_women=[0], team_ids=[1001])
    kenpom = _kenpom_eff_df(team_ids=[1001])
    result = merge_kenpom_efficiency(adj_eff, kenpom)
    assert result.loc[result["TeamID"] == 1001, "AdjOE"].iloc[0] == pytest.approx(120.0)


def test_merge_preserves_womens_rows():
    adj_eff = _adj_eff_df(men_women=[0, 1], team_ids=[1001, 2001])
    kenpom = _kenpom_eff_df(team_ids=[1001])
    result = merge_kenpom_efficiency(adj_eff, kenpom)
    womens_row = result[result["men_women"] == 1]
    assert womens_row["AdjOE"].iloc[0] == pytest.approx(100.0)


def test_merge_preserves_unmatched_mens_rows():
    adj_eff = _adj_eff_df(men_women=[0, 0], team_ids=[1001, 1002])
    kenpom = _kenpom_eff_df(team_ids=[1001])
    result = merge_kenpom_efficiency(adj_eff, kenpom)
    unmatched = result[result["TeamID"] == 1002]
    assert unmatched["AdjOE"].iloc[0] == pytest.approx(100.0)


def test_merge_preserves_iters_column():
    adj_eff = _adj_eff_df(men_women=[0], team_ids=[1001])
    kenpom = _kenpom_eff_df(team_ids=[1001])
    result = merge_kenpom_efficiency(adj_eff, kenpom)
    assert "iters" in result.columns


def test_merge_no_kenpom_data_returns_original():
    adj_eff = _adj_eff_df(men_women=[0], team_ids=[1001])
    kenpom = _kenpom_eff_df(team_ids=[9999])
    result = merge_kenpom_efficiency(adj_eff, kenpom)
    assert result.loc[result["TeamID"] == 1001, "AdjOE"].iloc[0] == pytest.approx(100.0)
