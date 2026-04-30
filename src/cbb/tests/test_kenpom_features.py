"""Tests for cbb.kenpom.features."""

import logging
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from cbb.kenpom.features import (
    build_team_name_map,
    load_kenpom_efficiency,
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


def test_unmatched_logged_as_warning(caplog):
    kaggle = _kaggle_teams("Duke")
    kenpom = _kenpom_teams("NoSuchTeamXYZ")
    with caplog.at_level(logging.WARNING, logger="cbb.kenpom.features"):
        result = build_team_name_map(kaggle, kenpom)
    assert "NoSuchTeamXYZ" not in result
    assert "no Kaggle match" in caplog.text


def test_unmatched_team_excluded_from_map():
    kaggle = _kaggle_teams("Duke")
    kenpom = _kenpom_teams("Duke", "NoSuchTeamXYZ")
    result = build_team_name_map(kaggle, kenpom)
    assert len(result) == 1
    assert "NoSuchTeamXYZ" not in result


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
