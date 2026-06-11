"""Tests for cbb.kenpom.rich_features (offline — no API calls)."""

import logging

import pandas as pd

from cbb.kenpom.rich_features import (
    KP_RICH_FEATURES,
    build_kenpom_rich_features,
    join_kenpom_rich,
)


def _ratings(*names) -> pd.DataFrame:
    n = len(names)
    return pd.DataFrame({
        "TeamName": list(names),
        "AdjEM": [10.0] * n, "AdjOE": [110.0] * n, "AdjDE": [100.0] * n,
        "SOS": [5.0 + i for i in range(n)],
        "Luck": [0.01 * i for i in range(n)],
        "APL_Off": [17.0] * n, "APL_Def": [18.0] * n,
    })


def _height(*names) -> pd.DataFrame:
    n = len(names)
    return pd.DataFrame({
        "TeamName": list(names),
        "AvgHgt": [77.0 + i for i in range(n)], "HgtEff": [78.0] * n,
        "Exp": [2.0] * n, "Bench": [30.0] * n, "Continuity": [0.5] * n,
    })


# ── build_kenpom_rich_features ────────────────────────────────────────────────

def test_builds_both_groups_keyed_to_teamid():
    team_map = {"Duke": 1001, "Kansas": 1002}
    result = build_kenpom_rich_features(2024, team_map, _ratings("Duke", "Kansas"), _height("Duke", "Kansas"))
    assert set(result.columns) >= {"Season", "men_women", "TeamID", *KP_RICH_FEATURES}
    assert (result["Season"] == 2024).all()
    assert (result["men_women"] == 0).all()
    assert sorted(result["TeamID"]) == [1001, 1002]


def test_ratings_only_when_no_height():
    result = build_kenpom_rich_features(2024, {"Duke": 1001}, _ratings("Duke"), None)
    assert "kp_SOS" in result.columns
    assert "kp_AvgHgt" not in result.columns


def test_height_only_when_no_ratings():
    result = build_kenpom_rich_features(2024, {"Duke": 1001}, None, _height("Duke"))
    assert "kp_AvgHgt" in result.columns
    assert "kp_SOS" not in result.columns


def test_empty_when_no_sources():
    result = build_kenpom_rich_features(2024, {"Duke": 1001}, None, None)
    assert list(result.columns) == ["Season", "men_women", "TeamID"]
    assert len(result) == 0


def test_unmatched_team_excluded():
    # "Nowhere" is not in the name map → dropped
    result = build_kenpom_rich_features(2024, {"Duke": 1001}, _ratings("Duke", "Nowhere"), None)
    assert sorted(result["TeamID"]) == [1001]


def test_missing_source_column_logged_and_skipped(caplog):
    ratings = _ratings("Duke").drop(columns=["Luck"])
    with caplog.at_level(logging.WARNING, logger="cbb.kenpom.rich_features"):
        result = build_kenpom_rich_features(2024, {"Duke": 1001}, ratings, None)
    assert "kp_Luck" not in result.columns
    assert "kp_SOS" in result.columns
    assert "Luck" in caplog.text


# ── join_kenpom_rich ──────────────────────────────────────────────────────────

def _matchups() -> pd.DataFrame:
    return pd.DataFrame({
        "Season": [2024], "men_women": [0], "A_TeamID": [1001], "B_TeamID": [1002],
    })


def test_join_builds_differentials():
    kp = build_kenpom_rich_features(2024, {"Duke": 1001, "Kansas": 1002}, _ratings("Duke", "Kansas"), _height("Duke", "Kansas"))
    joined, added = join_kenpom_rich(_matchups(), kp)
    # SOS: Duke=5.0, Kansas=6.0 → d = -1.0
    assert joined["d_kp_SOS"].iloc[0] == -1.0
    assert "A_kp_SOS" in added and "B_kp_SOS" in added and "d_kp_SOS" in added


def test_join_missing_team_leaves_nan():
    # B_TeamID 1002 absent from kp_feats → B_/d_ columns are NaN, not an error
    kp = build_kenpom_rich_features(2024, {"Duke": 1001}, _ratings("Duke"), None)
    joined, _ = join_kenpom_rich(_matchups(), kp)
    assert pd.isna(joined["B_kp_SOS"].iloc[0])
    assert pd.isna(joined["d_kp_SOS"].iloc[0])
    assert joined["A_kp_SOS"].iloc[0] == 5.0


def test_join_empty_features_is_noop():
    empty = build_kenpom_rich_features(2024, {"Duke": 1001}, None, None)
    joined, added = join_kenpom_rich(_matchups(), empty)
    assert added == []
    assert list(joined.columns) == ["Season", "men_women", "A_TeamID", "B_TeamID"]
