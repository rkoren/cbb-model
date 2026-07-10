"""Tests for the dashboard wiring helpers (DASH-002)."""

import pandas as pd

from cbb.benchmark.slate import COMPARATOR_COLS
from cbb.dashboard.wiring import (
    add_game_date,
    attach_kenpom_slate,
    build_name_map,
    dayzero_by_gender_season,
    dedupe_symmetric,
)

_MSEASONS = pd.DataFrame({"Season": [2025, 2026], "DayZero": ["11/04/2024", "11/03/2025"]})
_WSEASONS = pd.DataFrame({"Season": [2026], "DayZero": ["11/05/2025"]})


def test_dayzero_keyed_by_gender_and_season():
    dz = dayzero_by_gender_season(_MSEASONS, _WSEASONS)
    assert dz[(2026, 0)] == pd.Timestamp("2025-11-03")   # men
    assert dz[(2026, 1)] == pd.Timestamp("2025-11-05")   # women — its own DayZero
    assert dz[(2025, 0)] == pd.Timestamp("2024-11-04")


def test_add_game_date_offsets_by_daynum_per_gender():
    dz = dayzero_by_gender_season(_MSEASONS, _WSEASONS)
    log = pd.DataFrame({"Season": [2026, 2026], "men_women": [0, 1], "DayNum": [10, 10]})
    out = add_game_date(log, dz)
    # Same DayNum, different gender DayZero → different calendar dates.
    assert out.loc[0, "game_date"] == pd.Timestamp("2025-11-13")   # men: 11-03 + 10
    assert out.loc[1, "game_date"] == pd.Timestamp("2025-11-15")   # women: 11-05 + 10


def test_add_game_date_missing_season_is_nat():
    dz = dayzero_by_gender_season(_MSEASONS, _WSEASONS)
    log = pd.DataFrame({"Season": [1999], "men_women": [0], "DayNum": [10]})
    assert pd.isna(add_game_date(log, dz).loc[0, "game_date"])


def test_add_game_date_defaults_men_when_no_gender_column():
    dz = dayzero_by_gender_season(_MSEASONS, _WSEASONS)
    log = pd.DataFrame({"Season": [2026], "DayNum": [1]})
    assert add_game_date(log, dz).loc[0, "game_date"] == pd.Timestamp("2025-11-04")


def test_dedupe_symmetric_keeps_one_orientation_per_game():
    # Both orientations of one game (winner-as-A and loser-as-A) → keep A_TeamID < B_TeamID.
    log = pd.DataFrame({
        "A_TeamID": [1101, 1102, 1103], "B_TeamID": [1102, 1101, 1104],
        "pred_margin": [6.0, -6.0, 3.0],
    })
    out = dedupe_symmetric(log)
    assert len(out) == 2                                   # the 1101/1102 mirror collapses to one
    assert (out["A_TeamID"] < out["B_TeamID"]).all()
    assert set(zip(out["A_TeamID"], out["B_TeamID"])) == {(1101, 1102), (1103, 1104)}


def test_attach_kenpom_slate_left_joins_cmp_and_keeps_unmatched():
    log = pd.DataFrame({
        "Season": [2026, 2026], "DayNum": [10, 11],
        "A_TeamID": [1101, 1103], "B_TeamID": [1102, 1104],   # 2nd game has no comparator
        "pred_margin": [6.0, 4.0],
    })
    comparator = pd.DataFrame([{
        "Season": 2026, "DayNum": 10, "home_id": 1101, "vis_id": 1102,
        "cmp_margin": 3.0, "cmp_total": 150.0, "cmp_home_wp": 0.7}], columns=COMPARATOR_COLS)
    out = attach_kenpom_slate(log, comparator)
    assert len(out) == 2                                   # no rows dropped
    m = out.set_index("A_TeamID")
    assert m.loc[1101, "cmp_margin"] == 3.0                # matched game gets KenPom
    assert pd.isna(m.loc[1103, "cmp_margin"])              # unmatched stays null


def test_attach_kenpom_slate_noop_on_empty_comparator():
    log = pd.DataFrame({"Season": [2026], "DayNum": [10], "A_TeamID": [1101],
                        "B_TeamID": [1102], "pred_margin": [6.0]})
    out = attach_kenpom_slate(log, pd.DataFrame(columns=COMPARATOR_COLS))
    assert "cmp_margin" not in out.columns and len(out) == 1


def test_build_name_map_merges_frames():
    mteams = pd.DataFrame({"TeamID": [1101, 1102], "TeamName": ["Duke", "Kansas"]})
    wteams = pd.DataFrame({"TeamID": [3101], "TeamName": ["UConn"]})
    m = build_name_map(mteams, wteams)
    assert m == {1101: "Duke", 1102: "Kansas", 3101: "UConn"}


def test_build_name_map_skips_none():
    m = build_name_map(pd.DataFrame({"TeamID": [1101], "TeamName": ["Duke"]}), None)
    assert m == {1101: "Duke"}
