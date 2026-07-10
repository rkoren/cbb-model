"""Tests for the dashboard payload builder (DASH-007).

Load-bearing: TeamIDs resolve to names, the slate groups by game date and surfaces the us−KenPom
gap, unmatched games carry a null KenPom side (not a crash), and everything is JSON-clean (no NaN).
"""

import json
import math

import pandas as pd
import pytest

from cbb.dashboard.payload import build_payload, build_slate

NAMES = {1101: "Duke", 1102: "Kansas", 1103: "Iowa"}


def _pred(rows):
    return pd.DataFrame(rows)


def _slate_row(**kw):
    base = dict(game_date="2026-03-01", A_TeamID=1101, B_TeamID=1102, men_women=0,
                pred_margin=6.0, pred_total=148.0, pred_prob=0.70,
                Margin=4.0, Total=150.0, Outcome=1,
                cmp_margin=3.0, cmp_total=151.0, cmp_prob=0.62)
    base.update(kw)
    return base


def test_slate_groups_by_date_and_maps_names():
    log = _pred([_slate_row(), _slate_row(game_date="2026-03-02", A_TeamID=1103, B_TeamID=1101)])
    slate = build_slate(log, NAMES)
    assert set(slate) == {"2026-03-01", "2026-03-02"}
    g = slate["2026-03-01"][0]
    assert g["a"] == "Duke" and g["b"] == "Kansas"
    assert g["our"]["margin"] == 6.0 and g["kp"]["margin"] == 3.0
    assert g["actual"]["won"] == 1


def test_slate_reconstructs_team_scores_from_total_and_margin():
    g = build_slate(_pred([_slate_row()]), NAMES)["2026-03-01"][0]
    assert (g["our"]["a"], g["our"]["b"]) == (77, 71)        # (148 ± 6) / 2
    assert (g["kp"]["a"], g["kp"]["b"]) == (77, 74)          # (151 ± 3) / 2
    assert (g["actual"]["a"], g["actual"]["b"]) == (77, 73)  # (150 ± 4) / 2
    assert g["our"]["prob"] == 0.70


def test_slate_gap_is_us_minus_kenpom_and_sorts_by_magnitude():
    log = _pred([
        _slate_row(A_TeamID=1101, B_TeamID=1102, pred_margin=6.0, cmp_margin=3.0),   # gap +3
        _slate_row(A_TeamID=1103, B_TeamID=1101, pred_margin=1.0, cmp_margin=-9.0),  # gap +10
    ])
    games = build_slate(log, NAMES)["2026-03-01"]
    assert games[0]["gap_margin"] == 10.0   # biggest disagreement first
    assert games[1]["gap_margin"] == 3.0


def test_unmatched_game_has_null_kenpom_side():
    log = _pred([_slate_row(cmp_margin=float("nan"), cmp_total=float("nan"), cmp_prob=float("nan"))])
    g = build_slate(log, NAMES)["2026-03-01"][0]
    assert g["kp"]["margin"] is None and g["gap_margin"] is None
    assert g["our"]["margin"] == 6.0        # our side still populated


def test_slate_without_cmp_columns_sets_kp_none():
    log = _pred([{k: v for k, v in _slate_row().items() if not k.startswith("cmp_")}])
    g = build_slate(log, NAMES)["2026-03-01"][0]
    assert g["kp"] is None


def test_slate_raises_on_missing_required_column():
    log = _pred([{"A_TeamID": 1101}])
    with pytest.raises(ValueError, match="missing columns"):
        build_slate(log, NAMES)


def test_gender_flag_maps_from_men_women():
    log = _pred([_slate_row(men_women=1)])
    assert build_slate(log, NAMES)["2026-03-01"][0]["gender"] == "W"


def _ratings(rows):
    return pd.DataFrame(rows)


def _rating_row(**kw):
    base = dict(Season=2026, ArchiveDate=pd.Timestamp("2026-03-01"), TeamID=1101,
                our_AdjEM=20.0, our_AdjOE=115.0, our_AdjDE=95.0, our_AdjTempo=68.0,
                cmp_AdjEM=18.0, cmp_AdjOE=113.0, cmp_AdjDE=95.0, cmp_AdjTempo=69.0,
                our_rank=1.0, cmp_rank=2.0, d_AdjEM=2.0, d_rank=-1.0)
    base.update(kw)
    return base


def test_payload_shape_and_json_serializable():
    preds = _pred([_slate_row(), _slate_row(game_date="2026-02-20")])
    rates = _ratings([_rating_row(), _rating_row(ArchiveDate=pd.Timestamp("2026-02-15"), TeamID=1102,
                                                 our_rank=2.0)])
    payload = build_payload(preds, rates, NAMES, generated="2026-07-09")
    assert payload["slate_dates"] == ["2026-02-20", "2026-03-01"]     # sorted
    assert payload["rating_dates"] == ["2026-02-15", "2026-03-01"]
    assert payload["meta"] == {"generated": "2026-07-09", "n_games": 2, "n_snapshots": 2}
    # No NaN anywhere → strict JSON round-trips.
    dumped = json.dumps(payload, allow_nan=False)
    assert "NaN" not in dumped
    assert not any(isinstance(v, float) and math.isnan(v)
                   for team in payload["ratings"]["2026-03-01"]
                   for v in [team["d_em"], team["our"]["em"]])


def test_ratings_team_carries_gender():
    rates = _ratings([_rating_row(TeamID=1101), _rating_row(TeamID=3101, our_rank=1.0)])
    teams = build_payload(_pred([_slate_row()]), rates, {**NAMES, 3101: "UConn"})["ratings"]["2026-03-01"]
    by_id = {t["team"]: t["gender"] for t in teams}
    assert by_id["Duke"] == "M" and by_id["UConn"] == "W"   # TeamID < 2000 → M, >= 2000 → W


def test_ratings_sorted_by_our_rank():
    rates = _ratings([
        _rating_row(TeamID=1102, our_rank=3.0),
        _rating_row(TeamID=1101, our_rank=1.0),
    ])
    teams = build_payload(_pred([_slate_row()]), rates, NAMES)["ratings"]["2026-03-01"]
    assert [t["our"]["rank"] for t in teams] == [1, 3]
