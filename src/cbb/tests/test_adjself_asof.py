"""Tests for self-computed as-of efficiency snapshots (WM-002).

The two things that must hold: (1) **leak-freeness** — a snapshot stamped at cutoff ``cut`` reflects
only games with ``DayNum < cut`` (a team that debuts later cannot appear in an earlier snapshot);
(2) **shape** — the frame matches the KenPom/Torvik as-of contract (``Season, TeamID, ArchiveDate,
adjself_*_asof``, both genders, no ``men_women`` column) so ``add_asof_features`` consumes it.
"""

import pandas as pd

from cbb.features.adjself_asof import ADJSELF_FEATURES, compute_adjself_asof_snapshots


def _sym_game(season, mw, daynum, a, b, sa, sb, home):
    """Two symmetric rows (T1 = each team) for one game, with valid box-score columns."""
    box = dict(FGA=55, OR=10, TO=12, FTA=20)
    rows = []
    for (t1, t2, s1, s2, h) in ((a, b, sa, sb, home), (b, a, sb, sa, -home)):
        row = {"Season": season, "men_women": mw, "DayNum": daynum,
               "T1_TeamID": t1, "T2_TeamID": t2, "T1_Score": s1, "T2_Score": s2, "T1_home": h}
        for k, v in box.items():
            row[f"T1_{k}"] = v
            row[f"T2_{k}"] = v
        rows.append(row)
    return rows


def _reg_sym():
    # Season 2020: men 101v102 on day5, 101v103 on day25 (103 debuts day25); women 3101v3102 on day5.
    rows = []
    rows += _sym_game(2020, 0, 5, 101, 102, 80, 60, 1)
    rows += _sym_game(2020, 0, 25, 101, 103, 75, 70, 0)
    rows += _sym_game(2020, 1, 5, 3101, 3102, 70, 68, 1)
    return pd.DataFrame(rows)


_DAYZERO = {2020: pd.Timestamp("2019-11-01")}


def _snaps():
    # cutoffs 14, 21, 28: day5 games feed cut=14/21; day25 game only reaches cut=28.
    return compute_adjself_asof_snapshots(_reg_sym(), _DAYZERO,
                                          first_daynum=14, last_daynum=30, step=7)


def test_shape_matches_asof_contract():
    s = _snaps()
    assert list(s.columns) == ["Season", "TeamID", "ArchiveDate", *ADJSELF_FEATURES]
    assert "men_women" not in s.columns
    assert pd.api.types.is_datetime64_any_dtype(s["ArchiveDate"])


def test_archive_date_is_dayzero_plus_cutoff():
    s = _snaps()
    # earliest snapshot is the cut=14 stamp: DayZero(2019-11-01) + 14 days.
    assert s["ArchiveDate"].min() == pd.Timestamp("2019-11-15")
    assert set(s["ArchiveDate"].dt.day) <= {15, 22, 29}  # 14/21/28 days past Nov 1


def test_leak_free_debut_absent_from_earlier_snapshot():
    s = _snaps()
    early = s[s["ArchiveDate"] == pd.Timestamp("2019-11-15")]  # cut=14 → only DayNum<14 (day5)
    assert 103 not in set(early["TeamID"])          # 103 debuts day25 → not yet rated
    assert {101, 102} <= set(early["TeamID"])        # day5 participants are rated
    late = s[s["ArchiveDate"] == pd.Timestamp("2019-11-29")]  # cut=28 → includes day25
    assert 103 in set(late["TeamID"])


def test_both_genders_present():
    s = _snaps()
    assert (s["TeamID"] < 2000).any()   # men
    assert (s["TeamID"] >= 3000).any()  # women


def test_empty_without_dayzero():
    s = compute_adjself_asof_snapshots(_reg_sym(), {})
    assert s.empty and list(s.columns) == ["Season", "TeamID", "ArchiveDate", *ADJSELF_FEATURES]
