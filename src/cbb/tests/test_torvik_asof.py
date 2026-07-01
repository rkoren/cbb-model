"""Tests for the BartTorvik women's as-of loader (WM-003)."""

import numpy as np
import pandas as pd

from cbb.features.torvik_asof import TV_FEATURES, load_torvik_women_snapshots


def test_load_maps_derives_adjem_and_parses_dates(tmp_path):
    raw = pd.DataFrame({
        "ArchiveDate": ["2025-11-10", "2026-01-15"],
        "team": ["Connecticut", "Connecticut"],
        "tv_AdjOE": [115.0, 128.0], "tv_AdjDE": [70.0, 65.0], "tv_AdjTempo": [66.0, 67.0],
    })
    raw.to_parquet(tmp_path / "torvik_women_2026.parquet", index=False)
    out = load_torvik_women_snapshots(2026, tmp_path, {"Connecticut": 3163})
    assert list(out["TeamID"].unique()) == [3163]
    assert out["ArchiveDate"].dtype == np.dtype("datetime64[ns]")
    assert set(TV_FEATURES).issubset(out.columns)
    # AdjEM derived = AdjOE − AdjDE
    assert out.iloc[1]["tv_AdjEM_asof"] == 128.0 - 65.0
    assert out.iloc[1]["tv_AdjOE_asof"] == 128.0


def test_load_drops_unmapped_teams(tmp_path):
    raw = pd.DataFrame({
        "ArchiveDate": ["2026-01-15", "2026-01-15"],
        "team": ["Connecticut", "NobodyU"],
        "tv_AdjOE": [128.0, 100.0], "tv_AdjDE": [65.0, 105.0], "tv_AdjTempo": [67.0, 68.0],
    })
    raw.to_parquet(tmp_path / "torvik_women_2026.parquet", index=False)
    out = load_torvik_women_snapshots(2026, tmp_path, {"Connecticut": 3163})
    assert list(out["TeamID"]) == [3163]  # unmapped NobodyU dropped


def test_load_absent_file_returns_empty(tmp_path):
    assert len(load_torvik_women_snapshots(2026, tmp_path, {"Connecticut": 3163})) == 0
