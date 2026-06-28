"""Tests for src/evaluate/run.py — the kitchen evaluate adapter (SC-004).

Guards the contract that evaluate emits *honestly named* in-sample metrics
(`insample_brier`) and no longer the leak-aware `loto_brier` name (which the train
stage owns) — so the run's headline `loto_brier` can't be overwritten by the
optimistic resubstitution number.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

# src/evaluate/run.py isn't an importable package path, so load it directly.
_SPEC = importlib.util.spec_from_file_location(
    "evaluate_run", Path(__file__).resolve().parents[3] / "src" / "evaluate" / "run.py"
)
evaluate_run = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluate_run)
evaluate = evaluate_run.evaluate


class _Store:
    """Minimal DataStore stand-in: serves matchups, points processed_dir at tmp."""
    def __init__(self, matchups, processed_dir):
        self._matchups = matchups
        self.processed_dir = processed_dir

    def load_parquet(self, name):
        if name == "matchups.parquet":
            return self._matchups
        raise FileNotFoundError(name)


class _Model:
    features = ["d_AdjEM"]
    def predict_batch(self, df):
        return [0.5] * len(df)


def _matchups():
    return pd.DataFrame({
        "Season": [2024, 2024, 2025, 2025],
        "Outcome": [1.0, 0.0, 1.0, 0.0],
        "d_AdjEM": [5.0, -5.0, 3.0, -3.0],
    })


def test_emits_insample_not_loto(tmp_path):
    out = evaluate(
        _Model(), {"evaluate": {"metrics_file": str(tmp_path / "metrics.json")}},
        _Store(_matchups(), tmp_path),
    )
    assert "insample_brier" in out
    assert "loto_brier" not in out            # train owns that name; evaluate must not collide
    assert out["insample_brier"] == pytest.approx(0.25)  # all preds 0.5 → (0.5)^2


def test_per_season_keys_renamed(tmp_path):
    out = evaluate(
        _Model(), {"evaluate": {"metrics_file": str(tmp_path / "m.json")}},
        _Store(_matchups(), tmp_path),
    )
    assert "insample_brier_2024" in out and "insample_brier_2025" in out
    assert "brier_2024" not in out            # old name would read as a LOTO fold


def test_holdout_skipped_when_no_parquet(tmp_path):
    # No holdout_2026.parquet in processed_dir → no holdout_* metrics, no error.
    out = evaluate(
        _Model(), {"evaluate": {"metrics_file": str(tmp_path / "m.json")}},
        _Store(_matchups(), tmp_path),
    )
    assert not any(k.startswith("holdout_") for k in out)


def test_metrics_file_written(tmp_path):
    mfile = tmp_path / "metrics.json"
    evaluate(_Model(), {"evaluate": {"metrics_file": str(mfile)}}, _Store(_matchups(), tmp_path))
    assert mfile.exists()
