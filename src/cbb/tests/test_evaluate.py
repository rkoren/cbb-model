"""Tests for cbb.evaluate."""

import numpy as np
import pytest

from cbb.evaluate import (
    TourneyEvalResult,
    eval_loto,
    from_season_dict,
    per_season_brier,
)


# ── per_season_brier ──────────────────────────────────────────────────────────

def test_per_season_brier_perfect():
    result = per_season_brier([2024, 2024], [1.0, 0.0], [1.0, 0.0])
    assert result[2024] == pytest.approx(0.0)


def test_per_season_brier_groups_correctly():
    seasons = [2024, 2024, 2025, 2025]
    y_true = [1.0, 0.0, 1.0, 0.0]
    y_prob = [1.0, 0.0, 0.5, 0.5]
    result = per_season_brier(seasons, y_true, y_prob)
    assert result[2024] == pytest.approx(0.0)
    assert result[2025] == pytest.approx(0.25)


def test_per_season_brier_returns_all_seasons():
    result = per_season_brier([2023, 2024, 2025], [1, 1, 1], [0.5, 0.5, 0.5])
    assert set(result.keys()) == {2023, 2024, 2025}


def test_per_season_brier_uses_platform_brier():
    y_true = [1.0, 0.0]
    y_prob = [0.8, 0.2]
    result = per_season_brier([2026, 2026], y_true, y_prob)
    expected = np.mean((np.array(y_prob) - np.array(y_true)) ** 2)
    assert result[2026] == pytest.approx(expected)


# ── eval_loto ─────────────────────────────────────────────────────────────────

def test_eval_loto_returns_tourney_eval_result():
    result = eval_loto([2024, 2024], [1.0, 0.0], [0.5, 0.5])
    assert isinstance(result, TourneyEvalResult)


def test_eval_loto_overall_brier():
    result = eval_loto([2024, 2024], [1.0, 0.0], [0.5, 0.5])
    assert result.overall_brier == pytest.approx(0.25)


def test_eval_loto_n_games():
    result = eval_loto([2024, 2024, 2025], [1, 0, 1], [0.5, 0.5, 0.5])
    assert result.n_games == 3


# ── from_season_dict ──────────────────────────────────────────────────────────

def test_from_season_dict_overall_is_mean():
    d = {2024: 0.10, 2025: 0.20, 2026: 0.15}
    result = from_season_dict(d)
    assert result.overall_brier == pytest.approx(0.15)


def test_from_season_dict_preserves_seasons():
    d = {2024: 0.12, 2026: 0.09}
    result = from_season_dict(d, n_games=64)
    assert result.per_season == d
    assert result.n_games == 64


def test_from_season_dict_empty():
    result = from_season_dict({})
    assert np.isnan(result.overall_brier)


# ── TourneyEvalResult.season() ────────────────────────────────────────────────

def test_season_accessor_returns_value():
    result = from_season_dict({2026: 0.12})
    assert result.season(2026) == pytest.approx(0.12)


def test_season_accessor_returns_none_for_missing():
    result = from_season_dict({2026: 0.12})
    assert result.season(9999) is None
