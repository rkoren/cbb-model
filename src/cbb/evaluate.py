"""CBB-specific evaluation methodology built on mlplatform.evaluate.

Defines how tournament prediction quality is measured: per-season Brier
breakdown over LOTO folds, with a named holdout season surfaced separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mlplatform.evaluate import brier_score


@dataclass
class TourneyEvalResult:
    overall_brier: float
    per_season: dict[int, float] = field(default_factory=dict)
    n_games: int = 0

    def season(self, s: int) -> float | None:
        return self.per_season.get(s)


def per_season_brier(seasons, y_true, y_prob) -> dict[int, float]:
    seasons = np.asarray(seasons)
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return {
        int(s): brier_score(y_true[seasons == s], y_prob[seasons == s])
        for s in np.unique(seasons)
    }


def eval_loto(seasons, y_true, y_prob) -> TourneyEvalResult:
    """Build a TourneyEvalResult from raw LOTO predictions."""
    per_s = per_season_brier(seasons, y_true, y_prob)
    return TourneyEvalResult(
        overall_brier=brier_score(y_true, y_prob),
        per_season=per_s,
        n_games=len(np.asarray(y_true)),
    )


def from_season_dict(brier_by_season: dict[int, float], n_games: int = 0) -> TourneyEvalResult:
    """Build a TourneyEvalResult from a pre-computed season → Brier dict (e.g. from LoTOResult)."""
    overall = float(np.mean(list(brier_by_season.values()))) if brier_by_season else float("nan")
    return TourneyEvalResult(
        overall_brier=overall,
        per_season=dict(brier_by_season),
        n_games=n_games,
    )
