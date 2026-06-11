"""CBB model evaluation — kitchen-platform adapter.

Called by ``kitchen run evaluate`` as::

    from src.evaluate.run import evaluate
    evaluate(model, params, DataStore())

Computes per-season Brier score on tournament games from the matchup dataset,
writes ``metrics.json``, and returns a flat metrics dict. The model is loaded
by the kitchen CLI from the MLflow registry before calling this function.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from kitchen.store import DataStore

from cbb.evaluate import TourneyEvalResult, eval_loto, per_season_brier
from cbb.train.model import CBBModel

log = logging.getLogger(__name__)


def evaluate(model: CBBModel | object, params: dict, store: DataStore) -> dict[str, float]:
    """Evaluate a CBBModel on tournament games and write metrics.json.

    Loads the matchup dataset from ``data/processed/matchups.parquet``, filters
    to tournament rows, runs batch prediction, and computes Brier score per
    season and overall.

    Args:
        model: A ``CBBModel`` instance (or any object with a ``predict_batch``
               method). When loaded via ``kitchen run evaluate --flavor sklearn``
               from the MLflow registry, this is a deserialized ``CBBModel``.
        params: Parsed ``params.yaml`` dict.
        store: ``DataStore`` rooted at the project directory.

    Returns:
        Flat dict of metric_name → float, e.g.::

            {
                "loto_brier": 0.1693,
                "brier_2024": 0.1712,
                "brier_2025": 0.1650,
                "brier_2026": 0.1718,
            }

        Also written to ``metrics.json`` (path from ``params.evaluate.metrics_file``).
    """
    matchups = store.load_parquet("matchups.parquet")
    # Every row in matchups.parquet is a tournament game — build_matchup_dataset is
    # constructed from the tournament game log (tourn_sym), so there is no is_tourn
    # column to filter on and no need for one.
    tourn = matchups.copy()
    log.info("Evaluating on %d tournament matchup rows", len(tourn))

    # predict_batch is available on CBBModel; when model is loaded via
    # mlflow.sklearn.load_model it's a deserialized CBBModel with the same method.
    if hasattr(model, "predict_batch"):
        probs = model.predict_batch(tourn)
    else:
        raise TypeError(
            f"model must have a predict_batch() method; got {type(model).__name__}"
        )

    result: TourneyEvalResult = eval_loto(
        seasons=tourn["Season"].values,
        y_true=tourn["Outcome"].values,
        y_prob=probs,
    )

    metrics: dict[str, float] = {"loto_brier": result.overall_brier}
    for season, brier in sorted(result.per_season.items()):
        metrics[f"brier_{season}"] = brier
    log.info(
        "Evaluation complete — overall Brier: %.6f  seasons: %s",
        result.overall_brier,
        sorted(result.per_season),
    )

    metrics_file = params.get("evaluate", {}).get("metrics_file", "metrics.json")
    Path(metrics_file).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("Metrics written → %s", metrics_file)

    return metrics
