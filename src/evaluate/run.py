"""CBB model evaluation — kitchen-platform adapter.

Called by ``kitchen run evaluate`` as::

    from src.evaluate.run import evaluate
    evaluate(model, params, DataStore())

The model handed in is the **production champion**, trained on every season in
``matchups.parquet`` (2003–2025). Scoring it back on those same rows is therefore
*in-sample* (resubstitution) — optimistic and NOT a generalization metric. SC-004:
those numbers are emitted as ``insample_brier`` (clearly named so they can't be
read as leave-one-tournament-out folds). This stage owns **only** that in-sample
diagnostic: the two trusted, leak-free numbers are owned by other parts of the
platform so they can't collide here (the CBB-019 last-write-wins footgun) — the
CV ``loto_brier`` by the *train* stage, and the 2026 ``holdout_brier`` by the
platform's ``holdout:`` config (CBB-017), which scores every train run's model on
the frozen holdout the features stage builds. (evaluate used to emit ``loto_brier``
and score the holdout itself; both were removed.)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from kitchen.store import DataStore

from cbb.evaluate import per_season_brier
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
        Flat dict of metric_name → float — the ``insample_*`` resubstitution
        diagnostics (champion scored on its own training seasons, optimistic by
        construction)::

            {
                "insample_brier": 0.1221,        # optimistic — fit check, NOT generalization
                "insample_brier_2024": 0.1180,
                "insample_brier_2025": 0.1150,
            }

        Also written to ``metrics.json`` (path from ``params.evaluate.metrics_file``).

        The leak-free generalization number (``holdout_brier``) is no longer produced
        here: the platform's ``holdout:`` config (CBB-017) scores every train run's model
        on the frozen 2026 holdout and logs it to MLflow. Emitting it here too would
        double-log the metric into one run (the CBB-019 last-write-wins footgun).
    """
    matchups = store.load_parquet("matchups.parquet")
    # Every row in matchups.parquet is a tournament game — build_matchup_dataset is
    # constructed from the tournament game log (tourn_sym), so there is no is_tourn
    # column to filter on and no need for one.
    tourn = matchups.copy()
    log.info("Evaluating on %d tournament matchup rows (in-sample)", len(tourn))

    # predict_batch is available on CBBModel; when model is loaded via
    # mlflow.sklearn.load_model it's a deserialized CBBModel with the same method.
    if not hasattr(model, "predict_batch"):
        raise TypeError(
            f"model must have a predict_batch() method; got {type(model).__name__}"
        )

    # ── In-sample (resubstitution) Brier — a fit diagnostic, NOT generalization. ──
    # The champion trained on every one of these seasons, so this is optimistic by
    # construction; named `insample_*` so it can never be mistaken for a LOTO fold.
    probs = model.predict_batch(tourn)
    y = np.asarray(tourn["Outcome"].values, dtype=float)
    probs = np.asarray(probs, dtype=float)
    from kitchen.evaluate import brier_score  # noqa: PLC0415

    per_season = per_season_brier(tourn["Season"].values, y, probs)
    metrics: dict[str, float] = {"insample_brier": float(brier_score(y, probs))}
    for season, brier in sorted(per_season.items()):
        metrics[f"insample_brier_{season}"] = brier
    log.info(
        "In-sample Brier: %.6f (optimistic) over seasons %s",
        metrics["insample_brier"], sorted(per_season),
    )

    # ── Leak-free generalization (holdout_brier) is now the platform's job. ───────
    # The platform's `holdout:` config (CBB-017) scores every train run's model on the
    # frozen 2026 holdout parquet — which `src/features/run.py` still builds with feature
    # parity — and logs `holdout_<metric>` onto the run. This stage only owns the in-sample
    # diagnostic above; scoring the holdout here too would double-log it (CBB-019).

    metrics_file = params.get("evaluate", {}).get("metrics_file", "metrics.json")
    Path(metrics_file).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("Metrics written → %s", metrics_file)

    return metrics
