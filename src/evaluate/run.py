"""CBB model evaluation — kitchen-platform adapter.

Called by ``kitchen run evaluate`` as::

    from src.evaluate.run import evaluate
    evaluate(model, params, DataStore())

The model handed in is the **production champion**, trained on every season in
``matchups.parquet`` (2003–2025). Scoring it back on those same rows is therefore
*in-sample* (resubstitution) — optimistic and NOT a generalization metric. SC-004:
those numbers are emitted as ``insample_brier`` (clearly named so they can't be
read as leave-one-tournament-out folds), and the genuinely leak-free generalization
number comes from scoring the champion on the 2026 holdout (``holdout_brier`` etc.),
which it never trained on. The leak-aware CV ``loto_brier`` is owned by the *train*
stage; evaluate no longer emits that name (it used to, which overwrote train's real
LOTO with the optimistic in-sample value in each run's metrics — poisoning the
threshold gate and leaderboard).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from kitchen.store import DataStore

from cbb.evaluate import per_season_brier
from cbb.holdout import HOLDOUT_PARQUET, score_holdout
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
        Flat dict of metric_name → float. The ``insample_*`` keys are resubstitution
        diagnostics (champion scored on its own training seasons); the ``holdout_*``
        keys are the leak-free generalization metrics (2026, never trained on)::

            {
                "insample_brier": 0.1221,        # optimistic — fit check, NOT generalization
                "insample_brier_2024": 0.1180,
                "insample_brier_2025": 0.1150,
                "holdout_brier": 0.1718,         # leak-free headline
                "holdout_ece": 0.0901,
                "holdout_n_games": 67,
            }

        Also written to ``metrics.json`` (path from ``params.evaluate.metrics_file``).
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

    # ── Leak-free generalization: score the champion on the 2026 holdout. ─────────
    # Use the features the *loaded model* expects (model.features) rather than
    # re-deriving from menu.yaml, which could drift from what the champion trained on.
    # No-op until the holdout parquet is built (mirrors the train-stage guard).
    holdout_path = store.processed_dir / HOLDOUT_PARQUET
    if holdout_path.exists() and getattr(model, "features", None):
        try:
            hscore = score_holdout(model, store.load_parquet(HOLDOUT_PARQUET), model.features)
            metrics.update({k: float(v) for k, v in hscore.items()})
            extra = ""
            if "holdout_margin_mae" in hscore:
                extra = "  margin_MAE %.2f  total_MAE %.2f" % (
                    hscore["holdout_margin_mae"], hscore["holdout_total_mae"],
                )
            log.info(
                "Holdout 2026 Brier %.6f  ECE %.4f over %d games%s",
                hscore["holdout_brier"], hscore["holdout_ece"], hscore["holdout_n_games"], extra,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Holdout scoring skipped: %s", exc)
    else:
        log.info("No %s (or model lacks .features) — holdout_brier not logged", HOLDOUT_PARQUET)

    metrics_file = params.get("evaluate", {}).get("metrics_file", "metrics.json")
    Path(metrics_file).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("Metrics written → %s", metrics_file)

    return metrics
