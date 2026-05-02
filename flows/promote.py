"""Promote the best-performing model to Production in the MLflow Model Registry.

Finds the run with the lowest value of the target metric across all runs (or
filtered by model_variant), registers its calibrator artifact, and transitions
it to the Production stage. The production URI can then be loaded by the serving
layer or any inference script.

Usage:
    # Compare all runs, promote the one with best loto_brier
    python flows/promote.py

    # Promote best challenger run only
    python flows/promote.py --variant challenger

    # Dry-run: print the winner without registering anything
    python flows/promote.py --dry-run

    # Compare runs from the UI first:
    mlflow ui --backend-store-uri ./mlruns
"""
from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

load_dotenv()

from kitchen import tracking
from kitchen.registry import get_best_run, get_production_uri, promote_model, register_model

EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "cbb-tournament")
MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "cbb-tournament-model")


def show_comparison(experiment: str, metric: str) -> None:
    """Print a side-by-side comparison of best baseline vs challenger."""
    import mlflow.tracking
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        print(f"Experiment {experiment!r} not found.")
        return

    print(f"\nExperiment: {experiment}  |  Ranked by: {metric}  (lower = better)\n")
    print(f"{'Variant':<15} {'Run ID':<12} {'LOTO Brier':<14} {'Brier 2026':<14}")
    print("-" * 60)

    for variant in ("baseline", "challenger"):
        try:
            run = get_best_run(experiment, metric, lower_is_better=True,
                               tag_filter={"model_variant": variant})
            loto = run.data.metrics.get("loto_brier", float("nan"))
            b2026 = run.data.metrics.get("brier_2026", float("nan"))
            loto_str = f"{loto:.6f}" if loto == loto else "n/a"
            b2026_str = f"{b2026:.6f}" if b2026 == b2026 else "n/a (no 2026 data)"
            print(f"{variant:<15} {run.info.run_id[:8]:<12} {loto_str:<14} {b2026_str:<14}")
        except ValueError:
            print(f"{variant:<15} {'(no runs)':<12}")

    print()


def promote(
    metric: str = "loto_brier",
    variant: str | None = None,
    model_name: str = MODEL_NAME,
    dry_run: bool = False,
) -> None:
    tracking.configure_from_env()

    show_comparison(EXPERIMENT, metric)

    if variant:
        run = get_best_run(EXPERIMENT, metric, lower_is_better=True,
                           tag_filter={"model_variant": variant})
    else:
        # Always pick from known tagged variants — never promote untagged legacy runs
        candidates = []
        for v in ("baseline", "challenger"):
            try:
                candidates.append(
                    get_best_run(EXPERIMENT, metric, lower_is_better=True,
                                 tag_filter={"model_variant": v})
                )
            except ValueError:
                pass
        if not candidates:
            raise ValueError("No baseline or challenger runs found in experiment")
        run = min(candidates, key=lambda r: r.data.metrics.get(metric, float("inf")))

    run_id = run.info.run_id
    brier = run.data.metrics.get(metric, float("nan"))
    variant_tag = run.data.tags.get("model_variant", "unknown")

    print(f"Winner: {run_id} ({variant_tag})  {metric}={brier:.6f}")

    current = get_production_uri(model_name)
    if current:
        print(f"Current production model: {current}")

    if dry_run:
        print("Dry run — skipping registration and promotion.")
        return

    version = register_model(run_id, "calibrator", model_name)
    print(f"Registered {model_name} v{version}")

    promote_model(model_name, version)
    print(f"Promoted {model_name} v{version} → champion")
    print(f"Load with: mlflow.sklearn.load_model('models:/{model_name}@champion')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote the best CBB model to Production.")
    parser.add_argument(
        "--metric", default="loto_brier",
        help="Metric to compare across runs (lower is better). Default: loto_brier",
    )
    parser.add_argument(
        "--variant", default=None,
        help="Restrict to a specific model_variant tag (baseline or challenger).",
    )
    parser.add_argument(
        "--model-name", default=MODEL_NAME,
        help=f"MLflow registered model name. Default: {MODEL_NAME}",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the winning run without registering or promoting.",
    )
    args = parser.parse_args()

    promote(
        metric=args.metric,
        variant=args.variant,
        model_name=args.model_name,
        dry_run=args.dry_run,
    )
