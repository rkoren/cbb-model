"""Migrate MLflow runs from file store (./mlruns) to SQLite (mlruns.db).

Artifact files are NOT moved — artifact URIs in the new runs point to the
same absolute paths in ./mlruns/ that they always have.

Usage:
    python3 scripts/migrate_to_sqlite.py
"""

from __future__ import annotations

import os
import time

import mlflow
from mlflow.tracking import MlflowClient

FILE_STORE_URI = "file:./mlruns"
SQLITE_URI = "sqlite:///mlruns.db"
MODEL_NAME = "cbb-tournament-model"
CHAMPION_METRIC = "loto_brier"
LOWER_IS_BETTER = True


def migrate() -> None:
    src = MlflowClient(tracking_uri=FILE_STORE_URI)
    dst = MlflowClient(tracking_uri=SQLITE_URI)

    # ── 1. Copy experiments ────────────────────────────────────────────────────
    src_exps = {e.name: e for e in src.search_experiments()
                if e.name != "Default"}
    dst_exps = {e.name: e for e in dst.search_experiments()}

    exp_id_map: dict[str, str] = {}  # src experiment_id → dst experiment_id

    for name, src_exp in src_exps.items():
        if name not in dst_exps:
            dst_exp_id = dst.create_experiment(
                name,
                artifact_location=src_exp.artifact_location,
                tags=src_exp.tags,
            )
            print(f"  Created experiment '{name}' → {dst_exp_id}")
        else:
            dst_exp_id = dst_exps[name].experiment_id
            print(f"  Experiment '{name}' already exists → {dst_exp_id}")
        exp_id_map[src_exp.experiment_id] = dst_exp_id

    # ── 2. Copy runs ───────────────────────────────────────────────────────────
    run_id_map: dict[str, str] = {}  # src run_id → dst run_id

    # Check which runs are already in dst (by matching src_run_id tag we'll add)
    already_migrated: set[str] = set()
    for dst_exp_id in exp_id_map.values():
        for r in dst.search_runs([dst_exp_id]):
            src_id = r.data.tags.get("migrated_from_run_id")
            if src_id:
                already_migrated.add(src_id)

    for src_exp_id, dst_exp_id in exp_id_map.items():
        src_runs = src.search_runs([src_exp_id])
        print(f"\n  Migrating {len(src_runs)} runs from experiment {src_exp_id}:")

        for src_run in src_runs:
            ri = src_run.info
            if ri.run_id in already_migrated:
                print(f"    {ri.run_id[:8]}  {ri.run_name}  — already migrated, skip")
                continue

            # Create run in dst, preserving artifact URI so it keeps pointing
            # at the existing files in ./mlruns/
            dst_run = dst.create_run(
                experiment_id=dst_exp_id,
                run_name=ri.run_name,
                start_time=ri.start_time,
                tags={
                    **src_run.data.tags,
                    "migrated_from_run_id": ri.run_id,
                },
            )
            new_id = dst_run.info.run_id

            # Manually set artifact_uri to point at the original file store path
            # MLflow doesn't expose a setter — we use update_run which accepts
            # end_time and status only, so we store the original artifact path
            # as a tag instead for reference.
            dst.set_tag(new_id, "original_artifact_uri", ri.artifact_uri)

            # Params
            for k, v in src_run.data.params.items():
                dst.log_param(new_id, k, v)

            # Metrics (log_metric needs a timestamp; use run start_time)
            ts = ri.start_time or int(time.time() * 1000)
            for k, v in src_run.data.metrics.items():
                dst.log_metric(new_id, k, v, timestamp=ts)

            # Terminal status
            dst.update_run(new_id, status=ri.status)

            run_id_map[ri.run_id] = new_id
            print(f"    {ri.run_id[:8]} → {new_id[:8]}  {ri.run_name}  "
                  f"loto_brier={src_run.data.metrics.get('loto_brier', 'n/a')}")

    # ── 3. Register champion in model registry ─────────────────────────────────
    print(f"\n  Registering champion in model registry ('{MODEL_NAME}')...")

    # Find the best LOTO run by metric
    loto_runs = [
        (run_id_map.get(r.info.run_id, r.info.run_id),
         r.data.metrics.get(CHAMPION_METRIC))
        for exp_id in exp_id_map.values()
        for r in dst.search_runs([exp_id])
        if CHAMPION_METRIC in r.data.metrics
    ]
    if not loto_runs:
        print("  No runs with loto_brier found — skipping champion registration.")
        return

    best_run_id, best_val = min(loto_runs, key=lambda x: x[1]) if LOWER_IS_BETTER \
        else max(loto_runs, key=lambda x: x[1])
    print(f"  Best run: {best_run_id[:8]}  {CHAMPION_METRIC}={best_val:.6f}")

    # Find the original artifact URI for this run (stored as tag)
    best_run = dst.get_run(best_run_id)
    artifact_uri = best_run.data.tags.get("original_artifact_uri",
                                           best_run.info.artifact_uri)
    model_uri = f"{artifact_uri}/cbb_model"

    # Register model version
    try:
        dst.create_registered_model(MODEL_NAME)
        print(f"  Created registered model '{MODEL_NAME}'")
    except mlflow.exceptions.MlflowException:
        print(f"  Registered model '{MODEL_NAME}' already exists")

    mv = dst.create_model_version(
        name=MODEL_NAME,
        source=model_uri,
        run_id=best_run_id,
    )
    print(f"  Created model version {mv.version} from run {best_run_id[:8]}")

    # Set champion alias
    dst.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
    print(f"  Set alias 'champion' → version {mv.version}")

    print("\n✓ Migration complete.")
    print(f"  Add to .env:  MLFLOW_TRACKING_URI={SQLITE_URI}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    migrate()
