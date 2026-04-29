"""Evidently drift reports: compare predictions vs. actuals post-tournament.

Run after tournament results are available to detect where the model's
confidence was miscalibrated — which seed tiers, conferences, or game types
drove the most Brier loss.
"""

from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import ClassificationPreset, DataDriftPreset
from evidently.report import Report


def build_calibration_report(
    matchups: pd.DataFrame,
    output_path: str | Path = "monitoring/reports/calibration.html",
    reference_seasons: list[int] | None = None,
    current_seasons: list[int] | None = None,
) -> None:
    """Generate Evidently classification report comparing predicted vs. actual outcomes.

    Splits data into reference (older seasons) and current (recent seasons) to detect
    whether calibration has drifted — useful after each tournament to check if the
    model is over/under-confident in new ways.

    Args:
        matchups: Full matchup dataset with pred_prob_final and Outcome columns.
        output_path: Where to write the HTML report.
        reference_seasons: Seasons to use as reference (baseline). Defaults to all
                           seasons except the most recent two.
        current_seasons: Seasons to treat as "current" data. Defaults to most recent two.
    """
    all_seasons = sorted(matchups["Season"].unique())

    if reference_seasons is None:
        reference_seasons = all_seasons[:-2] if len(all_seasons) > 2 else all_seasons[:1]
    if current_seasons is None:
        current_seasons = all_seasons[-2:] if len(all_seasons) >= 2 else all_seasons[-1:]

    scored = ~matchups["is_playin"] if "is_playin" in matchups.columns else pd.Series(True, index=matchups.index)

    ref = matchups[matchups["Season"].isin(reference_seasons) & scored][
        ["pred_prob_final", "Outcome", "men_women", "seed_gap"]
    ].rename(columns={"pred_prob_final": "prediction", "Outcome": "target"})

    cur = matchups[matchups["Season"].isin(current_seasons) & scored][
        ["pred_prob_final", "Outcome", "men_women", "seed_gap"]
    ].rename(columns={"pred_prob_final": "prediction", "Outcome": "target"})

    column_mapping = ColumnMapping(
        target="target",
        prediction="prediction",
        numerical_features=["seed_gap"],
        categorical_features=["men_women"],
    )

    report = Report(metrics=[ClassificationPreset(), DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur, column_mapping=column_mapping)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_path))
    print(f"Calibration report saved to {output_path}")


def build_feature_drift_report(
    ref_matchups: pd.DataFrame,
    cur_matchups: pd.DataFrame,
    features: list[str],
    output_path: str | Path = "monitoring/reports/feature_drift.html",
) -> None:
    """Detect drift in input feature distributions between two seasons.

    Useful for checking whether the 2027 regular season looks statistically
    similar to training data — large drift in efficiency distributions can
    signal that the model is extrapolating outside its training range.

    Args:
        ref_matchups: Historical matchups (training distribution).
        cur_matchups: Current season matchups (prediction distribution).
        features: Feature columns to monitor for drift.
        output_path: Where to write the HTML report.
    """
    d_features = [f for f in features if f.startswith("d_")]

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=ref_matchups[d_features].fillna(0),
        current_data=cur_matchups[d_features].fillna(0),
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_path))
    print(f"Feature drift report saved to {output_path}")
