"""Calibration must be monotonic and fitted only on development OOF data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from PipelinePython.common.calibration import (
    DEVELOPMENT_COHORT,
    calibration_metrics,
    fit_grouped_platt_calibration,
    fit_platt_from_nested_oof,
)


def _estimator() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )


def test_grouped_calibration_uses_complete_development_cohort() -> None:
    rows = []
    for index, participant in enumerate(DEVELOPMENT_COHORT):
        for sample in range(2):
            value = float(index + sample) / 30.0
            rows.append((participant, value, int(index >= 14)))
    frame = pd.DataFrame(rows, columns=["participant_id", "feature", "label"])

    calibration, oof = fit_grouped_platt_calibration(
        estimator_factory=_estimator,
        features=frame[["feature"]],
        labels=frame["label"],
        groups=frame["participant_id"],
    )

    assert calibration.fitted_cohort == DEVELOPMENT_COHORT
    assert calibration.fitted_on == "participant_grouped_development_oof"
    assert calibration.calibration_status == "validated_development_oof"
    assert np.isfinite(oof).all()
    calibrated = calibration.apply(np.array([0.1, 0.5, 0.9]))
    assert np.all(np.diff(calibrated) > 0)


def test_calibration_metrics_are_finite_and_bounded() -> None:
    metrics = calibration_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])

    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert metrics["log_loss"] >= 0.0
    assert 0.0 <= metrics["ece"] <= 1.0


def test_nested_outer_oof_calibration_uses_equal_participant_phase_weight() -> None:
    participants = np.repeat(DEVELOPMENT_COHORT, 2)
    phases = np.tile(["relax", "task"], len(DEVELOPMENT_COHORT))
    labels = np.tile([0, 1], len(DEVELOPMENT_COHORT))
    probabilities = np.where(labels == 1, 0.8, 0.2)

    calibration = fit_platt_from_nested_oof(
        labels,
        probabilities,
        participants,
        np.char.add(np.char.add(participants, "|"), phases),
    )

    assert calibration.fitted_on == "nested_outer_development_oof"
    assert calibration.calibration_status == "validated_nested_outer_oof"
    assert np.all(np.diff(calibration.apply([0.1, 0.5, 0.9])) > 0)


def test_nested_outer_oof_calibration_accepts_explicit_ready_cohort() -> None:
    ready_cohort = DEVELOPMENT_COHORT[:-1]
    participants = np.repeat(ready_cohort, 2)
    phases = np.tile(["relax", "task"], len(ready_cohort))
    labels = np.tile([0, 1], len(ready_cohort))
    probabilities = np.where(labels == 1, 0.8, 0.2)

    calibration = fit_platt_from_nested_oof(
        labels,
        probabilities,
        participants,
        np.char.add(np.char.add(participants, "|"), phases),
        fitted_cohort=ready_cohort,
    )

    assert calibration.fitted_cohort == tuple(ready_cohort)
