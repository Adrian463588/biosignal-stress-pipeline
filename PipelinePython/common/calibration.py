"""Participant-safe probability calibration for the canonical stress models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

CALIBRATION_VERSION = "platt-v1-dev-oof"
NESTED_CALIBRATION_VERSION = "platt-v2-nested-outer-oof"
DEVELOPMENT_COHORT = tuple(f"ID-{index:02d}" for index in range(1, 29))


def _logit(probability: np.ndarray) -> np.ndarray:
    bounded = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(bounded / (1.0 - bounded))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-value))


def calibration_metrics(
    labels: Sequence[int], probability: Sequence[float], bins: int = 10
) -> dict[str, float]:
    """Return deterministic Brier, log-loss, and equal-width ECE metrics."""

    y_true = np.asarray(labels, dtype=np.float64)
    predicted = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    if y_true.shape != predicted.shape or y_true.size == 0:
        raise ValueError("calibration_shape_mismatch")
    if not np.isfinite(predicted).all() or not np.isfinite(y_true).all():
        raise ValueError("non_finite_calibration_input")
    if not np.isin(y_true, [0.0, 1.0]).all():
        raise ValueError("calibration_labels_must_be_binary")

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (predicted >= edges[index]) & (predicted <= edges[index + 1])
        else:
            mask = (predicted >= edges[index]) & (predicted < edges[index + 1])
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(predicted[mask].mean()) - float(y_true[mask].mean())
            )

    return {
        "brier_score": float(np.mean((predicted - y_true) ** 2)),
        "log_loss": float(
            -np.mean(
                y_true * np.log(predicted) + (1.0 - y_true) * np.log(1.0 - predicted)
            )
        ),
        "ece": float(ece),
    }


@dataclass(frozen=True)
class PlattCalibration:
    method: str
    version: str
    slope: float
    intercept: float
    fitted_cohort: tuple[str, ...]
    fitted_on: str
    calibration_status: str
    brier_score: float
    log_loss: float
    ece: float

    def apply(self, probability: Sequence[float] | float) -> np.ndarray:
        calibrated = _sigmoid(
            self.slope * _logit(np.asarray(probability)) + self.intercept
        )
        if not np.isfinite(calibrated).all():
            raise ValueError("non_finite_calibrated_probability")
        return np.clip(calibrated, 0.0, 1.0)

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["fitted_cohort"] = list(self.fitted_cohort)
        return result


def fit_platt_from_nested_oof(
    labels: Sequence[int],
    probabilities: Sequence[float],
    groups: Sequence[str],
    phase_groups: Sequence[str],
    fitted_cohort: Sequence[str] = DEVELOPMENT_COHORT,
) -> PlattCalibration:
    """Fit Platt parameters on predictions from the locked nested outer CV."""

    y_true = np.asarray(labels, dtype=np.int64)
    oof = np.asarray(probabilities, dtype=np.float64)
    participant_values = np.asarray(groups, dtype=str)
    phase_values = np.asarray(phase_groups, dtype=str)
    if not (len(y_true) == len(oof) == len(participant_values) == len(phase_values)):
        raise ValueError("nested_calibration_shape_mismatch")
    if y_true.size == 0 or not np.isfinite(oof).all():
        raise ValueError("invalid_nested_calibration_oof")
    if set(participant_values) != set(fitted_cohort):
        raise ValueError("nested_calibration_cohort_mismatch")
    if not np.isin(y_true, [0, 1]).all() or np.unique(y_true).size != 2:
        raise ValueError("nested_calibration_labels_must_be_binary")
    unique_phase_groups, counts = np.unique(phase_values, return_counts=True)
    count_by_group = dict(zip(unique_phase_groups, counts, strict=True))
    weights = np.asarray(
        [1.0 / float(count_by_group[value]) for value in phase_values],
        dtype=np.float64,
    )
    calibrator = LogisticRegression(solver="lbfgs", C=1e6, random_state=42)
    calibrator.fit(
        _logit(oof).reshape(-1, 1),
        y_true,
        sample_weight=weights,
    )
    calibrated = _sigmoid(
        calibrator.coef_[0, 0] * _logit(oof) + calibrator.intercept_[0]
    )
    parameters = calibration_metrics(y_true, calibrated)
    return PlattCalibration(
        method="platt",
        version=NESTED_CALIBRATION_VERSION,
        slope=float(calibrator.coef_[0, 0]),
        intercept=float(calibrator.intercept_[0]),
        fitted_cohort=tuple(fitted_cohort),
        fitted_on="nested_outer_development_oof",
        calibration_status="validated_nested_outer_oof",
        **parameters,
    )


def fit_grouped_platt_calibration(
    estimator_factory: Callable[[], object],
    features: object,
    labels: Sequence[int],
    groups: Sequence[str],
    fitted_cohort: Sequence[str] = DEVELOPMENT_COHORT,
    feature_selector: Callable[[object], Sequence[str]] | None = None,
    phase_groups: Sequence[str] | None = None,
) -> tuple[PlattCalibration, np.ndarray]:
    """Fit Platt parameters on participant-grouped development OOF only."""

    y_true = np.asarray(labels, dtype=np.int64)
    group_values = np.asarray(groups, dtype=str)
    if set(group_values) != set(fitted_cohort):
        raise ValueError("calibration_cohort_mismatch")
    if len(y_true) != len(group_values):
        raise ValueError("calibration_shape_mismatch")
    weight_values = None
    if phase_groups is not None:
        weight_groups = np.asarray(phase_groups, dtype=str)
        if len(weight_groups) != len(y_true):
            raise ValueError("calibration_weight_group_mismatch")
        counts = np.unique(weight_groups, return_counts=True)
        count_map = dict(zip(counts[0], counts[1], strict=True))
        weight_values = np.asarray(
            [1.0 / float(count_map[group]) for group in weight_groups],
            dtype=np.float64,
        )

    oof = np.full(y_true.shape[0], np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=5)
    for train_index, validation_index in splitter.split(features, y_true, group_values):
        estimator = estimator_factory()
        train_features = features.iloc[train_index]
        validation_features = features.iloc[validation_index]
        if feature_selector is not None:
            selected = tuple(feature_selector(train_features))
            if not selected:
                raise ValueError("calibration_feature_selection_empty")
            train_features = train_features.loc[:, selected]
            validation_features = validation_features.loc[:, selected]
        if isinstance(estimator, Pipeline):
            fit_kwargs = {
                "classifier__sample_weight": (
                    weight_values[train_index] if weight_values is not None else None
                )
            }
            if fit_kwargs["classifier__sample_weight"] is None:
                fit_kwargs = {}
            estimator.fit(train_features, y_true[train_index], **fit_kwargs)
        else:
            estimator.fit(train_features, y_true[train_index])
        oof[validation_index] = estimator.predict_proba(validation_features)[:, 1]
    if not np.isfinite(oof).all():
        raise ValueError("incomplete_calibration_oof")

    calibrator = LogisticRegression(solver="lbfgs", C=1e6, random_state=42)
    if weight_values is None:
        calibrator.fit(_logit(oof).reshape(-1, 1), y_true)
    else:
        calibrator.fit(
            _logit(oof).reshape(-1, 1),
            y_true,
            sample_weight=weight_values,
        )
    parameters = calibration_metrics(
        y_true,
        _sigmoid(calibrator.coef_[0, 0] * _logit(oof) + calibrator.intercept_[0]),
    )
    return (
        PlattCalibration(
            method="platt",
            version=CALIBRATION_VERSION,
            slope=float(calibrator.coef_[0, 0]),
            intercept=float(calibrator.intercept_[0]),
            fitted_cohort=tuple(fitted_cohort),
            fitted_on="participant_grouped_development_oof",
            calibration_status="validated_development_oof",
            **parameters,
        ),
        oof,
    )
