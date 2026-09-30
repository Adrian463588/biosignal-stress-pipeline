from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from PipelinePython.common.training_v3 import (
    _inner_oof_probability,
    _metrics,
    _threshold_for_balanced_accuracy,
)


def test_inner_group_predictions_are_deterministic() -> None:
    groups = np.repeat([f"ID-{index:02d}" for index in range(1, 7)], 10)
    labels = np.tile(np.repeat([0, 1], 5), 6)
    values = np.column_stack(
        [
            labels + np.arange(labels.size) * 1.0e-4,
            np.sin(np.arange(labels.size)),
        ]
    )
    features = pd.DataFrame(values, columns=["feature_a", "feature_b"])
    estimator = LogisticRegression(random_state=42, max_iter=1000)

    first = _inner_oof_probability(estimator, features, labels, groups, 3)
    second = _inner_oof_probability(estimator, features, labels, groups, 3)

    np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)
    assert _threshold_for_balanced_accuracy(labels, first) == (
        _threshold_for_balanced_accuracy(labels, second)
    )


def test_metrics_accept_fold_local_predictions() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=int)
    probability = np.asarray([0.4, 0.6, 0.4, 0.6], dtype=float)
    prediction = np.asarray([0, 0, 1, 1], dtype=int)

    metrics = _metrics(labels, probability, 0.5, prediction=prediction)

    assert metrics.balanced_accuracy == 1.0
    assert metrics.threshold == 0.5
