from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from PipelinePython.common.export_onnx import ONNXExporter
from PipelinePython.common.settings import PipelinePaths, PipelineSettings


def _tmp_paths(tmp_path: Path) -> PipelinePaths:
    artifacts = tmp_path / "data" / "artifacts"
    return PipelinePaths(
        project_root=tmp_path,
        raw_data_root=tmp_path / "data" / "raw",
        anchor_csv_path=tmp_path / "anchors.csv",
        artifacts_root=artifacts,
        bronze_dir=tmp_path / "data" / "bronze",
        silver_dir=tmp_path / "data" / "silver",
        gold_dir=tmp_path / "data" / "gold",
        gold_features_path=artifacts / "gold" / "gold_features.csv",
        eda_dir=artifacts / "eda",
        validation_dir=artifacts / "validation",
        models_dir=artifacts / "models",
        export_dir=artifacts / "export",
    )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_export_writes_onnx_parity_and_manifest(tmp_path: Path) -> None:
    paths = _tmp_paths(tmp_path)
    settings = PipelineSettings(random_seed=42)
    sensor = "ppg_ir"
    feature_cols = ["num_peaks", "hrv_HRV_RMSSD", "mean_rr_ms", "ppg_ir_mean"]

    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "num_peaks": rng.normal(80, 5, 120),
            "hrv_HRV_RMSSD": rng.normal(35, 4, 120),
            "mean_rr_ms": rng.normal(740, 20, 120),
            "ppg_ir_mean": rng.normal(0.0, 1.0, 120),
        }
    )
    y = np.array([0, 1] * 60)
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(X, y)

    model_dir = paths.models_dir / "stress" / sensor
    model_dir.mkdir(parents=True)
    joblib.dump(model, model_dir / "best_model.joblib")
    (model_dir / "selected_features.json").write_text(
        json.dumps(
            {
                "selected_features": feature_cols,
                "sensor_channel": sensor,
                "feature_version": "v1.2.0_180s",
                "model_window_seconds": 180,
                "model_window_step_seconds": 10,
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "split_and_metrics.json").write_text(
        json.dumps(
            {
                "model_name": "logreg",
                "feature_version": "v1.2.0_180s",
                "model_window_seconds": 180,
                "model_window_step_seconds": 10,
                "test_metrics": {
                    "balanced_accuracy": 0.9,
                    "f1": 0.9,
                    "roc_auc": 0.9,
                    "below_threshold": False,
                },
            }
        ),
        encoding="utf-8",
    )

    gold = X.copy()
    gold["phase"] = ["Relax", "Task"] * 60
    gold["sensor_channel"] = sensor
    gold["quality_tier"] = "excellent"
    paths.gold_features_path.parent.mkdir(parents=True)
    gold.to_csv(paths.gold_features_path, index=False)

    result = ONNXExporter(paths, settings).run(sensor)

    assert result["checked_rows"] == 100
    assert result["parity_status"] == "ok"
    assert result["parity_probability_max_abs_diff"] <= 1e-5
    assert (paths.export_dir / f"stress_{sensor}_model.onnx").exists()
    manifest = json.loads(
        (paths.export_dir / f"model_manifest_{sensor}.json").read_text()
    )
    assert manifest["feature_version"] == "v1.2.0_180s"
    assert manifest["model_window_seconds"] == 180
    assert manifest["sensor_channel"] == sensor

    parity = json.loads((paths.export_dir / f"onnx_parity_{sensor}.json").read_text())
    assert parity["model_window_seconds"] == 180
