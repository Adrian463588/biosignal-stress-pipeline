from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "03_stress_onnx"
    / "tools"
    / "validate_model_index.py"
)


def _load_validator():
    spec = importlib.util.spec_from_file_location("model_index_validator", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_blocked_empty_index_is_valid(tmp_path: Path) -> None:
    validator = _load_validator()
    index_path = tmp_path / "model_index.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "release_status": "blocked_no_release_candidate",
                "models": [],
            }
        ),
        encoding="utf-8",
    )
    validator.EXPORT_ROOT = tmp_path
    validator.INDEX_PATH = index_path

    validator.validate()


def test_accepted_index_requires_device_parity_receipt(tmp_path: Path) -> None:
    validator = _load_validator()
    index_path = tmp_path / "model_index.json"
    files = {
        kind: {"path": f"{kind}.json", "sha256": "0" * 64}
        for kind in ("model", "manifest", "parity", "test_vectors")
    }
    index_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "release_status": "accepted",
                "models": [
                    {
                        "model_version": "v1",
                        "sensor_channel": "ppg_green",
                        "release_status": "accepted",
                        "android_ready": True,
                        "source_rate_hz": 25.0,
                        "model_rate_hz": 25.0,
                        "cohort_ids": sorted(validator.PARITY_POOL),
                        "files": files,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    validator.EXPORT_ROOT = tmp_path
    validator.INDEX_PATH = index_path

    with pytest.raises(RuntimeError, match="incomplete or unexpected artifact set"):
        validator.validate()
