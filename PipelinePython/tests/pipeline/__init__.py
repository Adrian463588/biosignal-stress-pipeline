from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_canonical_index_is_fail_closed_until_parity_evidence_exists() -> None:
    index = json.loads(
        (ROOT / "data" / "artifacts" / "export" / "model_index.json").read_text(
            encoding="utf-8"
        )
    )
    assert index["schema_version"] == "1.0.0"
    if not index["models"]:
        assert index["release_status"].startswith("blocked_")
        return

    for candidate in index["models"]:
        parity_descriptor = candidate["files"]["parity"]
        parity = json.loads(
            (
                ROOT / "data" / "artifacts" / "export" / parity_descriptor["path"]
            ).read_text(encoding="utf-8")
        )
        assert parity.get("max_abs_diff", parity.get("parity_max_diff")) < 1e-5
        assert parity["onnx_checker_passed"] is True
        assert parity["real_batch1_inference_passed"] is True
