from __future__ import annotations

import pytest

from PipelinePython.__main__ import _parser
from PipelinePython.common.internal_publish import InternalTestEvidenceStore
from PipelinePython.common.workflows.stress_onnx import _published_internal_channels


def test_internal_evidence_claim_is_one_time_and_precedes_publication(tmp_path) -> None:
    store = InternalTestEvidenceStore(tmp_path)
    fingerprint = "a" * 64

    store.claim("ppg_green", fingerprint, {"run_id": "20260916T000000Z"})
    with pytest.raises(RuntimeError, match="access_already_claimed"):
        store.claim("ppg_green", fingerprint, {"run_id": "20260916T000000Z"})

    payload = {"internal_test_status": "GO", "fingerprint": fingerprint}
    assert store.write_once("ppg_green", fingerprint, payload) == payload
    assert store.write_once("ppg_green", fingerprint, payload) == payload
    with pytest.raises(RuntimeError, match="immutability_violation"):
        store.write_once("ppg_green", fingerprint, {"internal_test_status": "NO_GO"})


def test_publish_internal_requires_an_explicit_existing_run_id() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(["publish-internal"])


def test_only_successfully_exported_channels_can_wait_for_device_parity() -> None:
    exported = {
        "ppg_green": {
            "onnx_export_status": "GO",
            "internal_android_status": "PENDING_DEVICE_PARITY",
        },
        "ppg_ir": {"status": "not_published", "reason": "converter_failed"},
        "ppg_red": {"status": "blocked", "reason": "no_candidate"},
    }

    assert _published_internal_channels(exported) == ["ppg_green"]
