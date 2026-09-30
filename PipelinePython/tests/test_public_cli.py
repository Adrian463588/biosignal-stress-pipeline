from __future__ import annotations

from PipelinePython.__main__ import FLOW_RUNNERS, _parser


def test_public_cli_exposes_only_canonical_workflows() -> None:
    parser = _parser()
    parsed = parser.parse_args(
        ["stress-onnx", "--run-id", "20260821T120000Z", "--sensor", "ppg_ir"]
    )

    assert parsed.command == "stress-onnx"
    assert parsed.run_id == "20260821T120000Z"
    assert parsed.sensor == "ppg_ir"
    assert set(FLOW_RUNNERS) == {
        "data",
        "training-gold",
        "peaks",
        "stress-onnx",
        "all",
    }
    assert all(path.is_file() for path in FLOW_RUNNERS.values())


def test_training_gold_command_requires_explicit_source_run() -> None:
    parsed = _parser().parse_args(
        [
            "training-gold",
            "--source-run-id",
            "20260828T213118Z",
            "--run-id",
            "20260828T233731Z",
        ]
    )

    assert parsed.command == "training-gold"
    assert parsed.source_run_id == "20260828T213118Z"


def test_public_cli_defaults_to_pipeline_config() -> None:
    parsed = _parser().parse_args(["data"])

    assert parsed.config.name == "data.yaml"
    assert parsed.config.parent.name == "configs"


def test_live_ppg_capture_verifier_requires_explicit_run_and_capture() -> None:
    parsed = _parser().parse_args(
        [
            "verify-live-ppg-capture",
            "--run-id",
            "20260916T132736Z",
            "--capture-dir",
            "captured-watch-session",
        ]
    )

    assert parsed.command == "verify-live-ppg-capture"
    assert parsed.run_id == "20260916T132736Z"
    assert parsed.capture_dir.name == "captured-watch-session"
