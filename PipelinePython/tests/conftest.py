import pytest


@pytest.fixture
def sample_raw_data_path():
    return "data/artifacts/bronze/canonical_signals.csv"


@pytest.fixture
def sample_frozen_gates_path():
    return "PipelinePython/configs/frozen_gates.yaml"


@pytest.fixture
def mock_onnx_model_path():
    return "data/artifacts/export/mock_version/stress_ppg_ir_fp32.onnx"
