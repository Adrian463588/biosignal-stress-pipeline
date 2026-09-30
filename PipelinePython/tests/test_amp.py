import pytest

torch = pytest.importorskip("torch")
from PipelinePython.common.amp import AMPContext  # noqa: E402


def test_amp_context_instantiation():
    ctx = AMPContext(device_type="cpu", dtype=torch.float32, enabled=False)
    assert ctx.device_type == "cpu"
    assert ctx.enabled is False


def test_amp_rtx2060_bfloat16_protection():
    with pytest.raises(ValueError, match="does not support bfloat16"):
        AMPContext(device_type="cuda", dtype=torch.bfloat16)


def test_amp_autocast_and_scaler():
    ctx = AMPContext(device_type="cpu", dtype=torch.float32, enabled=False)
    assert isinstance(ctx.autocast(), torch.amp.autocast)

    # scaler requires cuda typically for AMP but in PyTorch 2.0+ can be instantiated
    scaler = ctx.get_scaler()
    assert isinstance(scaler, torch.amp.GradScaler)
