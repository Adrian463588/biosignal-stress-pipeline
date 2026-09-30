"""Unit tests for Deep Learning training pipeline, models, and GPU acceleration."""

from pathlib import Path
import numpy as np

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import TensorDataset, DataLoader  # noqa: E402

from PipelinePython.common.models.base import StressClassifier  # noqa: E402
from PipelinePython.common.models.cnn1d import CNN1DModel, TorchCNNClassifier  # noqa: E402
from PipelinePython.common.settings import load_pipeline_config  # noqa: E402
from PipelinePython.common.training_dl_v4 import _setup_cuda_environment  # noqa: E402


def test_device_setup():
    """Verify device setup detects CUDA or falls back to CPU cleanly."""
    device = _setup_cuda_environment()
    assert device in ("cuda", "cpu")
    if torch.cuda.is_available():
        assert device == "cuda"
        assert torch.cuda.get_device_name(0) is not None


def test_cnn1d_model_forward():
    """Test 1D-CNN model output shape for raw waveform tensors."""
    model = CNN1DModel(in_channels=1, base_filters=16)
    # Batch size 4, 1 channel, 4500 time samples
    dummy_input = torch.randn(4, 1, 4500)
    logits = model(dummy_input)
    assert logits.shape == (4, 1) or logits.shape == (4,)


def test_stress_classifier_interface():
    """Verify TorchCNNClassifier implements StressClassifier interface contracts."""
    classifier = TorchCNNClassifier(lr=1e-3)
    assert isinstance(classifier, StressClassifier)

    # Test predict_proba shape
    X_dummy = np.random.randn(8, 4500).astype(np.float32)
    probs = classifier.predict_proba(X_dummy)
    assert isinstance(probs, np.ndarray)
    assert len(probs) == 8
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_amp_fp16_training(tmp_path: Path):
    """Test mixed-precision training loop and model persistence."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = TorchCNNClassifier(device=device)

    # Dummy dataset
    X_tr = torch.randn(16, 1, 4500)
    y_tr = torch.randint(0, 2, (16,)).float()
    ds = TensorDataset(X_tr, y_tr)
    loader = DataLoader(ds, batch_size=4)

    history = classifier.fit(loader, loader, num_epochs=2)
    assert "train_loss" in history and "val_loss" in history
    assert len(history["train_loss"]) == 2

    # Test checkpoint save and load
    ckpt_file = str(tmp_path / "model.pt")
    classifier.save(ckpt_file)
    assert Path(ckpt_file).exists()

    new_classifier = TorchCNNClassifier(device=device)
    new_classifier.load(ckpt_file)
    probs = new_classifier.predict_proba(np.random.randn(4, 4500).astype(np.float32))
    assert len(probs) == 4


def test_locked_performance_gate_config():
    """Verify canonical research and preferred performance gates."""
    config = load_pipeline_config()
    assert config.training.min_balanced_accuracy == 0.80
    assert config.training.minimum_research_balanced_accuracy == 0.70
    assert config.training.preferred_balanced_accuracy == 0.80
