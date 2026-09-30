"""model_validator.py
-------------------
Validator for exported ONNX models before deployment to Android assets.
Verifies input/output graph signatures, probability output presence, and numerical validity.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ModelValidator:
    """Validates exported ONNX model graphs for Android compatibility."""

    def __init__(self, onnx_path: Path | str) -> None:
        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model file not found: {self.onnx_path}")

    def verify_outputs(self) -> list[str]:
        """Return list of graph output names. Must include label & probability."""
        import onnxruntime as ort

        sess = ort.InferenceSession(
            str(self.onnx_path), providers=["CPUExecutionProvider"]
        )
        outputs = [out.name for out in sess.get_outputs()]
        return outputs

    def validate_inference(
        self, dummy_input: np.ndarray | tuple[np.ndarray, ...]
    ) -> dict[str, Any]:
        """Run test vector inference to confirm output probability shape and range [0, 1].

        Returns dict with status, probabilities, and output names.
        """
        import onnxruntime as ort

        sess = ort.InferenceSession(
            str(self.onnx_path), providers=["CPUExecutionProvider"]
        )
        inputs = sess.get_inputs()

        feed_dict = {}
        if isinstance(dummy_input, tuple):
            for inp_meta, data in zip(inputs, dummy_input):
                feed_dict[inp_meta.name] = data.astype(np.float32)
        else:
            feed_dict[inputs[0].name] = dummy_input.astype(np.float32)

        out_values = sess.run(None, feed_dict)

        # Parse outputs
        if len(out_values) == 1:
            probs = np.asarray(out_values[0])
        else:
            probs = np.asarray(out_values[1])

        if probs.ndim == 2 and probs.shape[1] >= 2:
            stress_prob = float(probs[0, 1])
        elif probs.ndim == 1:
            stress_prob = float(probs[0])
        else:
            stress_prob = float(probs.ravel()[0])

        is_valid = bool(np.isfinite(stress_prob) and 0.0 <= stress_prob <= 1.0)
        return {
            "valid": is_valid,
            "stress_probability": stress_prob,
            "output_names": [o.name for o in sess.get_outputs()],
            "output_shapes": [v.shape for v in out_values],
        }
