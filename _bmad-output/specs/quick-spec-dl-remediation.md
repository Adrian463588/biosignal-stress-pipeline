# Quick Spec: Deep Learning Remediation for PPG_IR and PPG_RED

## 1. Problem Statement
Classical tabular baselines (LightGBM, XGBoost, Random Forest) plateaued below the locked `balanced_accuracy >= 0.80` performance gate for `ppg_ir` and `ppg_red` on the 4-participant untouched holdout set (`ID-29` to `ID-32`).

Hand-crafted HRV features compress 100 Hz raw time-series into summary statistics, losing pulse morphology details. Raw waveform deep learning (1D-CNN, CNN-BiLSTM, TCN) is adopted per ADR-004 to recover this morphological signal without altering the performance gate.

## 2. Capabilities & Architecture
- **Raw Waveform Input**: 180-second windowed raw signals downsampled to 25 Hz (4,500 samples per window).
- **Nested LOSO Cross-Validation**: 28 outer participant folds with 5 inner GroupKFold splits for Optuna hyperparameter optimization.
- **Untouched Holdout**: 4 participants (`ID-29` .. `ID-32`) reserved strictly for final one-time model verification.
- **Model Ladder**:
  1. `1D_CNN`: Multi-block 1D Convolutional Neural Network.
  2. `CNN_BiLSTM`: Convolutional feature extractor + Bidirectional LSTM temporal encoder.
  3. `TCN`: Temporal Convolutional Network with dilated residual blocks.
- **SOLID Abstraction**: `StressClassifier` abstract base class defining `fit`, `predict_proba`, `to`, `save`, and `load` lifecycle contracts.

## 3. Hardware & Execution Constraints
- **Target Device**: NVIDIA GeForce RTX 2060 Laptop GPU (6GB VRAM, Turing Architecture, Tensor Cores).
- **Precision**: PyTorch Automatic Mixed Precision (AMP) using `torch.float16` and `GradScaler("cuda")`. **`bfloat16` is prohibited** due to lack of native hardware support on Turing GPUs.
- **Performance Gate (LOCKED)**: `balanced_accuracy >= 0.80` on 4-participant holdout. Never lowered ad-hoc.
- **ONNX Export Gate**: `max_diff < 1e-5` on 100 deterministic test vectors for Android Dimensity 8400 Ultra deployment.

## 4. Non-Goals
- Do not modify `ppg_green` classical baseline (which already cleared the 0.80 gate).
- Do not edit or lower `PipelinePython/configs/frozen_gates.yaml`.

## 5. Success Signal
- Nested LOSO & Holdout `balanced_accuracy >= 0.80` for `ppg_ir` and `ppg_red`.
- Deterministic numerical parity check between PyTorch `.pt` and exported `.onnx` models.
