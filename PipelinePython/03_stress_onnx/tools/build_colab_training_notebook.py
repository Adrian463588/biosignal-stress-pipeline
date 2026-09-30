"""Generate the canonical waveform-only Google Colab T4 notebook.

The generator is local-only. It writes a reproducible notebook template; it
does not upload raw data, mount Drive, train a model, or publish a release.
Manual Colab execution is conditional on a local classical-ML ``requires_dl``
decision and a checksum-bound waveform handoff.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = (
    ROOT
    / "PipelinePython"
    / "03_stress_onnx"
    / "notebooks"
    / "BiosignalStressTraining_Colab.ipynb"
)


def _cell(cell_type: str, source: str) -> dict[str, Any]:
    normalized = textwrap.dedent(source).strip()
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in normalized.splitlines()],
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
    }


def build_notebook() -> dict[str, Any]:
    """Return an nbformat v4 notebook for manual, development-only DL."""

    cells = [
        _cell(
            "markdown",
            """
            # Conditional PPG waveform DL: 1D-CNN, TCN, or CNN-GRU

            This notebook is a development-only escalation path. Open it only
            when local participant-safe classical ML marks a channel
            `requires_dl=true`. It accepts one Samsung PPG wavelength at a
            time, plus its observed/interpolated masks. ECG, PAT, IBI, motion,
            quality metadata, sibling channels, and holdout participants are
            never model inputs.

            The old `gold_ibi_sequences_v3.parquet` / `ibi_values` /
            `ibi_mask` contract is deliberately rejected. Those names remain
            in this note only so an operator can recognize and remove a stale
            handoff; they are not loaded by this notebook.

            The notebook does not claim Android readiness or clinical validity.
            A successful development run produces a candidate report for local
            holdout governance and independent ONNX verification.
            """,
        ),
        _cell(
            "code",
            """
            import hashlib
            import json
            import os
            import random
            from pathlib import Path

            import numpy as np
            import torch
            from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
            from sklearn.model_selection import GroupKFold
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset

            SEED = 42
            BATCH_SIZE = 64
            MAX_EPOCHS = 40
            EARLY_STOPPING_PATIENCE = 8
            MIN_PARITY_SAMPLES = 100
            ARCHITECTURES = ("1d_cnn", "tcn", "cnn_bilstm_or_gru")
            DEVELOPMENT_PARTICIPANTS = tuple(f"ID-{index:02d}" for index in range(1, 29))
            HOLDOUT_PARTICIPANTS = tuple(f"ID-{index:02d}" for index in range(29, 33))

            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            """,
        ),
        _cell(
            "code",
            """
            # Manual action: select a Google Colab T4 runtime and connect Drive.
            from google.colab import drive

            drive.mount("/content/drive")
            assert torch.cuda.is_available(), (
                "A Colab GPU is required; select Runtime > Change runtime type > T4 GPU."
            )
            GPU_NAME = torch.cuda.get_device_name(0)
            assert "T4" in GPU_NAME.upper(), f"T4 GPU is required, found: {GPU_NAME}"
            DEVICE = torch.device("cuda")
            print(f"Using required Colab GPU: {GPU_NAME}")
            """,
        ),
        _cell(
            "code",
            """
            # Set this to the run ID printed by the canonical local workflow.
            RUN_ID = "REPLACE_WITH_LOCAL_RUN_ID"
            DRIVE_RUN_DIR = Path("/content/drive/MyDrive/BiosignalStress") / RUN_ID
            HANDOFF_PATH = DRIVE_RUN_DIR / "colab_dl_waveform_manifest.json"
            CHECKPOINT_ROOT = DRIVE_RUN_DIR / "dl_checkpoints"
            RESULT_ROOT = DRIVE_RUN_DIR / "dl_results"

            assert RUN_ID != "REPLACE_WITH_LOCAL_RUN_ID", "Set RUN_ID before running training."
            assert HANDOFF_PATH.is_file(), "Copy the checksum-bound waveform handoff first."
            CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
            RESULT_ROOT.mkdir(parents=True, exist_ok=True)
            """,
        ),
        _cell(
            "code",
            """
            def sha256_file(path: Path) -> str:
                digest = hashlib.sha256()
                with path.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
                return digest.hexdigest()


            def json_sha256(value: object) -> str:
                encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
                return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


            def atomic_json_write(destination: Path, payload: dict) -> None:
                temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
                temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
                os.replace(temporary, destination)


            # A legacy IBI handoff is unsafe for this waveform contract.
            LEGACY_GOLD_NAME = "gold_ibi_sequences_v3.parquet"
            LEGACY_VALUE_COLUMN = "ibi_values"
            LEGACY_MASK_COLUMN = "ibi_mask"
            assert not (DRIVE_RUN_DIR / LEGACY_GOLD_NAME).is_file(), (
                "Legacy IBI handoff detected; use colab_dl_waveform_manifest.json."
            )
            """,
        ),
        _cell(
            "code",
            """
            handoff = json.loads(HANDOFF_PATH.read_text(encoding="utf-8"))
            assert handoff["schema_version"] == "colab-dl-waveform-v1"
            assert handoff["run_id"] == RUN_ID
            assert handoff["t4_required"] is True
            assert tuple(handoff["inner_group_splits"] for _ in [0]) == (5,)
            assert tuple(tuple(fold) for fold in handoff["outer_folds"]) == tuple(
                tuple(DEVELOPMENT_PARTICIPANTS[index:index + 4])
                for index in range(0, 28, 4)
            )
            assert set(handoff["dl_channels"]) <= {"ppg_green", "ppg_ir", "ppg_red"}
            assert handoff["dl_channels"], "Local ML did not hand off any channel."
            assert handoff["gold_manifest_path"]
            assert len(handoff["gold_manifest_sha256"]) == 64
            assert len(handoff["source_tree_sha256"]) == 64
            assert len(handoff["config_fingerprint"]) == 64
            assert len(handoff["run_contract_sha256"]) == 64
            FOLD_CONTRACT_SHA256 = json_sha256(
                {
                    "development": list(DEVELOPMENT_PARTICIPANTS),
                    "holdout": list(HOLDOUT_PARTICIPANTS),
                    "outer_folds": handoff["outer_folds"],
                    "inner_group_splits": handoff["inner_group_splits"],
                }
            )
            print("Waveform DL channels:", handoff["dl_channels"])
            """,
        ),
        _cell(
            "code",
            """
            def load_waveform_group(item: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                relative = item.get("colab_relative_path")
                assert isinstance(relative, str) and relative.startswith("gold_waveforms/")
                artifact_path = DRIVE_RUN_DIR / relative
                assert artifact_path.is_file(), f"Missing manually copied Gold waveform: {artifact_path}"
                expected_sha = item.get("sha256")
                assert expected_sha == sha256_file(artifact_path), f"Waveform checksum mismatch: {artifact_path}"
                with np.load(artifact_path, allow_pickle=False) as arrays:
                    names = set(arrays.files)
                    forbidden = sorted(
                        name for name in names
                        if any(token in name.lower() for token in ("ibi", "ecg", "pat", "motion", "tabular"))
                    )
                    assert not forbidden, f"Forbidden waveform artifact fields: {forbidden}"
                    required = {
                        "waveforms", "observed_mask", "interpolated_mask",
                        "phases", "participant_ids", "sampling_rate_hz",
                        "source_sampling_rate_hz", "target_sampling_rate_hz",
                        "window_start_utc", "window_end_utc", "source_checksums",
                        "source_files", "window_ids", "session_ids",
                        "sensor_channels", "quality_tiers",
                        "normalization_mean", "normalization_std",
                    }
                    assert required <= names, f"Waveform artifact missing: {sorted(required - names)}"
                    values = np.asarray(arrays["waveforms"], dtype=np.float32)
                    observed = np.asarray(arrays["observed_mask"], dtype=np.float32)
                    interpolated = np.asarray(arrays["interpolated_mask"], dtype=np.float32)
                    participants = np.asarray(arrays["participant_ids"]).astype(str)
                    phases = np.asarray(arrays["phases"]).astype(str)
                    assert values.ndim == 3 and values.shape[-1] == 1
                    assert observed.shape == values.shape[:2]
                    assert interpolated.shape == values.shape[:2]
                    assert np.isfinite(values).all()
                    assert np.isin(observed, (0.0, 1.0)).all()
                    assert np.isin(interpolated, (0.0, 1.0)).all()
                    assert np.all((observed + interpolated) == 1.0)
                    assert np.all(observed.sum(axis=1) > 0.0)
                    normalization_mean = np.asarray(arrays["normalization_mean"], dtype=np.float32)
                    normalization_std = np.asarray(arrays["normalization_std"], dtype=np.float32)
                    assert normalization_mean.shape == (values.shape[0], 1)
                    assert normalization_std.shape == (values.shape[0], 1)
                    assert np.isfinite(normalization_mean).all()
                    assert np.isfinite(normalization_std).all()
                    assert np.all(normalization_std > 0.0)
                    for lineage_name in (
                        "window_ids", "session_ids", "sensor_channels", "quality_tiers",
                        "source_files", "source_checksums",
                    ):
                        assert np.asarray(arrays[lineage_name]).shape == (values.shape[0],)
                    for rate_name in ("sampling_rate_hz", "source_sampling_rate_hz", "target_sampling_rate_hz"):
                        rates = np.asarray(arrays[rate_name], dtype=np.float64)
                        assert rates.shape == (values.shape[0],) and np.isfinite(rates).all()
                        assert np.all(rates > 0.0)
                assert values.ndim == 3 and values.shape[-1] == 1
                assert observed.shape == values.shape[:2]
                assert interpolated.shape == values.shape[:2]
                assert len(participants) == len(phases) == values.shape[0]
                assert np.isfinite(values).all()
                assert np.isin(observed, (0.0, 1.0)).all()
                assert np.isin(interpolated, (0.0, 1.0)).all()
                assert np.all(observed.sum(axis=1) > 0.0)
                assert set(participants) <= set(DEVELOPMENT_PARTICIPANTS)
                assert set(phases) <= {"relax", "task"}
                return values, observed, interpolated, participants, phases


            def channel_tensors(channel: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
                entries = [
                    item for item in handoff["waveform_artifacts"]
                    if item.get("sensor_channel") == channel
                    and item.get("participant_id") in DEVELOPMENT_PARTICIPANTS
                ]
                expected = {
                    (participant, phase, channel)
                    for participant in DEVELOPMENT_PARTICIPANTS
                    for phase in ("relax", "task")
                }
                observed_groups = {
                    (str(item.get("participant_id")), str(item.get("phase")), channel)
                    for item in entries
                }
                assert observed_groups == expected, "DL handoff is missing a development group."
                values, masks, interpolated, labels, participants, phases = [], [], [], [], [], []
                sample_count = None
                for item in sorted(entries, key=lambda value: (value["participant_id"], value["phase"])):
                    group_values, group_observed, group_interpolated, group_participants, group_phases = load_waveform_group(item)
                    assert set(group_participants) == {item["participant_id"]}
                    assert set(group_phases) == {item["phase"]}
                    if sample_count is None:
                        sample_count = group_values.shape[1]
                    assert group_values.shape[1] == sample_count, "Fixed waveform shape contract mismatch."
                    values.append(group_values)
                    masks.append(group_observed)
                    interpolated.append(group_interpolated)
                    participants.extend(group_participants.tolist())
                    phases.extend(group_phases.tolist())
                    labels.extend([0 if phase == "relax" else 1 for phase in group_phases])
                joined_values = np.concatenate(values, axis=0)
                joined_masks = np.concatenate(masks, axis=0)
                joined_interpolated = np.concatenate(interpolated, axis=0)
                assert not set(participants) & set(HOLDOUT_PARTICIPANTS)
                # Conv1d uses [batch, channel, samples]; the model still sees one PPG wavelength.
                return (
                    torch.from_numpy(np.transpose(joined_values, (0, 2, 1))),
                    torch.from_numpy(joined_masks[:, None, :]),
                    torch.tensor(labels, dtype=torch.long),
                    np.asarray(participants, dtype=str),
                    np.asarray(phases, dtype=str),
                )
            """,
        ),
        _cell(
            "code",
            """
            def masked_input(waveform: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
                return waveform * observed_mask


            class OneDCNN(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.network = nn.Sequential(
                        nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.ReLU(),
                        nn.MaxPool1d(2), nn.Conv1d(32, 64, kernel_size=5, padding=2),
                        nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, 2),
                    )

                def forward(self, waveform: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
                    return self.network(masked_input(waveform, observed_mask))


            class TcnBlock(nn.Module):
                def __init__(self, channels: int, dilation: int) -> None:
                    super().__init__()
                    self.block = nn.Sequential(
                        nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
                        nn.ReLU(),
                        nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
                    )
                    self.activation = nn.ReLU()

                def forward(self, value: torch.Tensor) -> torch.Tensor:
                    return self.activation(value + self.block(value))


            class TCN(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.stem = nn.Conv1d(1, 32, 1)
                    self.blocks = nn.Sequential(
                        TcnBlock(32, 1), TcnBlock(32, 2), TcnBlock(32, 4)
                    )
                    self.head = nn.Sequential(
                        nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(32, 2)
                    )

                def forward(self, waveform: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
                    value = masked_input(waveform, observed_mask)
                    return self.head(self.blocks(self.stem(value)))


            class CNNGRU(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.stem = nn.Sequential(
                        nn.Conv1d(1, 32, 7, padding=3), nn.ReLU(), nn.MaxPool1d(4)
                    )
                    self.gru = nn.GRU(32, 32, batch_first=True, bidirectional=True)
                    self.head = nn.Linear(64, 2)

                def forward(self, waveform: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
                    value = self.stem(masked_input(waveform, observed_mask)).transpose(1, 2)
                    sequence, _ = self.gru(value)
                    return self.head(sequence[:, -1, :])


            def create_model(architecture: str) -> nn.Module:
                if architecture == "1d_cnn":
                    return OneDCNN()
                if architecture == "tcn":
                    return TCN()
                if architecture == "cnn_bilstm_or_gru":
                    return CNNGRU()
                raise ValueError(f"Unsupported architecture: {architecture}")
            """,
        ),
        _cell(
            "code",
            """
            def participant_phase_weights(participants: np.ndarray, phases: np.ndarray) -> torch.Tensor:
                keys = np.asarray([f"{participant}|{phase}" for participant, phase in zip(participants, phases, strict=True)])
                unique, counts = np.unique(keys, return_counts=True)
                count_map = dict(zip(unique.tolist(), counts.tolist(), strict=True))
                return torch.tensor([1.0 / count_map[key] for key in keys], dtype=torch.float32)


            def grouped_metrics(labels: np.ndarray, probabilities: np.ndarray, participants: np.ndarray, phases: np.ndarray) -> dict:
                rows = {}
                for label, probability, participant, phase in zip(labels, probabilities, participants, phases, strict=True):
                    key = (str(participant), str(phase))
                    rows.setdefault(key, {"label": int(label), "probabilities": []})["probabilities"].append(float(probability))
                y_true = np.asarray([row["label"] for row in rows.values()], dtype=np.int64)
                y_probability = np.asarray([np.median(row["probabilities"]) for row in rows.values()], dtype=np.float64)
                y_prediction = (y_probability >= 0.5).astype(np.int64)
                metrics = {
                    "balanced_accuracy": float(balanced_accuracy_score(y_true, y_prediction)),
                    "f1": float(f1_score(y_true, y_prediction, zero_division=0)),
                    "session_count": int(len(y_true)),
                }
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_probability)) if len(np.unique(y_true)) == 2 else float("nan")
                return metrics


            def loss_for_batch(model: nn.Module, waveform: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
                logits = model(waveform.to(DEVICE), mask.to(DEVICE))
                losses = nn.functional.cross_entropy(logits, labels.to(DEVICE), reduction="none")
                return (losses * weights.to(DEVICE)).sum() / weights.to(DEVICE).sum()


            def fit_epochs(model: nn.Module, train_indices: np.ndarray, validation_indices: np.ndarray, tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray], epochs: int) -> tuple[nn.Module, float, int]:
                waveform, mask, labels, participants, phases = tensors
                weights = participant_phase_weights(participants[train_indices], phases[train_indices])
                loader = DataLoader(
                    TensorDataset(waveform[train_indices], mask[train_indices], labels[train_indices], weights),
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                )
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                best_score = (float("-inf"), float("-inf"), float("-inf"))
                best_state = None
                stale = 0
                epochs_used = 0
                for _ in range(epochs):
                    epochs_used += 1
                    model.train()
                    for batch_waveform, batch_mask, batch_labels, batch_weights in loader:
                        optimizer.zero_grad(set_to_none=True)
                        loss = loss_for_batch(model, batch_waveform, batch_mask, batch_labels, batch_weights)
                        loss.backward()
                        optimizer.step()
                    model.eval()
                    with torch.no_grad():
                        validation_probability = torch.softmax(
                            model(
                                waveform[validation_indices].to(DEVICE),
                                mask[validation_indices].to(DEVICE),
                            ),
                            dim=1,
                        )[:, 1].cpu().numpy()
                    validation_metrics = grouped_metrics(
                        labels[validation_indices].numpy(),
                        validation_probability,
                        participants[validation_indices],
                        phases[validation_indices],
                    )
                    validation_score = tuple(
                        float(validation_metrics.get(name, float("-inf")))
                        for name in ("balanced_accuracy", "f1", "roc_auc")
                    )
                    validation_score = tuple(
                        value if np.isfinite(value) else float("-inf")
                        for value in validation_score
                    )
                    if validation_score > best_score:
                        best_score = validation_score
                        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                        stale = 0
                    else:
                        stale += 1
                    if stale >= EARLY_STOPPING_PATIENCE:
                        break
                assert best_state is not None
                model.load_state_dict(best_state)
                return model, best_score[0], epochs_used


            def fit_fixed_epochs(
                model: nn.Module,
                optimizer: torch.optim.Optimizer,
                train_indices: np.ndarray,
                tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray],
                epochs: int,
                start_epoch: int,
                checkpoint_callback,
            ) -> tuple[nn.Module, torch.optim.Optimizer]:
                waveform, mask, labels, participants, phases = tensors
                weights = participant_phase_weights(participants[train_indices], phases[train_indices])
                loader = DataLoader(
                    TensorDataset(waveform[train_indices], mask[train_indices], labels[train_indices], weights),
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                )
                for epoch in range(start_epoch, epochs):
                    model.train()
                    for batch_waveform, batch_mask, batch_labels, batch_weights in loader:
                        optimizer.zero_grad(set_to_none=True)
                        loss = loss_for_batch(model, batch_waveform, batch_mask, batch_labels, batch_weights)
                        loss.backward()
                        optimizer.step()
                    if checkpoint_callback is not None:
                        checkpoint_callback(epoch + 1, model, optimizer)
                return model, optimizer
            """,
        ),
        _cell(
            "code",
            """
            def checkpoint_paths(channel: str, architecture: str) -> tuple[Path, Path]:
                directory = CHECKPOINT_ROOT / channel / architecture
                directory.mkdir(parents=True, exist_ok=True)
                return directory / "latest_checkpoint.pt", directory / "resume_manifest.json"


            def checkpoint_contract(channel: str, architecture: str) -> dict:
                return {
                    "schema_version": handoff["schema_version"],
                    "run_id": RUN_ID,
                    "gold_manifest_sha256": handoff["gold_manifest_sha256"],
                    "training_config_sha256": handoff["training_config_sha256"],
                    "source_tree_sha256": handoff["source_tree_sha256"],
                    "config_fingerprint": handoff["config_fingerprint"],
                    "run_contract_sha256": handoff["run_contract_sha256"],
                    "fold_contract_sha256": FOLD_CONTRACT_SHA256,
                    "channel": channel,
                    "architecture": architecture,
                }


            def save_checkpoint(channel: str, architecture: str, model: nn.Module, optimizer: torch.optim.Optimizer, state: dict) -> None:
                checkpoint_path, resume_path = checkpoint_paths(channel, architecture)
                temporary = checkpoint_path.with_suffix(".pt.tmp")
                torch.save(
                    {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "state": state},
                    temporary,
                )
                os.replace(temporary, checkpoint_path)
                state["checkpoint_sha256"] = sha256_file(checkpoint_path)
                atomic_json_write(resume_path, state)


            def validate_resume(state: dict, channel: str, architecture: str, checkpoint_path: Path) -> None:
                for key, value in checkpoint_contract(channel, architecture).items():
                    assert state.get(key) == value, f"Stale resume manifest: {key}"
                assert set(state.get("completed_outer_folds", [])) <= set(range(7))
                assert state.get("checkpoint_sha256") == sha256_file(checkpoint_path)


            def initial_state(channel: str, architecture: str) -> dict:
                return {
                    **checkpoint_contract(channel, architecture),
                    "completed_outer_folds": [],
                    "active_fold": None,
                    "active_epoch": 0,
                }
            """,
        ),
        _cell(
            "code",
            """
            def split_outer_fold_indices(participants: np.ndarray, validation_participants: set[str]) -> tuple[np.ndarray, np.ndarray]:
                is_validation = np.isin(participants, list(validation_participants))
                test_indices = np.flatnonzero(is_validation)
                outer_train_indices = np.flatnonzero(~is_validation)
                assert set(participants[test_indices]) <= validation_participants
                assert not set(participants[outer_train_indices]) & validation_participants
                assert len(test_indices) + len(outer_train_indices) == len(participants)
                return outer_train_indices, test_indices


            def train_outer_fold(channel: str, architecture: str, tensors: tuple, fold_index: int, state: dict) -> dict:
                waveform, mask, labels, participants, phases = tensors
                validation_participants = set(handoff["outer_folds"][fold_index])
                outer_train_indices, test_indices = split_outer_fold_indices(
                    participants, validation_participants
                )
                inner = GroupKFold(n_splits=5)
                inner_groups = participants[outer_train_indices]
                epoch_candidates = []
                for inner_train_relative, inner_validation_relative in inner.split(
                    outer_train_indices, labels[outer_train_indices].numpy(), inner_groups
                ):
                    inner_train = outer_train_indices[inner_train_relative]
                    inner_validation = outer_train_indices[inner_validation_relative]
                    model, _, epochs_used = fit_epochs(
                        create_model(architecture).to(DEVICE),
                        inner_train,
                        inner_validation,
                        tensors,
                        MAX_EPOCHS,
                    )
                    del model
                    epoch_candidates.append(epochs_used)
                selected_epochs = int(np.median(epoch_candidates))
                model = create_model(architecture).to(DEVICE)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                checkpoint_path, _ = checkpoint_paths(channel, architecture)
                start_epoch = 0
                if state.get("active_fold") == fold_index and checkpoint_path.is_file():
                    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
                    model.load_state_dict(checkpoint["model_state"])
                    optimizer.load_state_dict(checkpoint["optimizer_state"])
                    start_epoch = int(state.get("active_epoch", 0))

                def checkpoint_epoch(epoch: int, current_model: nn.Module, current_optimizer: torch.optim.Optimizer) -> None:
                    state["active_epoch"] = epoch
                    save_checkpoint(channel, architecture, current_model, current_optimizer, state)

                try:
                    # Checkpoints are written every epoch, atomically. Inner GroupKFold(5) determined the epoch budget; outer test participants remain unopened during fitting.
                    model, optimizer = fit_fixed_epochs(
                        model,
                        optimizer,
                        outer_train_indices,
                        tensors,
                        selected_epochs,
                        start_epoch,
                        checkpoint_epoch,
                    )
                    model.eval()
                    with torch.no_grad():
                        probabilities = torch.softmax(
                            model(waveform[test_indices].to(DEVICE), mask[test_indices].to(DEVICE)), dim=1
                        )[:, 1].cpu().numpy()
                    result = grouped_metrics(
                        labels[test_indices].numpy(), probabilities, participants[test_indices], phases[test_indices]
                    )
                    result.update({"outer_fold": fold_index, "participants": sorted(validation_participants), "epoch_budget": selected_epochs})
                    state["completed_outer_folds"].append(fold_index)
                    state.setdefault("outer_fold_results", []).append(result)
                    state.update({"active_fold": None, "active_epoch": 0})
                    save_checkpoint(channel, architecture, model, optimizer, state)
                    return result
                except KeyboardInterrupt:
                    save_checkpoint(channel, architecture, model, optimizer, state)
                    raise
                except Exception:
                    save_checkpoint(channel, architecture, model, optimizer, state)
                    raise


            def run_development(channel: str, architecture: str) -> dict:
                tensors = channel_tensors(channel)
                checkpoint_path, resume_path = checkpoint_paths(channel, architecture)
                state = initial_state(channel, architecture)
                results = list(state.get("outer_fold_results", []))
                if resume_path.is_file() or checkpoint_path.is_file():
                    assert resume_path.is_file() and checkpoint_path.is_file(), "Partial checkpoint pair is not resumable."
                    state = json.loads(resume_path.read_text(encoding="utf-8"))
                    validate_resume(state, channel, architecture, checkpoint_path)
                for fold_index in range(7):
                    if fold_index in state["completed_outer_folds"]:
                        continue
                    state.update({"active_fold": fold_index, "active_epoch": 0})
                    result = train_outer_fold(channel, architecture, tensors, fold_index, state)
                    results.append(result)
                    # fold_index % 5 == 0 is retained as a zero-based fold marker.
                    if (fold_index + 1) % 5 == 0:
                        atomic_json_write(RESULT_ROOT / f"{channel}_{architecture}_fold_{fold_index:02d}.json", {"results": results, "state": state})
                payload = {"channel": channel, "architecture": architecture, "results": results, "holdout_opened": False}
                atomic_json_write(RESULT_ROOT / f"{channel}_{architecture}_development.json", payload)
                return payload


            development_results = {
                f"{channel}:{architecture}": run_development(channel, architecture)
                for channel in handoff["dl_channels"]
                for architecture in ARCHITECTURES
            }
            atomic_json_write(RESULT_ROOT / "development_waveform_dl_results.json", development_results)
            print("Development waveform DL complete; holdout remains unopened.")


            def architecture_score(payload: dict) -> tuple[float, float, float, int]:
                rows = payload.get("results", [])
                if not rows:
                    return (float("-inf"), float("-inf"), float("-inf"), 0)
                balanced = float(np.nanmean([row["balanced_accuracy"] for row in rows]))
                f1 = float(np.nanmean([row["f1"] for row in rows]))
                auc = float(np.nanmean([row["roc_auc"] for row in rows]))
                simplicity = ARCHITECTURES.index(payload["architecture"])
                return balanced, f1, auc, -simplicity


            def best_architecture(channel: str) -> str:
                candidates = {
                    architecture: development_results[f"{channel}:{architecture}"]
                    for architecture in ARCHITECTURES
                }
                return max(candidates, key=lambda architecture: architecture_score(candidates[architecture]))


            def fit_full_development(channel: str, architecture: str) -> nn.Module:
                tensors = channel_tensors(channel)
                waveform, _, _, _, _ = tensors
                payload = development_results[f"{channel}:{architecture}"]
                epoch_budget = int(np.median([row["epoch_budget"] for row in payload["results"]]))
                model = create_model(architecture).to(DEVICE)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                all_indices = np.arange(len(waveform), dtype=np.int64)
                model, _ = fit_fixed_epochs(model, optimizer, all_indices, tensors, epoch_budget, 0, None)
                final_path = CHECKPOINT_ROOT / channel / architecture / "final_development_model.pt"
                temporary = final_path.with_suffix(".pt.tmp")
                torch.save({"model_state": model.state_dict(), "epoch_budget": epoch_budget}, temporary)
                os.replace(temporary, final_path)
                return model.eval()
            """,
        ),
        _cell(
            "code",
            """
            # Independent ONNX parity for a manually selected development candidate.
            # This cell never publishes an accepted Android model.
            import onnx
            import onnxruntime as ort

            SELECTED_CHANNEL = handoff["dl_channels"][0]
            SELECTED_ARCHITECTURE = best_architecture(SELECTED_CHANNEL)
            tensors = channel_tensors(SELECTED_CHANNEL)
            waveform, observed_mask, labels, participants, phases = tensors
            model = fit_full_development(SELECTED_CHANNEL, SELECTED_ARCHITECTURE)


            class ProbabilityWrapper(nn.Module):
                def __init__(self, classifier: nn.Module) -> None:
                    super().__init__()
                    self.classifier = classifier

                def forward(self, input_waveform: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
                    return torch.softmax(self.classifier(input_waveform, input_mask), dim=1)[:, 1]


            wrapper = ProbabilityWrapper(model).eval()
            parity_count = min(len(waveform), MIN_PARITY_SAMPLES)
            assert parity_count >= MIN_PARITY_SAMPLES, "At least 100 real development windows are required."
            parity_waveform = waveform[:parity_count].to(DEVICE)
            parity_mask = observed_mask[:parity_count].to(DEVICE)
            with torch.no_grad():
                expected_probability = wrapper(parity_waveform, parity_mask).cpu().numpy().astype(np.float32)
            onnx_path = RESULT_ROOT / f"stress_{SELECTED_CHANNEL}_{SELECTED_ARCHITECTURE}.onnx"
            temporary_onnx = onnx_path.with_suffix(".onnx.tmp")
            export_kwargs = {
                "f": temporary_onnx,
                "args": (parity_waveform[:1], parity_mask[:1]),
                "input_names": ["waveform", "observed_mask"],
                "output_names": ["stress_probability"],
                "opset_version": 17,
                "do_constant_folding": True,
            }
            try:
                torch.onnx.export(wrapper, dynamo=True, **export_kwargs)
            except TypeError:
                torch.onnx.export(wrapper, **export_kwargs)
            os.replace(temporary_onnx, onnx_path)
            onnx.checker.check_model(str(onnx_path))
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            actual_probability = np.concatenate(
                [
                    session.run(
                        ["stress_probability"],
                        {
                            "waveform": parity_waveform[index:index + 1].cpu().numpy(),
                            "observed_mask": parity_mask[index:index + 1].cpu().numpy(),
                        },
                    )[0].reshape(-1)
                    for index in range(parity_count)
                ]
            ).astype(np.float32)
            max_abs_diff = float(np.max(np.abs(expected_probability.astype(np.float64) - actual_probability.astype(np.float64))))
            assert max_abs_diff < 1e-5, f"ONNX/native parity failed: {max_abs_diff}"
            parity_manifest = {
                "schema_version": "onnx-waveform-development-v1",
                "run_id": RUN_ID,
                "sensor_channel": SELECTED_CHANNEL,
                "architecture": SELECTED_ARCHITECTURE,
                "native_model_path": str(
                    CHECKPOINT_ROOT
                    / SELECTED_CHANNEL
                    / SELECTED_ARCHITECTURE
                    / "final_development_model.pt"
                ),
                "native_model_sha256": sha256_file(
                    CHECKPOINT_ROOT
                    / SELECTED_CHANNEL
                    / SELECTED_ARCHITECTURE
                    / "final_development_model.pt"
                ),
                "development_metrics": development_results[
                    f"{SELECTED_CHANNEL}:{SELECTED_ARCHITECTURE}"
                ],
                "gold_manifest_sha256": handoff["gold_manifest_sha256"],
                "source_tree_sha256": handoff["source_tree_sha256"],
                "config_fingerprint": handoff["config_fingerprint"],
                "run_contract_sha256": handoff["run_contract_sha256"],
                "fold_contract_sha256": FOLD_CONTRACT_SHA256,
                "input_names": ["waveform", "observed_mask"],
                "output_name": "stress_probability",
                "parity_checked_real_development_windows": parity_count,
                "onnx_sha256": sha256_file(onnx_path),
                "max_abs_diff": max_abs_diff,
                "onnx_checker_passed": True,
                "holdout_opened": False,
                "release_status": "development_candidate_holdout_pending",
                "clinical_validation_status": "external_validation_required",
            }
            atomic_json_write(RESULT_ROOT / "waveform_onnx_development_manifest.json", parity_manifest)
            print("Independent native/ONNX parity passed:", max_abs_diff)
            """,
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(destination: Path) -> Path:
    """Write a deterministic notebook artifact without executing it."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(build_notebook(), indent=2) + "\n", encoding="utf-8"
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    output = write_notebook(arguments.output)
    print(output)


if __name__ == "__main__":
    main()
