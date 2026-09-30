"""Generate real Python-to-Android parity vectors from canonical Gold windows."""

from __future__ import annotations

import hashlib
import json
import struct
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd

from .contracts import (
    MODEL_PHASES,
    TRAINING_QUALITY_TIERS,
    VALID_PARTICIPANTS,
    assert_feature_tensor,
)
from .features_v3 import extract_ppg_features as extract_ppg_features_v3
from .features_v4 import extract_ppg_features as extract_ppg_features_v4
from .feature_policy import EXPERIMENTAL_FEATURES
from .peak_detection import PpgPeakDetector
from .settings import PipelineConfig, TRAINING_CHANNELS, pipeline_paths_from_config


# Android Home and Verbatim acquire only these channels. Red remains a valid
# research channel, but it must not accidentally become a production fallback
# before the Watch can acquire it and its Android parity is separately proven.
ANDROID_RUNTIME_CHANNELS = frozenset({"ppg_green", "ppg_ir"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


_RECEIPT_HASH_FIELDS = (
    "actual_sensor_acquisition",
    "apk_sha256",
    "device_fingerprint",
    "max_abs_diff",
    "model_version",
    "onnx_sha256",
    "schema_version",
    "sensor_channel",
    "timestamp_utc",
    "vector_count",
)


def _receipt_sha256(receipt: dict[str, Any]) -> str:
    """Return the Android/Python-stable receipt digest.

    JSON encoders are not a safe cross-language signature format: Java and
    Python render the same IEEE-754 value with different exponent spellings.
    This length-prefixed payload uses the identical hexadecimal floating-point
    representation on both runtimes while retaining every security-relevant
    receipt field.
    """

    missing = sorted(set(_RECEIPT_HASH_FIELDS) - set(receipt))
    if missing:
        raise RuntimeError(f"DeviceParityReceipt missing hash fields: {missing}")

    digest = hashlib.sha256()
    for key in _RECEIPT_HASH_FIELDS:
        value = receipt[key]
        if isinstance(value, bool):
            encoded = str(value).lower().encode("utf-8")
        elif isinstance(value, int):
            encoded = str(value).encode("utf-8")
        elif isinstance(value, float):
            encoded = struct.pack(">d", value).hex().encode("ascii")
        elif isinstance(value, str):
            encoded = value.encode("utf-8")
        else:
            raise RuntimeError(
                f"DeviceParityReceipt field {key} has unsupported type "
                f"{type(value).__name__}"
            )
        digest.update(f"{key}:{len(encoded)}:".encode("ascii"))
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


def _assert_vector_contract(
    cases: list[dict[str, Any]], *, valid_count: int, rejection_count: int
) -> None:
    if len(cases) != 100 or valid_count + rejection_count != 100:
        raise RuntimeError("Android parity requires exactly 100 vectors")
    identifiers = [str(case.get("case_id", "")) for case in cases]
    if (
        any(not identifier for identifier in identifiers)
        or len(set(identifiers)) != 100
    ):
        raise RuntimeError("Android parity case_id values must be unique and non-empty")
    observed_valid = sum(case.get("kind") == "valid_raw_window" for case in cases)
    if observed_valid != valid_count:
        raise RuntimeError(
            f"Android parity requires exactly {valid_count} real valid vectors"
        )
    if len(cases) - observed_valid != rejection_count:
        raise RuntimeError(
            f"Android parity requires exactly {rejection_count} rejection vectors"
        )


class DeviceParityReceiptVerifier:
    """Verify content-addressed physical-device parity evidence fail-closed."""

    _REQUIRED_FIELDS = {
        "schema_version",
        "device_fingerprint",
        "apk_sha256",
        "onnx_sha256",
        "model_version",
        "sensor_channel",
        "vector_count",
        "max_abs_diff",
        "timestamp_utc",
        "actual_sensor_acquisition",
        "receipt_sha256",
    }

    def __init__(self, *, max_abs_diff: float) -> None:
        self.max_abs_diff = float(max_abs_diff)

    def verify(
        self,
        receipt: dict[str, Any],
        *,
        manifest: dict[str, Any],
        model_path: Path,
        apk_path: Path,
    ) -> dict[str, Any]:
        missing = sorted(self._REQUIRED_FIELDS - set(receipt))
        if missing:
            raise RuntimeError(f"DeviceParityReceipt missing fields: {missing}")
        if receipt.get("receipt_sha256") != _receipt_sha256(receipt):
            raise RuntimeError("DeviceParityReceipt receipt_sha256 mismatch")
        if not str(receipt.get("device_fingerprint", "")).strip():
            raise RuntimeError("DeviceParityReceipt device_fingerprint is empty")
        try:
            timestamp = datetime.fromisoformat(
                str(receipt["timestamp_utc"]).replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise RuntimeError("DeviceParityReceipt timestamp_utc is invalid") from exc
        if timestamp.tzinfo is None:
            raise RuntimeError(
                "DeviceParityReceipt timestamp_utc must be timezone-aware"
            )
        if int(receipt["vector_count"]) != 100:
            raise RuntimeError("DeviceParityReceipt vector_count must equal 100")
        if not 0.0 <= float(receipt["max_abs_diff"]) < self.max_abs_diff:
            raise RuntimeError(
                "DeviceParityReceipt max_abs_diff failed the parity gate"
            )
        if str(receipt["sensor_channel"]) not in TRAINING_CHANNELS:
            raise RuntimeError("DeviceParityReceipt sensor_channel is unsupported")
        if str(receipt["sensor_channel"]) != str(manifest.get("sensor_channel")):
            raise RuntimeError("DeviceParityReceipt channel does not match manifest")
        if str(receipt["model_version"]) != str(manifest.get("model_version")):
            raise RuntimeError(
                "DeviceParityReceipt model_version does not match manifest"
            )
        if not model_path.is_file() or _sha256(model_path) != receipt["onnx_sha256"]:
            raise RuntimeError("DeviceParityReceipt ONNX checksum mismatch")
        if receipt["onnx_sha256"] != manifest.get("onnx_sha256"):
            raise RuntimeError("DeviceParityReceipt ONNX does not match manifest")
        if not apk_path.is_file() or _sha256(apk_path) != receipt["apk_sha256"]:
            raise RuntimeError("DeviceParityReceipt APK checksum mismatch")
        if (
            receipt["sensor_channel"] == "ppg_red"
            and receipt.get("actual_sensor_acquisition") is not True
        ):
            raise RuntimeError(
                "Red production promotion requires evidence of actual Red acquisition"
            )
        return dict(receipt)


class AndroidVectorGenerator:
    """Create 94 real valid vectors and six deterministic rejection vectors."""

    def __init__(self, project_root: Path, config: PipelineConfig) -> None:
        self.root = project_root.resolve()
        self.config = config
        self.paths = pipeline_paths_from_config(self.root, config)
        self._peak_detector = PpgPeakDetector(config.peak_detection)
        self.export_dir = self.paths.export_dir
        self.output_path = (
            self.root / "tests" / "fixtures" / "android_parity" / "test_vectors.json"
        )

    def _load_gold(
        self, sensor_channel: str, model_manifest: dict[str, Any]
    ) -> tuple[pd.DataFrame, Path]:
        source = self.paths.gold_features_path.resolve()
        if self.root not in source.parents:
            raise RuntimeError(f"Configured Gold source escapes project root: {source}")
        if not source.is_file():
            raise FileNotFoundError(
                "Configured Gold feature table is missing; select an immutable run "
                f"overlay instead of falling back to legacy Gold: {source}"
            )
        frame = (
            pd.read_parquet(source)
            if source.suffix == ".parquet"
            else pd.read_csv(source)
        )
        if "participant_id" not in frame and "participant_code" in frame:
            frame = frame.rename(columns={"participant_code": "participant_id"})
        required = {
            "participant_id",
            "phase",
            "sensor_channel",
            "window_id",
            "window_start_utc",
            "window_end_utc",
            "sampling_rate_hz",
            "source_file",
            "source_checksum",
            "source_silver_path",
            "source_silver_sha256",
            "quality_tier",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise RuntimeError(f"Gold source is missing lineage columns: {missing}")
        if frame.empty:
            raise RuntimeError("Gold source is empty")
        participants = set(frame["participant_id"].dropna().astype(str))
        if not participants <= VALID_PARTICIPANTS:
            raise RuntimeError(
                f"Gold source contains unknown participants: "
                f"{sorted(participants - VALID_PARTICIPANTS)}"
            )
        phases = set(frame["phase"].dropna().astype(str))
        if not phases <= MODEL_PHASES:
            raise RuntimeError(
                f"Android vectors require relax/task only, found: {sorted(phases)}"
            )
        channels = set(frame["sensor_channel"].dropna().astype(str))
        if not channels <= {"ppg_green", "ppg_ir", "ppg_red"}:
            raise RuntimeError(
                f"Gold source contains unsupported channels: {sorted(channels)}"
            )
        if sensor_channel not in channels:
            raise RuntimeError(f"Gold source has no rows for {sensor_channel}")
        checksums = frame["source_checksum"].astype("string")
        if (
            checksums.isna().any()
            or not checksums.str.fullmatch(r"[0-9a-fA-F]{64}").all()
        ):
            raise RuntimeError("Gold source has invalid source checksums")
        source_files = frame["source_file"].astype("string")
        if source_files.isna().any() or source_files.str.strip().eq("").any():
            raise RuntimeError("Gold source has empty source_file lineage")
        silver_paths = frame["source_silver_path"].astype("string")
        if silver_paths.isna().any() or silver_paths.str.strip().eq("").any():
            raise RuntimeError("Gold source has empty source_silver_path lineage")
        silver_checksums = frame["source_silver_sha256"].astype("string")
        if (
            silver_checksums.isna().any()
            or not silver_checksums.str.fullmatch(r"[0-9a-fA-F]{64}").all()
        ):
            raise RuntimeError("Gold source has invalid source_silver_sha256")
        if frame["window_id"].astype("string").str.strip().eq("").any():
            raise RuntimeError("Gold source has empty window_id lineage")
        sampling_rates = pd.to_numeric(frame["sampling_rate_hz"], errors="coerce")
        if (
            sampling_rates.isna().any()
            or not np.isfinite(sampling_rates).all()
            or (sampling_rates <= 0).any()
        ):
            raise RuntimeError("Gold source has invalid sampling_rate_hz values")
        starts = pd.to_datetime(frame["window_start_utc"], utc=True, errors="coerce")
        ends = pd.to_datetime(frame["window_end_utc"], utc=True, errors="coerce")
        if starts.isna().any() or ends.isna().any() or not (ends > starts).all():
            raise RuntimeError("Gold source has invalid window timestamps")
        expected_preprocessing = model_manifest.get("preprocessing_version")
        expected_features = model_manifest.get("feature_version")
        for column, expected in (
            ("preprocessing_version", expected_preprocessing),
            ("feature_version", expected_features),
        ):
            if (
                expected
                and column in frame
                and set(frame[column].dropna().astype(str)) != {str(expected)}
            ):
                raise RuntimeError(f"Gold {column} does not match the model manifest")
        if "training_eligible" in frame:
            eligible = frame[frame["sensor_channel"] == sensor_channel]
            if (
                eligible.empty
                or not eligible["quality_tier"]
                .astype(str)
                .isin(TRAINING_QUALITY_TIERS)
                .any()
            ):
                raise RuntimeError(
                    f"Gold source has no training-quality {sensor_channel} windows"
                )
        return frame, source

    def _load_window(
        self,
        row: pd.Series,
        cache: dict[Path, pd.DataFrame],
        model_manifest: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Load exactly the Silver waveform that produced one Gold row.

        ``source_file`` is the immutable raw-provenance field, not a permitted
        input to Android parity generation.  The vector must be derived from
        ``source_silver_path`` so a missing clean waveform fails explicitly
        rather than silently treating raw CSV as canonical preprocessing.
        """

        source_value = str(row.get("source_silver_path", "")).strip()
        if not source_value:
            raise RuntimeError("Gold window is missing canonical source_silver_path")
        source = Path(source_value)
        if not source.is_absolute():
            source = self.root / source
        source = source.resolve()
        if self.root not in source.parents:
            raise RuntimeError(f"Window source escapes project root: {source}")
        if not source.is_file():
            raise FileNotFoundError(f"Canonical Silver window is missing: {source}")
        expected_checksum = str(row.get("source_silver_sha256", "")).strip()
        if not expected_checksum:
            raise RuntimeError("Gold window is missing canonical source_silver_sha256")
        if _sha256(source) != expected_checksum:
            raise RuntimeError("Canonical Silver window checksum mismatch")
        if source not in cache:
            if source.suffix.lower() != ".parquet":
                raise RuntimeError(
                    "Canonical Silver waveform must be a Parquet artifact"
                )
            cache[source] = pd.read_parquet(source)
        frame = cache[source]
        required_signal_columns = {"clean_value", "raw_value"}
        missing_signal_columns = sorted(required_signal_columns - set(frame.columns))
        if missing_signal_columns:
            raise RuntimeError(
                "Canonical Silver window is missing required signal columns: "
                f"{missing_signal_columns}; clean_value cannot fall back to raw_value"
            )
        timestamp_column = (
            "timestamp_corrected_utc"
            if "timestamp_corrected_utc" in frame
            else "timestamp_raw_utc"
        )
        if timestamp_column not in frame:
            raise RuntimeError("Canonical Silver window is missing a timestamp column")
        timestamps = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
        values = pd.to_numeric(frame["clean_value"], errors="coerce")
        raw_values = pd.to_numeric(frame["raw_value"], errors="coerce")
        valid = timestamps.notna() & values.notna() & raw_values.notna()
        timestamps = timestamps[valid]
        values = values[valid].to_numpy(dtype=np.float64)
        raw_values = raw_values[valid].to_numpy(dtype=np.float64)
        if len(timestamps) < 2:
            raise RuntimeError(
                f"Window source has insufficient finite samples: {source}"
            )
        start = pd.to_datetime(row["window_start_utc"], utc=True)
        end = pd.to_datetime(row["window_end_utc"], utc=True)
        source_seconds = (
            (timestamps - start).dt.total_seconds().to_numpy(dtype=np.float64)
        )
        duration = float((end - start).total_seconds())
        selected = (source_seconds >= 0.0) & (source_seconds < duration)
        if selected.sum() < int(170 * float(row["sampling_rate_hz"])):
            raise RuntimeError(
                f"Window has less than 170 seconds of source coverage: {row.get('window_id')}"
            )
        source_seconds = source_seconds[selected]
        values = values[selected]
        raw_values = raw_values[selected]
        manifest = model_manifest or {}
        source_fs = float(manifest.get("source_rate_hz", 0.0))
        model_fs = float(manifest.get("model_rate_hz", 0.0))
        if not np.isfinite(source_fs) or source_fs <= 0.0:
            raise RuntimeError("Model manifest has no positive source_rate_hz")
        if not np.isfinite(model_fs) or model_fs <= 0.0:
            raise RuntimeError("Model manifest has no positive model_rate_hz")
        observed_source_fs = (
            (len(source_seconds) - 1) / float(source_seconds[-1] - source_seconds[0])
            if len(source_seconds) > 1 and source_seconds[-1] > source_seconds[0]
            else float("nan")
        )
        source_target = np.arange(0.0, duration, 1.0 / source_fs, dtype=np.float64)
        source_clean = np.interp(source_target, source_seconds, values)
        source_raw = np.interp(source_target, source_seconds, raw_values)
        target = (
            source_target
            if model_fs == source_fs
            else np.arange(0.0, duration, 1.0 / model_fs, dtype=np.float64)
        )
        clean = np.interp(target, source_target, source_clean)
        raw = np.interp(target, source_target, source_raw)
        return (
            raw.astype(np.float32),
            clean.astype(np.float32),
            model_fs,
            float(observed_source_fs),
        )

    def _feature_tensor(
        self,
        clean_values: np.ndarray,
        sampling_rate_hz: float,
        feature_order: tuple[str, ...],
        sensor_channel: str,
        window_seconds: int = 180,
    ) -> np.ndarray:
        """Recompute the model input from the exact vectorized clean waveform.

        Gold features select the real windows, but they are not a valid raw-to-
        Kotlin parity oracle after the waveform is resampled and serialized as
        float32. Recomputing here makes the expected tensor represent precisely
        the samples shipped in ``preprocessed_values``.
        """

        extractor = (
            extract_ppg_features_v4
            if self.config.feature_version.startswith("v4.")
            else extract_ppg_features_v3
        )
        features, _, _ = extractor(
            np.asarray(clean_values, dtype=np.float64),
            sampling_rate_hz,
            channel=sensor_channel,
            detector=self._peak_detector,
            **(
                {
                    "window_seconds": window_seconds,
                    "experimental_features": tuple(
                        name for name in feature_order if name in EXPERIMENTAL_FEATURES
                    ),
                }
                if extractor is extract_ppg_features_v4
                else {}
            ),
        )
        features["sampling_rate_hz"] = float(sampling_rate_hz)
        ordered = np.asarray(
            [[features[name] for name in feature_order]], dtype=np.float64
        )
        return assert_feature_tensor(ordered, n_features=len(feature_order))[0]

    @staticmethod
    def _publish_candidate_index(
        export_dir: Path,
        manifest_path: Path,
        parity_path: Path,
        vectors_path: Path,
        model_path: Path,
    ) -> None:
        """Expose instrumented-test assets without publishing a production model."""

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        from .onnx_v3 import _manifest_sha256

        if manifest.get("manifest_sha256") != _manifest_sha256(manifest):
            raise RuntimeError("Candidate manifest_sha256 mismatch")
        if (
            manifest.get("release_status") != "release_candidate"
            or manifest.get("gate_passed") is not True
            or manifest.get("android_ready") is not False
        ):
            raise RuntimeError(
                "Candidate index requires a passed, non-production manifest"
            )
        vectors = json.loads(vectors_path.read_text(encoding="utf-8"))
        _assert_vector_contract(
            list(vectors.get("cases", [])), valid_count=94, rejection_count=6
        )
        descriptors = {
            role: {
                "path": path.relative_to(export_dir).as_posix(),
                "sha256": _sha256(path),
            }
            for role, path in {
                "model": model_path,
                "manifest": manifest_path,
                "parity": parity_path,
                "test_vectors": vectors_path,
            }.items()
        }
        path = export_dir / "candidate_model_index.json"
        existing = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.is_file()
            else {"schema_version": "1.0.0", "models": []}
        )
        models = [
            model
            for model in existing.get("models", [])
            if not (
                model.get("model_version") == manifest["model_version"]
                and model.get("sensor_channel") == manifest["sensor_channel"]
            )
        ]
        models.append(
            {
                "model_version": manifest["model_version"],
                "sensor_channel": manifest["sensor_channel"],
                "release_status": "candidate_physical_parity_pending",
                "android_ready": False,
                "performance_policy_version": manifest.get(
                    "performance_policy_version"
                ),
                "performance_policy_sha256": manifest.get("performance_policy_sha256"),
                "performance_tier": manifest.get("performance_tier"),
                "deployment_scope": manifest.get("deployment_scope"),
                "research_disclaimer": manifest.get("research_disclaimer"),
                "research_only": manifest["sensor_channel"] == "ppg_red",
                "files": descriptors,
            }
        )
        _atomic_write_json(
            path,
            {
                "schema_version": "1.0.0",
                "release_status": "candidate_physical_parity_pending",
                "models": sorted(
                    models,
                    key=lambda item: (
                        item["model_version"],
                        item["sensor_channel"],
                    ),
                ),
            },
        )

    def generate(self, sensor_channel: str) -> dict[str, Any]:
        if sensor_channel not in TRAINING_CHANNELS:
            raise ValueError(
                f"Android model vectors are restricted to {TRAINING_CHANNELS}: "
                f"{sensor_channel}"
            )
        versioned_dir = self.export_dir / self.config.model_version / sensor_channel
        manifest_path = versioned_dir / "model_manifest.json"
        parity_path = versioned_dir / "onnx_parity.json"
        model_path = versioned_dir / f"stress_{sensor_channel}_fp32.onnx"
        if (
            not manifest_path.is_file()
            or not parity_path.is_file()
            or not model_path.is_file()
        ):
            raise FileNotFoundError(
                "Canonical ONNX model, manifest, and parity report are required before vector generation"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("release_status") != "release_candidate"
            or manifest.get("gate_passed") is not True
            or manifest.get("android_ready") is not False
        ):
            raise RuntimeError(
                "Android vectors require a passed ONNX candidate pending device parity"
            )
        if _sha256(model_path) != manifest.get("onnx_sha256"):
            raise RuntimeError("ONNX checksum does not match the canonical manifest")
        gold, gold_source = self._load_gold(sensor_channel, manifest)
        validation_cohort = set(manifest.get("validation_cohort_ids", []))
        if validation_cohort != {"ID-29", "ID-30", "ID-31", "ID-32"}:
            raise RuntimeError(
                "Canonical model manifest has an invalid validation cohort"
            )
        feature_order = tuple(manifest.get("feature_order", []))
        if not feature_order:
            raise RuntimeError("Canonical classical manifest has no feature order")
        holdout_pool = gold[
            (gold["sensor_channel"] == sensor_channel)
            & gold["participant_id"].astype(str).isin(manifest["validation_cohort_ids"])
        ].sort_values(["participant_id", "window_start_utc"])
        required_vectors = int(self.config.android.valid_vector_count)
        if len(holdout_pool) >= required_vectors:
            selected = holdout_pool.iloc[
                np.linspace(0, len(holdout_pool) - 1, required_vectors, dtype=int)
            ]
        else:
            development_pool = gold[
                (gold["sensor_channel"] == sensor_channel)
                & ~gold["participant_id"]
                .astype(str)
                .isin(manifest["validation_cohort_ids"])
            ].sort_values(["participant_id", "window_start_utc"])
            remaining = required_vectors - len(holdout_pool)
            if len(development_pool) < remaining:
                raise RuntimeError(
                    "Canonical Gold pool has fewer than the required real vectors"
                )
            development_selected = development_pool.iloc[
                np.linspace(0, len(development_pool) - 1, remaining, dtype=int)
            ]
            selected = pd.concat(
                [holdout_pool, development_selected], ignore_index=True
            )
        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        output_name = manifest["output"]["name"]
        cache: dict[Path, pd.DataFrame] = {}
        valid_cases: list[dict[str, Any]] = []
        for position, (row_index, row) in enumerate(selected.iterrows()):
            raw, clean, fs, observed_fs = self._load_window(row, cache, manifest)
            tensor = self._feature_tensor(
                clean,
                fs,
                feature_order,
                sensor_channel,
                int(manifest.get("window_seconds", 180)),
            )
            expected = float(
                np.asarray(
                    session.run([output_name], {input_name: tensor.reshape(1, -1)})[0]
                ).reshape(-1)[0]
            )
            if not np.isfinite(expected) or not 0.0 <= expected <= 1.0:
                raise RuntimeError(
                    "ONNX returned a non-finite probability for a real vector"
                )
            deltas = np.diff(
                np.arange(len(raw), dtype=np.float64) * (1000.0 / fs), prepend=0.0
            )
            valid_cases.append(
                {
                    "case_id": f"parity_{position:03d}_{row.get('window_id', row_index)}",
                    "kind": "valid_raw_window",
                    "source_file": str(row["source_file"]),
                    "source_checksum": str(row["source_checksum"]),
                    "source_window_id": str(row.get("window_id", row_index)),
                    "participant_id": str(row["participant_id"]),
                    "phase": str(row["phase"]),
                    "sensor_channel": sensor_channel,
                    "source_rate_hz": float(manifest["source_rate_hz"]),
                    "model_rate_hz": float(manifest["model_rate_hz"]),
                    "observed_rate_hz": observed_fs,
                    "sampling_rate_hz": fs,
                    "filter_contract": manifest.get("filter_contract"),
                    "raw_values": raw.tolist(),
                    "preprocessed_values": clean.tolist(),
                    "timestamp_deltas_ms": deltas.tolist(),
                    # Runtime wire names are intentionally identical to the
                    # Kotlin loader. Keep the old aliases for one transition
                    # cycle so external audit tooling can still read vectors.
                    # The tensor is already canonical float32. Decimal rounding
                    # a float32 at high precision can move it by several ULPs
                    # (for example 23464.586 -> 23464.583984375), invalidating
                    # an otherwise exact Kotlin parity vector.
                    "input": tensor.tolist(),
                    "onnx_probability": expected,
                    "expected_features": tensor.tolist(),
                    "expected_stress_probability": expected,
                    "expected_stress_label": int(
                        expected >= float(manifest["probability_threshold"])
                    ),
                }
            )
        edge_cases = [
            {
                "case_id": "edge_poor_quality",
                "kind": "quality_rejection",
                "quality_tier": "poor",
                "expected_status": "invalid_quality",
                "expected_probability": None,
                "expected_label": None,
            },
            {
                "case_id": "edge_invalid_quality",
                "kind": "quality_rejection",
                "quality_tier": "invalid",
                "expected_status": "invalid_quality",
                "expected_probability": None,
                "expected_label": None,
            },
            {
                "case_id": "edge_broken_timestamp",
                "kind": "timestamp_rejection",
                "timestamp_deltas_ms": [10.0, -1.0, 10.0],
                "expected_status": "invalid_timestamp",
                "expected_probability": None,
                "expected_label": None,
            },
            {
                "case_id": "edge_missing_feature",
                "kind": "schema_rejection",
                "feature_values": [0.0] * (len(feature_order) - 1),
                "expected_status": "schema_mismatch",
                "expected_probability": None,
                "expected_label": None,
            },
            {
                "case_id": "edge_hash_mismatch",
                "kind": "model_rejection",
                "model_sha256": "0" * 64,
                "expected_status": "model_hash_mismatch",
                "expected_probability": None,
                "expected_label": None,
            },
            {
                "case_id": "edge_model_unavailable",
                "kind": "model_rejection",
                "model_available": False,
                "expected_status": "model_unavailable",
                "expected_probability": None,
                "expected_label": None,
            },
        ]
        cases = valid_cases + edge_cases
        _assert_vector_contract(
            cases,
            valid_count=int(self.config.android.valid_vector_count),
            rejection_count=int(self.config.android.edge_vector_count),
        )
        payload = {
            "schema_version": "2.0.0",
            "vector_schema_version": "2.0.0",
            "model_version": manifest["model_version"],
            "preprocessing_version": manifest["preprocessing_version"],
            "feature_version": manifest["feature_version"],
            "sensor_channel": sensor_channel,
            "source_rate_hz": float(manifest["source_rate_hz"]),
            "model_rate_hz": float(manifest["model_rate_hz"]),
            "filter_contract": manifest.get("filter_contract"),
            "model_sha256": manifest["onnx_sha256"],
            "gold_source": gold_source.relative_to(self.root).as_posix(),
            "gold_source_sha256": _sha256(gold_source),
            "input_name": input_name,
            "output_name": output_name,
            "probability_threshold": manifest["probability_threshold"],
            "python_vectors_passed": len(cases),
            "python_vector_status": "passed",
            "android_instrumentation_status": "pending",
            "android_ready": False,
            "cases": cases,
        }
        vectors_path = versioned_dir / "test_vectors.json"
        _atomic_write_json(vectors_path, payload)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(self.output_path, payload)
        self._publish_candidate_index(
            self.export_dir, manifest_path, parity_path, vectors_path, model_path
        )
        return payload


class DeviceParityPromoter:
    """Promote verified Android candidates without mutating their evidence.

    Candidate artifacts remain immutable.  The production index is the runtime
    gate, therefore it is published only after every requested channel has been
    validated and its production artifacts have been materialized.
    """

    def __init__(self, project_root: Path, config: PipelineConfig) -> None:
        self.root = project_root.resolve()
        self.config = config
        self.paths = pipeline_paths_from_config(self.root, config)
        self.export_dir = self.paths.export_dir
        self.verifier = DeviceParityReceiptVerifier(
            max_abs_diff=self.config.onnx.max_abs_diff
        )

    def _candidate_paths(self, sensor_channel: str) -> dict[str, Path]:
        base = self.export_dir / self.config.model_version / sensor_channel
        return {
            "base": base,
            "model": base / f"stress_{sensor_channel}_fp32.onnx",
            "manifest": base / "model_manifest.json",
            "parity": base / "onnx_parity.json",
            "vectors": base / "test_vectors.json",
        }

    def _production_entries(self) -> list[dict[str, Any]]:
        index_path = self.export_dir / "model_index.json"
        if not index_path.is_file():
            return []
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        entries: list[dict[str, Any]] = []
        for entry in index.get("models", []):
            if entry.get("sensor_channel") not in ANDROID_RUNTIME_CHANNELS:
                continue
            if entry.get("release_status") != "accepted":
                continue
            if entry.get("android_ready") is not True:
                continue
            descriptor = entry.get("files", {}).get("manifest", {})
            manifest_path = self.export_dir / str(descriptor.get("path", ""))
            if not manifest_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if (
                manifest.get("release_status") == "accepted"
                and manifest.get("android_ready") is True
                and descriptor.get("sha256") == _sha256(manifest_path)
            ):
                entries.append(entry)
        return entries

    def _verify_candidate(self, receipt_path: Path, apk_path: Path) -> dict[str, Any]:
        if not receipt_path.is_file():
            raise FileNotFoundError(f"DeviceParityReceipt is missing: {receipt_path}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        channel = str(receipt.get("sensor_channel", ""))
        if channel not in ANDROID_RUNTIME_CHANNELS:
            raise RuntimeError(
                "Android production promotion is restricted to ppg_green and ppg_ir"
            )
        paths = self._candidate_paths(channel)
        for role in ("model", "manifest", "parity", "vectors"):
            if not paths[role].is_file():
                raise FileNotFoundError(f"Candidate {role} is missing: {paths[role]}")
        candidate_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        from .onnx_v3 import _manifest_sha256

        if candidate_manifest.get("manifest_sha256") != _manifest_sha256(
            candidate_manifest
        ):
            raise RuntimeError("Candidate model manifest_sha256 mismatch")
        if (
            candidate_manifest.get("release_status") != "release_candidate"
            or candidate_manifest.get("gate_passed") is not True
            or candidate_manifest.get("android_ready") is not False
        ):
            raise RuntimeError("Physical promotion requires a pending ONNX candidate")
        candidate_vectors = json.loads(paths["vectors"].read_text(encoding="utf-8"))
        _assert_vector_contract(
            list(candidate_vectors.get("cases", [])),
            valid_count=int(self.config.android.valid_vector_count),
            rejection_count=int(self.config.android.edge_vector_count),
        )
        verified = self.verifier.verify(
            receipt,
            manifest=candidate_manifest,
            model_path=paths["model"],
            apk_path=apk_path.resolve(),
        )
        return {
            "channel": channel,
            "paths": paths,
            "candidate_manifest": candidate_manifest,
            "candidate_vectors": candidate_vectors,
            "receipt": verified,
        }

    def _materialize_candidate(
        self, candidate: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Write non-index artifacts for one already-verified candidate.

        These files are intentionally written before the index.  They are not
        reachable by Android runtime until ``promote_many`` atomically writes
        the completed production index.
        """

        channel = str(candidate["channel"])
        paths = candidate["paths"]
        candidate_manifest = candidate["candidate_manifest"]
        candidate_vectors = candidate["candidate_vectors"]
        verified = candidate["receipt"]
        from .onnx_v3 import _android_contract_payload, _manifest_sha256

        production_receipt_path = paths["base"] / "device_parity_receipt.json"
        _atomic_write_json(production_receipt_path, verified)
        production_vectors = {
            **candidate_vectors,
            "android_instrumentation_status": "passed",
            "android_ready": True,
            "android_vector_count": 100,
            "android_max_abs_diff": float(verified["max_abs_diff"]),
            "device_parity_receipt_sha256": verified["receipt_sha256"],
        }
        production_vectors_path = paths["base"] / "production_test_vectors.json"
        _atomic_write_json(production_vectors_path, production_vectors)
        _assert_promotable_vectors(
            production_vectors,
            f"{candidate_manifest['model_version']}/{channel}",
        )

        production_manifest = {
            **candidate_manifest,
            "release_status": "accepted",
            "android_ready": True,
            "android_parity": {
                "status": "passed",
                "vector_count": 100,
                "max_abs_diff": float(verified["max_abs_diff"]),
                "device_fingerprint": verified["device_fingerprint"],
                "apk_sha256": verified["apk_sha256"],
                "timestamp_utc": verified["timestamp_utc"],
            },
            "device_parity_receipt_sha256": verified["receipt_sha256"],
            "source_candidate_manifest_sha256": _sha256(paths["manifest"]),
        }
        production_manifest["android_contract_sha256"] = hashlib.sha256(
            _android_contract_payload(production_manifest).encode("utf-8")
        ).hexdigest()
        production_manifest["manifest_sha256"] = _manifest_sha256(production_manifest)
        production_manifest_path = paths["base"] / "production_model_manifest.json"
        _atomic_write_json(production_manifest_path, production_manifest)

        descriptors = {
            role: {
                "path": path.relative_to(self.export_dir).as_posix(),
                "sha256": _sha256(path),
            }
            for role, path in {
                "model": paths["model"],
                "manifest": production_manifest_path,
                "parity": paths["parity"],
                "test_vectors": production_vectors_path,
                "device_parity_receipt": production_receipt_path,
            }.items()
        }
        conservative_metrics = production_manifest.get("conservative_metrics")
        if not isinstance(conservative_metrics, dict):
            raise RuntimeError(
                f"{production_manifest['model_version']}/{channel}: "
                "missing conservative_metrics for policy-tier verification"
            )
        try:
            conservative = {
                key: float(conservative_metrics[key])
                for key in ("balanced_accuracy", "f1", "roc_auc")
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{production_manifest['model_version']}/{channel}: "
                "invalid conservative_metrics for policy-tier verification"
            ) from exc
        entry = {
            "model_version": production_manifest["model_version"],
            "sensor_channel": channel,
            "model_family": production_manifest["model_family"],
            "calibration_version": production_manifest.get("calibration_version", ""),
            "release_status": "accepted",
            "android_ready": True,
            "balanced_accuracy_holdout": production_manifest[
                "balanced_accuracy_holdout"
            ],
            "f1_holdout": production_manifest["f1_holdout"],
            "roc_auc_holdout": production_manifest["roc_auc_holdout"],
            "balanced_accuracy_conservative": conservative["balanced_accuracy"],
            "f1_conservative": conservative["f1"],
            "roc_auc_conservative": conservative["roc_auc"],
            "performance_policy_version": production_manifest.get(
                "performance_policy_version", ""
            ),
            "performance_policy_sha256": production_manifest.get(
                "performance_policy_sha256", ""
            ),
            "performance_tier": production_manifest.get("performance_tier", ""),
            "deployment_scope": production_manifest.get("deployment_scope", ""),
            "research_disclaimer": production_manifest.get("research_disclaimer", ""),
            "disclaimer_required": bool(
                production_manifest.get("disclaimer_required", False)
            ),
            "analysis_seconds": production_manifest["analysis_seconds"],
            "window_seconds": production_manifest["window_seconds"],
            "stride_seconds": production_manifest["stride_seconds"],
            "expected_windows_per_session": production_manifest[
                "expected_windows_per_session"
            ],
            "source_rate_hz": production_manifest["source_rate_hz"],
            "model_rate_hz": production_manifest["model_rate_hz"],
            "cohort_ids": production_manifest["validation_cohort_ids"],
            "files": descriptors,
        }
        return entry, {
            "sensor_channel": channel,
            "model_version": production_manifest["model_version"],
            "receipt_sha256": verified["receipt_sha256"],
        }

    def promote_many(self, receipt_paths: list[Path], apk_path: Path) -> dict[str, Any]:
        """Atomically publish all requested Green/IR candidates after validation."""

        if not receipt_paths:
            raise ValueError("At least one DeviceParityReceipt is required")
        staged = [
            self._verify_candidate(receipt_path.resolve(), apk_path)
            for receipt_path in receipt_paths
        ]
        channels = [str(candidate["channel"]) for candidate in staged]
        if len(set(channels)) != len(channels):
            raise RuntimeError(
                "Each Android channel may be promoted only once per batch"
            )

        entries = [
            entry
            for entry in self._production_entries()
            if entry.get("sensor_channel") not in set(channels)
        ]
        promoted: list[dict[str, Any]] = []
        for candidate in staged:
            entry, result = self._materialize_candidate(candidate)
            entries.append(entry)
            promoted.append(result)
        production_index = {
            "schema_version": "1.0.0",
            "release_status": "accepted",
            "models": sorted(entries, key=lambda item: item["sensor_channel"]),
        }
        _atomic_write_json(self.export_dir / "model_index.json", production_index)
        return {
            "status": "android_ready",
            "promoted": promoted,
            "model_index": str(self.export_dir / "model_index.json"),
        }

    def promote(self, receipt_path: Path, apk_path: Path) -> dict[str, Any]:
        """Backward-compatible single-channel promotion boundary."""

        result = self.promote_many([receipt_path], apk_path)
        return {**result, **result["promoted"][0]}


def _assert_promotable_vectors(vectors: dict[str, Any], identity: str) -> None:
    """Require both Python and physical-device evidence before index promotion."""

    if vectors.get("python_vectors_passed") != 100:
        raise RuntimeError(f"{identity}: Python vector evidence is incomplete")
    if vectors.get("python_vector_status") != "passed":
        raise RuntimeError(f"{identity}: Python vector status is not passed")
    if vectors.get("android_instrumentation_status") != "passed":
        raise RuntimeError(
            f"{identity}: Android instrumentation evidence is not passed"
        )
    if vectors.get("android_ready") is not True:
        raise RuntimeError(f"{identity}: vector artifact is not Android-ready")
