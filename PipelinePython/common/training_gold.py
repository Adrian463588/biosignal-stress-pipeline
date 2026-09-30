"""Derive immutable training Gold variants from canonical clean Silver PPG."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .feature_policy import DEFAULT_PREDICTORS
from .features_v4 import extract_ppg_features
from .orchestration import source_tree_sha256
from .peak_detection import PpgPeakDetector
from .training_protocol import channel_group_readiness

FeatureExtractor = Callable[..., tuple[dict[str, float], np.ndarray, np.ndarray]]
MODEL_PHASES = ("relax", "task")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _variant_key(window_seconds: int, stride_seconds: int) -> str:
    return f"w{window_seconds:03d}_s{stride_seconds:03d}"


class TrainingGoldBuilder:
    """Window clean Silver once, without filtering or interpolation."""

    def __init__(
        self,
        project_root: Path,
        config: Any,
        source_run_root: Path,
        output_run_root: Path,
        *,
        extractor: FeatureExtractor = extract_ppg_features,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.config = config
        self.source_run_root = Path(source_run_root).resolve()
        self.output_run_root = Path(output_run_root).resolve()
        self.extractor = extractor
        peak_config = getattr(config, "peak_detection", None)
        self.detector = (
            PpgPeakDetector(peak_config) if peak_config is not None else None
        )

    def build(self) -> dict[str, Any]:
        if self.source_run_root == self.output_run_root:
            raise ValueError("source_and_output_run_must_differ")
        manifest_path = (
            self.source_run_root / "silver" / "preprocessing_manifest_v3.parquet"
        )
        if not manifest_path.is_file():
            raise FileNotFoundError(f"silver_manifest_missing:{manifest_path}")
        gold_root = self.output_run_root / "gold"
        if gold_root.exists() and any(gold_root.iterdir()):
            raise RuntimeError(f"derived_gold_output_not_empty:{gold_root}")
        gold_root.mkdir(parents=True, exist_ok=True)

        manifest = pd.read_parquet(manifest_path)
        required = {
            "participant_id",
            "phase",
            "sensor_channel",
            "quality_tier",
            "source_file",
            "source_checksum",
            "preprocessing_version",
            "path",
        }
        missing = sorted(required - set(manifest.columns))
        if missing:
            raise ValueError(f"silver_manifest_columns_missing:{','.join(missing)}")
        channels = tuple(self.config.training.training_channels)
        ppg = manifest[
            manifest["sensor_channel"].isin(channels)
            & manifest["phase"].astype(str).str.lower().isin(MODEL_PHASES)
        ].copy()
        ppg["phase"] = ppg["phase"].astype(str).str.lower()
        expected_ids = {f"ID-{index:02d}" for index in range(1, 33)}
        unexpected_ids = sorted(set(ppg["participant_id"].astype(str)) - expected_ids)
        if unexpected_ids:
            raise ValueError(f"unexpected_participant:{','.join(unexpected_ids)}")
        duplicates = ppg.duplicated(
            ["participant_id", "phase", "sensor_channel"], keep=False
        )
        if duplicates.any():
            raise ValueError("duplicate_silver_participant_phase_channel")

        variants: dict[str, dict[str, list[dict[str, Any]]]] = {
            _variant_key(window_seconds, stride_seconds): {
                "features": [],
                "sequences": [],
                "exclusions": [],
            }
            for window_seconds, stride_seconds in self.config.training.window_candidates
        }
        source_artifacts: list[dict[str, Any]] = []
        for row in ppg.sort_values(
            ["participant_id", "phase", "sensor_channel"], kind="mergesort"
        ).to_dict(orient="records"):
            source_path = Path(str(row["path"]))
            if not source_path.is_absolute():
                source_path = self.project_root / source_path
            if not source_path.is_file():
                self._exclude_source(variants, row, "silver_artifact_missing")
                continue
            source_sha256 = _sha256(source_path)
            source_artifacts.append(
                {
                    "participant_id": str(row["participant_id"]),
                    "phase": str(row["phase"]),
                    "sensor_channel": str(row["sensor_channel"]),
                    "silver_path": str(source_path),
                    "silver_sha256": source_sha256,
                    "raw_source_file": str(row["source_file"]),
                    "raw_source_checksum": str(row["source_checksum"]),
                }
            )
            frame = (
                pd.read_csv(source_path)
                if source_path.suffix.lower() == ".csv"
                else pd.read_parquet(source_path)
            )
            self._build_source_windows(variants, row, frame, source_path, source_sha256)

        readiness_by_variant: dict[str, Any] = {}
        for (window_seconds, stride_seconds), (key, payload) in zip(
            self.config.training.window_candidates, variants.items(), strict=True
        ):
            readiness_by_variant[key] = self._write_variant(
                gold_root,
                key,
                payload,
                int(window_seconds),
                int(stride_seconds),
            )

        baseline_key = _variant_key(180, 10)
        baseline = variants[baseline_key]
        baseline_features = self._feature_frame(baseline["features"])
        baseline_sequences = self._sequence_frame(baseline["sequences"])
        expanded_tiers = set(self.config.training.expanded_quality_tiers)
        baseline_features = baseline_features[
            baseline_features["quality_tier"].isin(expanded_tiers)
        ].copy()
        baseline_sequences = baseline_sequences[
            baseline_sequences["quality_tier"].isin(expanded_tiers)
        ].copy()
        baseline_features.to_parquet(
            gold_root / "gold_features_v3.parquet", index=False
        )
        baseline_features.to_csv(gold_root / "gold_features_v3.csv", index=False)
        baseline_sequences.to_parquet(
            gold_root / "gold_ibi_sequences_v3.parquet", index=False
        )
        pd.DataFrame(
            [row for payload in variants.values() for row in payload["exclusions"]]
        ).to_csv(gold_root / "training_exclusions.csv", index=False)

        groups = {
            (str(row.participant_id), str(row.phase), str(row.sensor_channel))
            for row in baseline_features[["participant_id", "phase", "sensor_channel"]]
            .drop_duplicates()
            .itertuples(index=False)
        }
        readiness = self._readiness(groups)
        result = {
            "schema_version": "training-gold-v1",
            "source_run_id": self.source_run_root.name,
            "source_run_root": str(self.source_run_root),
            "source_silver_manifest": str(manifest_path),
            "source_silver_manifest_sha256": _sha256(manifest_path),
            "source_silver_artifact_count": len(source_artifacts),
            "source_silver_artifacts_sha256": hashlib.sha256(
                json.dumps(
                    source_artifacts, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
            "feature_order": list(DEFAULT_PREDICTORS),
            "feature_version": getattr(self.config, "feature_version", "unknown"),
            "model_version": getattr(self.config, "model_version", "unknown"),
            "window_variants": readiness_by_variant,
            "expected_training_groups": 192,
            "observed_training_groups": readiness["observed_total_groups"],
            "gold_channel_completeness_status": readiness["channel_status"],
            "training_channel_status": readiness["training_channel_status"],
            "training_development_coverage": readiness["development_coverage"],
            "ppg_gold_status": readiness["status"],
            "training_status": readiness["training_status"],
            "raw_or_silver_modified": False,
        }
        source_contract_path = self.source_run_root / "run_contract_fingerprint.json"
        result["source_run_contract_sha256"] = (
            _sha256(source_contract_path)
            if source_contract_path.is_file()
            else "UNAVAILABLE:source_run_contract_missing"
        )
        run_contract_path = self.output_run_root / "run_contract_fingerprint.json"
        if run_contract_path.is_file():
            run_contract = json.loads(run_contract_path.read_text(encoding="utf-8"))
            if run_contract.get("run_id") != self.output_run_root.name:
                raise RuntimeError("training_gold_run_contract_id_mismatch")
            current_source_tree = source_tree_sha256(self.project_root)
            if run_contract.get("source_tree_sha256") != current_source_tree:
                raise RuntimeError("training_gold_source_tree_mismatch")
            config_fingerprint = str(run_contract["config_fingerprint"])
        else:
            # Unit fixtures may exercise this pure builder without the canonical
            # orchestrator. Research runs always enter through ``training-gold``.
            config_fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "training": repr(self.config.training),
                        "peak_detection": repr(
                            getattr(self.config, "peak_detection", None)
                        ),
                        "feature_version": getattr(
                            self.config, "feature_version", "unknown"
                        ),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            run_contract = {
                "schema_version": "training-gold-test-contract-v1",
                "run_id": self.output_run_root.name,
                "source_tree_sha256": source_tree_sha256(self.project_root),
                "config_fingerprint": config_fingerprint,
            }
            run_contract_path.write_text(
                json.dumps(run_contract, indent=2, sort_keys=True), encoding="utf-8"
            )
        run_contract_sha256 = _sha256(run_contract_path)
        artifact_checksums = {
            "gold_features_v3.parquet": _sha256(gold_root / "gold_features_v3.parquet"),
            "gold_ibi_sequences_v3.parquet": _sha256(
                gold_root / "gold_ibi_sequences_v3.parquet"
            ),
        }
        result.update(
            {
                "run_id": self.output_run_root.name,
                "run_contract_sha256": run_contract_sha256,
                "source_tree_sha256": run_contract["source_tree_sha256"],
                "config_fingerprint": config_fingerprint,
                "expected_training_groups_by_channel": readiness["expected_groups"],
                "observed_training_groups_by_channel": readiness["observed_groups"],
                "missing_training_groups_by_channel": readiness["missing_groups"],
                "gold_artifact_checksums": artifact_checksums,
            }
        )
        (gold_root / "manifest_v3.json").write_text(
            json.dumps(
                {
                    "schema_version": "training-gold-manifest-v1",
                    "run_id": self.output_run_root.name,
                    "source_run_id": self.source_run_root.name,
                    "source_tree_sha256": run_contract["source_tree_sha256"],
                    "config_fingerprint": config_fingerprint,
                    "run_contract_sha256": run_contract_sha256,
                    "feature_order": list(DEFAULT_PREDICTORS),
                    "feature_count": len(DEFAULT_PREDICTORS),
                    "artifact_checksums": artifact_checksums,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (gold_root / "source_silver_lineage.json").write_text(
            json.dumps(
                {**result, "source_silver_artifacts": source_artifacts},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (gold_root / "data_readiness.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        return result

    def _build_source_windows(
        self,
        variants: dict[str, dict[str, list[dict[str, Any]]]],
        row: dict[str, Any],
        frame: pd.DataFrame,
        source_path: Path,
        source_sha256: str,
    ) -> None:
        required = {"timestamp_corrected_utc", "clean_value"}
        missing = sorted(required - set(frame.columns))
        if missing:
            self._exclude_source(
                variants, row, f"silver_columns_missing:{','.join(missing)}"
            )
            return
        timestamps = pd.to_datetime(frame["timestamp_corrected_utc"], utc=True)
        order = np.argsort(timestamps.to_numpy(), kind="mergesort")
        frame = frame.iloc[order].reset_index(drop=True)
        timestamps = timestamps.iloc[order].reset_index(drop=True)
        if "core_start_utc" in frame and frame["core_start_utc"].notna().any():
            core_start = pd.to_datetime(
                frame.loc[frame["core_start_utc"].notna(), "core_start_utc"].iloc[0],
                utc=True,
            )
        else:
            core_start = timestamps.iloc[0]
        elapsed = (timestamps - core_start).dt.total_seconds().to_numpy(dtype=float)
        finite_delta = np.diff(elapsed)
        finite_delta = finite_delta[np.isfinite(finite_delta) & (finite_delta > 0)]
        sampling_rate_hz = (
            float(1.0 / np.median(finite_delta))
            if finite_delta.size
            else float(frame.get("sampling_rate_hz", pd.Series([np.nan])).iloc[0])
        )
        if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
            self._exclude_source(variants, row, "invalid_silver_sampling_rate")
            return
        values = frame["clean_value"].to_numpy(dtype=np.float64)
        interpolation = (
            frame["interpolated_mask"].fillna(False).to_numpy(dtype=bool)
            if "interpolated_mask" in frame
            else np.zeros(len(frame), dtype=bool)
        )
        for window_seconds, stride_seconds in self.config.training.window_candidates:
            key = _variant_key(int(window_seconds), int(stride_seconds))
            count = ((240 - int(window_seconds)) // int(stride_seconds)) + 1
            for window_index in range(count):
                start = window_index * int(stride_seconds)
                end = start + int(window_seconds)
                mask = (elapsed >= start) & (elapsed < end)
                selected_elapsed = elapsed[mask]
                tolerance = max(0.5, 1.5 / sampling_rate_hz)
                if (
                    selected_elapsed.size < 2
                    or selected_elapsed[0] > start + tolerance
                    or selected_elapsed[-1] < end - tolerance
                ):
                    variants[key]["exclusions"].append(
                        self._exclusion_row(
                            row, key, window_index, "insufficient_window_coverage"
                        )
                    )
                    continue
                signal = values[mask]
                if not np.isfinite(signal).all():
                    variants[key]["exclusions"].append(
                        self._exclusion_row(
                            row, key, window_index, "non_finite_clean_value"
                        )
                    )
                    continue
                # Silver keeps the observed timestamps and therefore a valid
                # 180-second window can contain slightly fewer samples than
                # ``nominal_rate * duration``.  Derive the rate from that
                # window's real elapsed coverage instead of padding or
                # fabricating samples merely to satisfy a nominal count.
                window_sampling_rate_hz = float(signal.size / (end - start))
                if (
                    not np.isfinite(window_sampling_rate_hz)
                    or window_sampling_rate_hz <= 0
                ):
                    variants[key]["exclusions"].append(
                        self._exclusion_row(
                            row, key, window_index, "invalid_window_sampling_rate"
                        )
                    )
                    continue
                detection = (
                    self.detector.detect(
                        signal,
                        window_sampling_rate_hz,
                        str(row["sensor_channel"]),
                    )
                    if self.detector is not None
                    else None
                )
                if detection is not None and detection.status != "AVAILABLE":
                    variants[key]["exclusions"].append(
                        self._exclusion_row(
                            row,
                            key,
                            window_index,
                            "peak_detection_unavailable:" + detection.status_reason,
                        )
                    )
                    continue
                try:
                    keyword_arguments = {
                        "channel": str(row["sensor_channel"]),
                        "window_seconds": int(window_seconds),
                        "experimental_features": (),
                    }
                    if detection is not None:
                        keyword_arguments.update(
                            detector=self.detector,
                            detection=detection,
                        )
                    features, sequence, sequence_mask = self.extractor(
                        signal,
                        window_sampling_rate_hz,
                        **keyword_arguments,
                    )
                except (ValueError, RuntimeError) as exc:
                    variants[key]["exclusions"].append(
                        self._exclusion_row(
                            row, key, window_index, f"feature_error:{exc}"
                        )
                    )
                    continue
                predictors = {name: features.get(name) for name in DEFAULT_PREDICTORS}
                if not all(
                    value is not None and np.isfinite(float(value))
                    for value in predictors.values()
                ):
                    variants[key]["exclusions"].append(
                        self._exclusion_row(
                            row, key, window_index, "non_finite_predictor"
                        )
                    )
                    continue
                window_id = (
                    f"{row['participant_id']}_{row['phase']}_{row['sensor_channel']}_"
                    f"{key}_{window_index:02d}"
                )
                quality_tier = str(row["quality_tier"])
                metadata = {
                    "participant_id": str(row["participant_id"]),
                    "session_id": str(
                        frame.get(
                            "session_id", pd.Series([row["participant_id"]])
                        ).iloc[0]
                    ),
                    "recording_id": str(
                        frame.get(
                            "recording_id",
                            pd.Series([f"{row['participant_id']}_{row['phase']}"]),
                        ).iloc[0]
                    ),
                    "phase": str(row["phase"]),
                    "label": 0 if str(row["phase"]) == "relax" else 1,
                    "label_name": "Normal" if str(row["phase"]) == "relax" else "Stres",
                    "sensor_channel": str(row["sensor_channel"]),
                    "window_id": window_id,
                    "window_index": window_index,
                    "window_seconds": int(window_seconds),
                    "stride_seconds": int(stride_seconds),
                    "window_start_utc": (
                        core_start + pd.Timedelta(seconds=start)
                    ).isoformat(),
                    "window_end_utc": (
                        core_start + pd.Timedelta(seconds=end)
                    ).isoformat(),
                    "sampling_rate_hz": window_sampling_rate_hz,
                    "quality_score": float(row.get("quality_score", np.nan)),
                    "quality_tier": quality_tier,
                    "quality_flags": str(row.get("quality_flags", "")),
                    "training_eligible": quality_tier
                    in set(self.config.training.expanded_quality_tiers),
                    "peak_detection_method": (
                        detection.method
                        if detection is not None
                        else "injected_extractor"
                    ),
                    "peak_detection_status": (
                        detection.status if detection is not None else "TEST_ONLY"
                    ),
                    "peak_detection_status_reason": (
                        detection.status_reason if detection is not None else ""
                    ),
                    "peak_count": (
                        int(detection.peaks.size) if detection is not None else np.nan
                    ),
                    "physiological_ibi_ratio": (
                        float(detection.physiological_ibi_ratio)
                        if detection is not None
                        else np.nan
                    ),
                    "interpolated_fraction": float(np.mean(interpolation[mask])),
                    "source_file": str(row["source_file"]),
                    "source_checksum": str(row["source_checksum"]),
                    "source_silver_path": str(source_path),
                    "source_silver_sha256": source_sha256,
                    "preprocessing_version": str(row["preprocessing_version"]),
                    "feature_version": getattr(
                        self.config, "feature_version", "unknown"
                    ),
                }
                variants[key]["features"].append(
                    {
                        **metadata,
                        **{name: float(value) for name, value in predictors.items()},
                    }
                )
                variants[key]["sequences"].append(
                    {
                        **metadata,
                        "ibi_values": np.asarray(sequence, dtype=np.float32).tolist(),
                        "ibi_mask": np.asarray(sequence_mask, dtype=np.uint8).tolist(),
                    }
                )

    def _write_variant(
        self,
        gold_root: Path,
        key: str,
        payload: dict[str, list[dict[str, Any]]],
        window_seconds: int,
        stride_seconds: int,
    ) -> dict[str, Any]:
        variant_root = gold_root / "training_variants" / key
        feature_frame = self._feature_frame(payload["features"])
        sequence_frame = self._sequence_frame(payload["sequences"])
        result: dict[str, Any] = {
            "window_seconds": window_seconds,
            "stride_seconds": stride_seconds,
            "cohorts": {},
        }
        for cohort, tiers in (
            ("strict", set(self.config.training.strict_quality_tiers)),
            ("expanded", set(self.config.training.expanded_quality_tiers)),
        ):
            cohort_root = variant_root / cohort
            cohort_root.mkdir(parents=True, exist_ok=True)
            features = feature_frame[feature_frame["quality_tier"].isin(tiers)].copy()
            sequences = sequence_frame[
                sequence_frame["quality_tier"].isin(tiers)
            ].copy()
            features.to_parquet(cohort_root / "gold_features.parquet", index=False)
            features.to_csv(cohort_root / "gold_features.csv", index=False)
            sequences.to_parquet(
                cohort_root / "gold_ibi_sequences.parquet", index=False
            )
            groups = {
                (str(item.participant_id), str(item.phase), str(item.sensor_channel))
                for item in features[["participant_id", "phase", "sensor_channel"]]
                .drop_duplicates()
                .itertuples(index=False)
            }
            result["cohorts"][cohort] = self._readiness(groups)
        pd.DataFrame(payload["exclusions"]).to_csv(
            variant_root / "exclusions.csv", index=False
        )
        return result

    def _readiness(self, groups: set[tuple[str, str, str]]) -> dict[str, Any]:
        training = self.config.training
        return channel_group_readiness(
            groups,
            minimum_training_development_group_coverage=float(
                getattr(
                    training,
                    "minimum_training_development_group_coverage",
                    0.90,
                )
            ),
            minimum_development_groups_per_phase=int(
                getattr(training, "minimum_development_groups_per_phase", 23)
            ),
            minimum_development_participants=int(
                getattr(training, "minimum_development_participants", 26)
            ),
        )

    @staticmethod
    def _feature_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
        metadata = [
            "participant_id",
            "session_id",
            "recording_id",
            "phase",
            "label",
            "label_name",
            "sensor_channel",
            "window_id",
            "window_index",
            "window_seconds",
            "stride_seconds",
            "window_start_utc",
            "window_end_utc",
            "sampling_rate_hz",
            "quality_score",
            "quality_tier",
            "quality_flags",
            "training_eligible",
            "peak_detection_method",
            "peak_detection_status",
            "peak_detection_status_reason",
            "peak_count",
            "physiological_ibi_ratio",
            "interpolated_fraction",
            "source_file",
            "source_checksum",
            "source_silver_path",
            "source_silver_sha256",
            "preprocessing_version",
            "feature_version",
        ]
        return pd.DataFrame(rows, columns=[*metadata, *DEFAULT_PREDICTORS])

    @staticmethod
    def _sequence_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
        if rows:
            return pd.DataFrame(rows)
        return pd.DataFrame(
            columns=[
                "participant_id",
                "phase",
                "sensor_channel",
                "window_id",
                "quality_tier",
                "ibi_values",
                "ibi_mask",
            ]
        )

    @staticmethod
    def _exclusion_row(
        row: dict[str, Any], key: str, window_index: int, reason: str
    ) -> dict[str, Any]:
        return {
            "participant_id": str(row["participant_id"]),
            "phase": str(row["phase"]),
            "sensor_channel": str(row["sensor_channel"]),
            "variant": key,
            "window_index": window_index,
            "exclusion_reason": reason,
            "source_file": str(row.get("source_file", "")),
            "source_checksum": str(row.get("source_checksum", "")),
        }

    def _exclude_source(
        self,
        variants: dict[str, dict[str, list[dict[str, Any]]]],
        row: dict[str, Any],
        reason: str,
    ) -> None:
        for key, payload in variants.items():
            payload["exclusions"].append(self._exclusion_row(row, key, -1, reason))
