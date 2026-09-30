"""Development-only benchmark for the independent PPG detector registry."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .hrv_agreement import HrvAgreementAnalyzer
from .peak_detection import PpgPeakDetection, PpgPeakDetector
from .config import load_sqa_config
from .peak_validation import (
    PpgEcgValidator,
    apply_morphology_status_policy,
    evaluate_morphology_policy_ablation,
)
from .settings import PipelineConfig, PipelinePaths

DEVELOPMENT_PARTICIPANTS = tuple(f"ID-{index:02d}" for index in range(1, 29))
HOLDOUT_PARTICIPANTS = frozenset(f"ID-{index:02d}" for index in range(29, 33))
PHASES = ("relax", "task")
CHANNELS = ("ppg_green", "ppg_ir", "ppg_red")
OUTER_GROUPED_FOLDS = tuple(
    tuple(f"ID-{index:02d}" for index in range(start, start + 4))
    for start in range(1, 29, 4)
)
_METHOD_COMPLEXITY = {
    "neurokit2_ppg_elgendi": 1,
    "neurokit2_ppg_bishop": 2,
    "neurokit2_ppg_charlton_msptdfast": 3,
    "scipy_find_peaks_adaptive": 4,
    "scipy_legacy_green_native": 5,
}


class _RegistryAuditDetector:
    """Expose exactly one registry candidate; never fall back silently."""

    def __init__(self, detector: PpgPeakDetector, method: str) -> None:
        self.detector = detector
        self.method = method

    def detect(
        self, signal: np.ndarray, sampling_rate_hz: float, channel: str
    ) -> PpgPeakDetection:
        return self.detector.evaluate_candidate(
            signal,
            sampling_rate_hz,
            channel,
            self.method,
        )

    def minimum_sample_count(self, sampling_rate_hz: float) -> int:
        return self.detector.minimum_sample_count(sampling_rate_hz)


class PeakDetectorAudit:
    """Rank PPG detectors on ID-01..ID-28 with an explicit no-go outcome."""

    def __init__(
        self,
        project_root: Path,
        config: PipelineConfig,
        paths: PipelinePaths,
    ) -> None:
        self.root = project_root.resolve()
        self.config = config
        self.paths = paths
        self.output_dir = paths.validation_dir / "detector_selection"

    def _run_identity(self) -> dict[str, str]:
        """Return the active run identity without inventing missing evidence."""

        run_id = str(self.config.artifacts.get("run_id", ""))
        contract_path = self.paths.artifacts_root / "run_contract_fingerprint.json"
        if contract_path.is_file():
            payload = json.loads(contract_path.read_text(encoding="utf-8"))
            contract_sha256 = hashlib.sha256(contract_path.read_bytes()).hexdigest()
            return {
                "run_id": run_id or str(payload.get("run_id", "")),
                "source_tree_sha256": str(
                    payload.get("source_tree_sha256", "UNAVAILABLE")
                ),
                "config_fingerprint": str(
                    payload.get("config_fingerprint", "UNAVAILABLE")
                ),
                "run_contract_sha256": contract_sha256,
            }
        config_payload = json.dumps(
            asdict(self.config), sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        return {
            "run_id": run_id,
            "source_tree_sha256": "UNAVAILABLE:run_contract_missing",
            "config_fingerprint": hashlib.sha256(config_payload).hexdigest(),
            "run_contract_sha256": "UNAVAILABLE:run_contract_missing",
        }

    def _candidate_specs(self, channel: str) -> list[dict[str, Any]]:
        """Return registry methods plus the configured SciPy prominence grid."""

        del channel
        specs: list[dict[str, Any]] = []
        for method in self.config.peak_detection.candidate_methods:
            if method == "scipy_find_peaks_adaptive":
                for (
                    multiplier
                ) in self.config.peak_detection.audit_prominence_mad_multipliers:
                    specs.append(
                        {
                            "candidate_method": method,
                            "prominence_mad_multiplier": float(multiplier),
                        }
                    )
            else:
                specs.append(
                    {
                        "candidate_method": method,
                        "prominence_mad_multiplier": None,
                    }
                )
        return specs

    def run(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        run_identity = self._run_identity()
        detail_rows: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        candidate_batches: list[
            tuple[str, str, float | None, list[dict[str, Any]]]
        ] = []
        shared_ecg_cache: dict[Path, Any] = {}
        shared_ppg_detection_cache: dict[tuple[Path, str, str], Any] = {}
        for channel in CHANNELS:
            configured = self.config.peak_detection.for_channel(channel)
            for spec in self._candidate_specs(channel):
                candidate_method = str(spec["candidate_method"])
                candidate_config = self.config.peak_detection
                if spec["prominence_mad_multiplier"] is not None:
                    candidate_channel = replace(
                        configured,
                        prominence_mad_multiplier=float(
                            spec["prominence_mad_multiplier"]
                        ),
                    )
                    candidate_config = replace(
                        candidate_config,
                        channels={
                            **candidate_config.channels,
                            channel: candidate_channel,
                        },
                    )
                detector = PpgPeakDetector(candidate_config)
                # A PPG path occurs once per candidate/channel.  Do not retain
                # all cropped waveform DataFrames across detector candidates;
                # the audit must be bounded even when the registry expands.
                candidate_ppg_cache: dict[Path, Any] = {}
                validator = PpgEcgValidator(
                    self.paths,
                    self.config,
                    ppg_peak_detector=_RegistryAuditDetector(
                        detector,
                        candidate_method,
                    ),
                    write_detail_artifacts=False,
                    compute_confidence_intervals=False,
                    ecg_cache=shared_ecg_cache,
                    ppg_cache=candidate_ppg_cache,
                    ppg_detection_cache=shared_ppg_detection_cache,
                    allowed_participants=frozenset(DEVELOPMENT_PARTICIPANTS),
                )
                # Collect the diagnostic view once, then derive the hard view
                # from the same immutable peak stream. This mirrors canonical
                # morphology ablation without detecting peaks twice.
                validator._active_morphology_policy = "diagnostic"
                validator._active_morphology_policy_by_channel = {
                    value: "diagnostic" for value in CHANNELS
                }
                candidate_rows: list[dict[str, Any]] = []
                for participant_id in DEVELOPMENT_PARTICIPANTS:
                    for phase in PHASES:
                        row = validator.validate_combination(
                            participant_id,
                            phase,
                            channel,
                        )
                        row["candidate_method"] = candidate_method
                        row["prominence_mad_multiplier"] = spec[
                            "prominence_mad_multiplier"
                        ]
                        row.update(run_identity)
                        candidate_rows.append(row)
                candidate_batches.append(
                    (
                        channel,
                        candidate_method,
                        (
                            float(spec["prominence_mad_multiplier"])
                            if spec["prominence_mad_multiplier"] is not None
                            else None
                        ),
                        candidate_rows,
                    )
                )
                del validator
                candidate_ppg_cache.clear()
                gc.collect()

        candidate_batches, morphology_ablations = self._apply_morphology_policy(
            candidate_batches
        )
        detail_rows.extend(
            row
            for _, _, _, candidate_rows in candidate_batches
            for row in candidate_rows
        )

        analyzer = HrvAgreementAnalyzer(self.config.hrv_agreement)
        frozen_ranges = analyzer.development_reference_ranges(pd.DataFrame(detail_rows))
        for channel, candidate_method, multiplier, candidate_rows in candidate_batches:
            summary = self._summarize(
                channel,
                candidate_method,
                multiplier,
                candidate_rows,
            )
            summary.update(self._hrv_candidate_summary(candidate_rows, frozen_ranges))
            summaries.append(summary)

        selected = {
            channel: self._select_channel(channel, summaries) for channel in CHANNELS
        }
        for channel, result in selected.items():
            baseline = self._baseline_summary(channel, summaries)
            result["baseline_valid_fraction"] = baseline["valid_fraction"]
            result["baseline_physiological_ibi_ratio"] = baseline[
                "median_physiological_ibi_ratio"
            ]
            result["coverage_not_decreased"] = bool(
                result["valid_fraction"] >= baseline["valid_fraction"]
            )
            candidate_ratio = result["median_physiological_ibi_ratio"]
            baseline_ratio = baseline["median_physiological_ibi_ratio"]
            result["physiological_ibi_ratio_not_decreased"] = bool(
                candidate_ratio is not None
                and baseline_ratio is not None
                and candidate_ratio >= baseline_ratio
            )
            result["detector_proven"] = bool(
                result["coverage_not_decreased"]
                and result["physiological_ibi_ratio_not_decreased"]
                and result["valid_fraction"] >= 0.80
                and result["median_beat_f1"] is not None
                and result["median_beat_f1"] >= 0.80
                and result["median_ibi_mae_ms"] is not None
                and result["median_ibi_mae_ms"] <= 100.0
                and result["median_hr_mae_bpm"] is not None
                and result["median_hr_mae_bpm"] <= 5.0
            )
            result["decision"] = (
                "detector_proven"
                if result["detector_proven"]
                else "detector_not_proven"
            )
            result["decision_role"] = "beat_diagnostic_only"

        selection_evidence_sufficient = any(
            result["hrv_development_decision"] == "GO" for result in selected.values()
        )
        selection_contract_frozen = bool(
            set(selected) == set(CHANNELS)
            and all(result["matches_frozen_config"] for result in selected.values())
            and all(
                int(result.get("valid_combinations", 0)) > 0
                for result in selected.values()
            )
        )
        frozen_config_verified = selection_contract_frozen
        detector_proven = all(result["detector_proven"] for result in selected.values())
        payload = {
            "schema_version": "ppg-peak-detector-audit-v4",
            "primary_selection_metric": "hrv_prv_time_domain_hardened_composite",
            "beat_diagnostics_control_primary_go_no_go": False,
            "selection_scope": "development_only",
            "development_participants": list(DEVELOPMENT_PARTICIPANTS),
            "holdout_participants_accessed": [],
            "holdout_participants_forbidden": sorted(HOLDOUT_PARTICIPANTS),
            "phases": list(PHASES),
            "candidate_registry": list(self.config.peak_detection.candidate_methods),
            "candidate_morphology_ablations": morphology_ablations,
            "ranking": [
                "hrv_development_evidence_status_desc",
                "hrv_minimum_composite_desc",
                "hrv_mean_composite_desc",
                "hrv_valid_development_rows_desc",
                "hrv_fold_minimum_composite_desc",
                "detector_complexity_asc",
            ],
            "metric_aggregation": "participant_median_then_outer_grouped_fold_median",
            "candidates": summaries,
            "selected": selected,
            "detector_proven": detector_proven,
            "overall_status": (
                "detector_proven" if detector_proven else "detector_not_proven"
            ),
            "beat_diagnostic_status": (
                "detector_proven" if detector_proven else "detector_not_proven"
            ),
            "deployment_decision": "DIAGNOSTIC_ONLY",
            "frozen_config_verified": frozen_config_verified,
            "selection_contract_frozen": selection_contract_frozen,
            "selection_evidence_sufficient": selection_evidence_sufficient,
            "config_update_required": not frozen_config_verified,
            "run_id": run_identity["run_id"],
            "source_tree_sha256": run_identity["source_tree_sha256"],
            "config_fingerprint": run_identity["config_fingerprint"],
            "run_contract_sha256": run_identity["run_contract_sha256"],
        }
        detail_frame = pd.DataFrame(detail_rows)
        detail_frame.to_csv(
            self.output_dir / "detector_selection_detail.csv",
            index=False,
        )
        pd.DataFrame(summaries).to_csv(
            self.output_dir / "detector_selection_metrics.csv",
            index=False,
        )
        self._atomic_json(self.output_dir / "detector_selection.json", payload)
        hrv_output = self.paths.validation_dir / "hrv_agreement"
        hrv_output.mkdir(parents=True, exist_ok=True)
        scalar_summaries = [
            {
                key: value
                for key, value in row.items()
                if key not in {"outer_grouped_folds", "hrv_metrics"}
            }
            for row in summaries
        ]
        pd.DataFrame(scalar_summaries).to_csv(
            hrv_output / "hrv_detector_benchmark.csv", index=False
        )
        self._atomic_json(
            hrv_output / "hrv_detector_benchmark.json",
            {
                "schema_version": "hrv-detector-benchmark-v1",
                "selection_scope": "ID-01..ID-28",
                "holdout_participants_accessed": [],
                "ranking": payload["ranking"],
                "candidates": summaries,
                "selected": selected,
                **run_identity,
            },
        )
        return payload

    @staticmethod
    def _candidate_key(
        method: str, multiplier: float | None
    ) -> tuple[str, float | None]:
        return method, None if multiplier is None else float(multiplier)

    def _apply_morphology_policy(
        self,
        batches: list[tuple[str, str, float | None, list[dict[str, Any]]]],
    ) -> tuple[
        list[tuple[str, str, float | None, list[dict[str, Any]]]],
        list[dict[str, Any]],
    ]:
        """Apply candidate-specific development morphology ablation."""

        grouped: dict[tuple[str, float | None], dict[str, list[dict[str, Any]]]] = {}
        for channel, method, multiplier, rows in batches:
            grouped.setdefault(self._candidate_key(method, multiplier), {})[channel] = (
                rows
            )

        sqa = load_sqa_config()
        requested = str(sqa.get("morphology_policy", "hard")).lower()
        if requested not in {"hard", "diagnostic"}:
            raise ValueError(f"invalid_morphology_policy:{requested}")
        effective: dict[tuple[str, float | None, str], list[dict[str, Any]]] = {}
        ablations: list[dict[str, Any]] = []
        for (method, multiplier), rows_by_channel in grouped.items():
            diagnostic = pd.DataFrame(
                [
                    row
                    for channel in CHANNELS
                    for row in rows_by_channel.get(channel, [])
                ]
            )
            hard = apply_morphology_status_policy(diagnostic, "hard")
            ablation = evaluate_morphology_policy_ablation(
                hard,
                diagnostic,
                maximum_f1_drop=float(sqa.get("morphology_ablation_max_f1_drop", 0.01)),
                maximum_ibi_mae_increase_ms=float(
                    sqa.get("morphology_ablation_max_ibi_mae_increase_ms", 5.0)
                ),
                maximum_hr_mae_increase_bpm=float(
                    sqa.get("morphology_ablation_max_hr_mae_increase_bpm", 0.5)
                ),
            )
            policy_by_channel = {
                row["sensor_channel"]: (
                    "diagnostic"
                    if requested == "diagnostic" and row["promotion_eligible"]
                    else "hard"
                )
                for row in ablation["channels"]
            }
            for channel in CHANNELS:
                source = (
                    diagnostic if policy_by_channel[channel] == "diagnostic" else hard
                )
                selected = source[source["sensor_channel"].eq(channel)].copy()
                selected["candidate_morphology_policy"] = policy_by_channel[channel]
                effective[(method, multiplier, channel)] = selected.to_dict(
                    orient="records"
                )
            ablations.append(
                {
                    "candidate_method": method,
                    "prominence_mad_multiplier": multiplier,
                    "requested_policy": requested,
                    "effective_policy_by_channel": policy_by_channel,
                    **ablation,
                }
            )

        return (
            [
                (
                    channel,
                    method,
                    multiplier,
                    effective[(method, multiplier, channel)],
                )
                for channel, method, multiplier, _ in batches
            ],
            ablations,
        )

    def _hrv_candidate_summary(
        self,
        rows: list[dict[str, Any]],
        development_ranges: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        analyzer = HrvAgreementAnalyzer(self.config.hrv_agreement)
        frame = pd.DataFrame(rows)
        payload, metrics = analyzer.evaluate(
            frame,
            evaluation_scope="selection",
            development_ranges=development_ranges,
        )
        channel = payload["channels"][str(frame["sensor_channel"].iloc[0])]
        fold_scores: list[float] = []
        for participants in OUTER_GROUPED_FOLDS:
            fold = frame[frame["participant_id"].isin(participants)]
            _, fold_metrics = analyzer.evaluate(
                fold,
                evaluation_scope="selection",
                development_ranges=development_ranges,
            )
            computed = fold_metrics[fold_metrics["status"].eq("COMPUTED")]
            if len(computed) == 18:
                fold_scores.append(float(computed["composite_score"].min()))
        return {
            "hrv_development_decision": channel["decision"],
            "hrv_minimum_composite": channel["minimum_composite"],
            "hrv_mean_composite": channel["mean_composite"],
            "hrv_valid_development_rows": channel["valid_pooled_rows"],
            "hrv_valid_relax_rows": channel["valid_relax_rows"],
            "hrv_valid_task_rows": channel["valid_task_rows"],
            "hrv_fold_minimum_composite": (
                float(min(fold_scores)) if fold_scores else None
            ),
            "hrv_fold_composite_std": (
                float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else None
            ),
            "hrv_decision_reasons": channel["reasons"],
            "ecg_reference_range_fingerprint": payload[
                "ecg_reference_range_fingerprint"
            ],
            "hrv_metrics": metrics.to_dict("records"),
        }

    @staticmethod
    def _numeric_median(
        frame: pd.DataFrame, column: str, default: float | None
    ) -> float | None:
        if column not in frame:
            return default
        values = pd.to_numeric(frame[column], errors="coerce")
        values = values[np.isfinite(values)]
        return float(values.median()) if not values.empty else default

    @classmethod
    def _participant_metric_median(
        cls,
        frame: pd.DataFrame,
        column: str,
        default: float | None = None,
    ) -> float | None:
        """Return a participant-weighted metric without inventing missing values."""

        if column not in frame or "participant_id" not in frame:
            return default
        valid = frame[frame["status"].eq("ok")].copy()
        if valid.empty:
            return default
        valid[column] = pd.to_numeric(valid[column], errors="coerce")
        valid = valid[np.isfinite(valid[column])]
        if valid.empty:
            return default
        participant_values = valid.groupby("participant_id", sort=True)[column].median()
        return (
            float(participant_values.median())
            if not participant_values.empty
            else default
        )

    @classmethod
    def _outer_fold_metrics(cls, frame: pd.DataFrame) -> list[dict[str, Any]]:
        """Evaluate frozen detector rows on seven predefined participant folds."""

        fold_metrics: list[dict[str, Any]] = []
        for fold_number, participants in enumerate(OUTER_GROUPED_FOLDS, start=1):
            fold_frame = frame[frame["participant_id"].isin(participants)]
            valid = fold_frame[fold_frame["status"].eq("ok")]
            fold_metrics.append(
                {
                    "fold_id": f"outer-{fold_number:02d}",
                    "participants": list(participants),
                    "evaluated_combinations": int(len(fold_frame)),
                    "valid_combinations": int(len(valid)),
                    "valid_fraction": (
                        float(len(valid) / len(fold_frame)) if len(fold_frame) else 0.0
                    ),
                    "participant_count": int(
                        valid["participant_id"].nunique() if not valid.empty else 0
                    ),
                    "median_beat_f1": cls._participant_metric_median(
                        fold_frame, "beat_f1"
                    ),
                    "median_ibi_mae_ms": cls._participant_metric_median(
                        fold_frame, "ibi_mae_ms"
                    ),
                    "median_hr_mae_bpm": cls._participant_metric_median(
                        fold_frame, "hr_mae_bpm"
                    ),
                    "pat_stability_mad_ms": cls._participant_metric_median(
                        fold_frame, "pat_mad_ms"
                    ),
                    "median_physiological_ibi_ratio": cls._participant_metric_median(
                        fold_frame, "ppg_physiological_ibi_ratio"
                    ),
                }
            )
        return fold_metrics

    @classmethod
    def _summarize(
        cls,
        channel: str,
        candidate_method: str,
        multiplier: float | None,
        rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        frame = pd.DataFrame(rows)
        valid = frame[frame.get("status", pd.Series(dtype=str)) == "ok"]
        outer_folds = cls._outer_fold_metrics(frame)
        outer_frame = pd.DataFrame(outer_folds)
        valid_fraction = float(len(valid) / len(frame)) if len(frame) else 0.0
        unavailable = int(
            frame.get("detection_fallback_reason", pd.Series(dtype=str))
            .astype(str)
            .str.contains("UNAVAILABLE|unavailable", case=False, regex=True)
            .sum()
        )
        consensus_fraction = (
            float(
                frame.get("ecg_reference_status", pd.Series(dtype=str))
                .astype(str)
                .eq("CONSENSUS")
                .mean()
            )
            if len(frame)
            else 0.0
        )
        return {
            "sensor_channel": channel,
            "candidate_method": candidate_method,
            "prominence_mad_multiplier": multiplier,
            "evaluated_combinations": len(frame),
            "valid_combinations": len(valid),
            "valid_fraction": valid_fraction,
            "participant_count": int(frame["participant_id"].nunique())
            if "participant_id" in frame
            else 0,
            "participant_count_with_valid_metrics": int(
                valid["participant_id"].nunique()
                if not valid.empty and "participant_id" in valid
                else 0
            ),
            "unavailable_combinations": unavailable,
            "median_beat_f1": cls._participant_metric_median(valid, "beat_f1"),
            "median_ibi_mae_ms": cls._participant_metric_median(valid, "ibi_mae_ms"),
            "median_hr_mae_bpm": cls._participant_metric_median(valid, "hr_mae_bpm"),
            "pat_stability_mad_ms": cls._participant_metric_median(valid, "pat_mad_ms"),
            "median_physiological_ibi_ratio": cls._participant_metric_median(
                valid, "ppg_physiological_ibi_ratio"
            ),
            "ecg_reference_consensus_fraction": consensus_fraction,
            "detector_complexity": _METHOD_COMPLEXITY.get(candidate_method, 99),
            "outer_grouped_folds": outer_folds,
            "outer_grouped_evaluated_folds": len(outer_folds),
            "outer_grouped_valid_fraction_median": cls._numeric_median(
                outer_frame, "valid_fraction", 0.0
            ),
            "outer_grouped_valid_fraction_min": float(
                pd.to_numeric(outer_frame["valid_fraction"], errors="coerce").min()
            ),
            "outer_grouped_median_beat_f1": cls._numeric_median(
                outer_frame, "median_beat_f1", None
            ),
            "outer_grouped_median_ibi_mae_ms": cls._numeric_median(
                outer_frame, "median_ibi_mae_ms", None
            ),
            "outer_grouped_median_hr_mae_bpm": cls._numeric_median(
                outer_frame, "median_hr_mae_bpm", None
            ),
            "outer_grouped_pat_stability_mad_median": cls._numeric_median(
                outer_frame, "pat_stability_mad_ms", None
            ),
            "outer_grouped_median_physiological_ibi_ratio": cls._numeric_median(
                outer_frame, "median_physiological_ibi_ratio", None
            ),
        }

    def _baseline_summary(
        self, channel: str, summaries: list[dict[str, Any]]
    ) -> dict[str, Any]:
        configured = self.config.peak_detection.primary_method
        candidates = [
            row
            for row in summaries
            if row["sensor_channel"] == channel
            and row["candidate_method"] == configured
        ]
        if not candidates:
            raise RuntimeError(
                f"configured_detector_missing_from_registry:{configured}"
            )
        return candidates[0]

    def _select_channel(
        self, channel: str, summaries: list[dict[str, Any]]
    ) -> dict[str, Any]:
        configured_method = self.config.peak_detection.selected_method_for(channel)
        configured_multiplier = float(
            self.config.peak_detection.for_channel(channel).prominence_mad_multiplier
        )
        candidates = [row for row in summaries if row["sensor_channel"] == channel]
        if not candidates:
            raise RuntimeError(f"no_detector_candidates:{channel}")
        ranked = sorted(
            candidates,
            key=lambda row: (
                -self._rank_value(
                    {
                        "GO": 2.0,
                        "NO_GO": 1.0,
                        "INSUFFICIENT_EVIDENCE": 0.0,
                    }.get(str(row.get("hrv_development_decision")), -1.0),
                    descending=True,
                ),
                -self._rank_value(
                    row.get("hrv_minimum_composite"),
                    descending=True,
                ),
                -self._rank_value(
                    row.get("hrv_mean_composite"),
                    descending=True,
                ),
                -self._rank_value(
                    row.get("hrv_valid_development_rows"),
                    descending=True,
                ),
                -self._rank_value(
                    row.get("hrv_fold_minimum_composite"),
                    descending=True,
                ),
                int(row["detector_complexity"]),
                str(row["candidate_method"]),
                float(row["prominence_mad_multiplier"])
                if row["prominence_mad_multiplier"] is not None
                else -1.0,
            ),
        )
        best_candidate = ranked[0]
        if best_candidate.get("hrv_development_decision") == "GO":
            selected = best_candidate
            selection_reason = "best_development_primary_time_domain_go"
        else:
            selected = self._baseline_summary(channel, summaries)
            selection_reason = (
                "no_candidate_reached_primary_time_domain_go_retained_baseline"
            )
        matches_frozen = selected["candidate_method"] == configured_method and (
            selected["candidate_method"] != "scipy_find_peaks_adaptive"
            or np.isclose(
                float(selected["prominence_mad_multiplier"]),
                configured_multiplier,
            )
        )
        return {
            **selected,
            "configured_method": configured_method,
            "configured_prominence_mad_multiplier": configured_multiplier,
            "matches_frozen_config": bool(matches_frozen),
            "selection_metric_source": "participant_level_outer_grouped_validation",
            "selection_reason": selection_reason,
            "best_candidate_method": best_candidate["candidate_method"],
            "best_candidate_prominence_mad_multiplier": best_candidate[
                "prominence_mad_multiplier"
            ],
            "best_candidate_development_decision": best_candidate[
                "hrv_development_decision"
            ],
        }

    @staticmethod
    def _rank_value(value: Any, *, descending: bool) -> float:
        """Map absent metrics to the losing side without inventing a value."""

        try:
            number = float(value)
        except (TypeError, ValueError):
            return float("-inf" if descending else "inf")
        return number if np.isfinite(number) else float("-inf" if descending else "inf")

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(
                PeakDetectorAudit._json_safe(payload),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): PeakDetectorAudit._json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [PeakDetectorAudit._json_safe(item) for item in value]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, Path):
            return str(value)
        return value
