"""Focused contracts for participant-safe local classical training."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

import PipelinePython.common.training_canonical as training_canonical
from PipelinePython.common.feature_policy import DEFAULT_PREDICTORS
from PipelinePython.common.settings import PipelinePaths, TrainingConfig
from PipelinePython.common.training_canonical import (
    CanonicalClassicalTrainer,
    HoldoutEvidenceStore,
    _fingerprint,
    _passes,
)


def test_nested_outer_oof_accepts_only_the_represented_development_set() -> None:
    folds = [
        {
            "window_predictions": [
                {
                    "participant_id": "ID-01",
                    "phase": "relax",
                    "sensor_channel": "ppg_green",
                    "label": 0,
                    "window_id": "ID-01-relax-0",
                    "probability": 0.2,
                    "outer_fold": 0,
                }
            ]
        }
    ]

    result = CanonicalClassicalTrainer._nested_outer_oof(
        folds,
        expected_participants={"ID-01"},
    )

    assert result["participant_id"].tolist() == ["ID-01"]


def _governance_trainer(tmp_path):
    config = SimpleNamespace(
        artifacts={"reports_dir": "reports"},
        model_version="tiered-policy-test",
        preprocessing_version="v3.0.0",
        feature_version="v4.0.0_robust_channel_independent",
        training=TrainingConfig(),
        features=SimpleNamespace(
            predictor_default=DEFAULT_PREDICTORS,
            promoted_experimental=(),
        ),
        window=SimpleNamespace(analysis_phase_duration_seconds=240),
    )
    paths = PipelinePaths(
        project_root=tmp_path,
        raw_data_root=tmp_path / "raw",
        anchor_csv_path=tmp_path / "anchors.csv",
        artifacts_root=tmp_path / "artifacts",
        bronze_dir=tmp_path / "bronze",
        silver_dir=tmp_path / "silver",
        gold_dir=tmp_path / "gold",
        gold_features_path=tmp_path / "gold" / "gold_features_v3.parquet",
        eda_dir=tmp_path / "eda",
        validation_dir=tmp_path / "validation",
        models_dir=tmp_path / "models",
        export_dir=tmp_path / "export",
    )
    return CanonicalClassicalTrainer(tmp_path, config, paths)


def test_development_candidate_is_immutable_and_has_no_holdout_metrics(
    tmp_path,
) -> None:
    trainer = _governance_trainer(tmp_path)
    trainer._gold_contract = {
        "features_path": "stale/w240_s240/gold_features.parquet",
        "features_sha256": "a" * 64,
    }
    selected_gold_contract = {
        "features_path": "selected/w060_s030/gold_features.parquet",
        "features_sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
    }
    report = {
        "model_name": "logistic_regression",
        "selected_parameters": {"C": 1.0},
        "feature_order": ["ppi_median_ms"],
        "calibration_input_feature_order": ["ppi_median_ms"],
        "selected_window_seconds": 180,
        "selected_stride_seconds": 10,
        "represented_development_participants": [
            f"ID-{index:02d}" for index in range(1, 29)
        ],
        "missing_development_participants": [],
        "threshold": 0.5,
        "threshold_source": "nested_outer_participant_grouped_oof",
        "development_metrics": {
            "balanced_accuracy": 0.8,
            "accuracy": 0.8,
            "f1": 0.8,
            "roc_auc": 0.8,
        },
        "development_raw_metrics": {},
        "probability_calibration": {
            "method": "platt",
            "version": "test",
        },
        "channel_data_fingerprint": "c" * 64,
        "fingerprint_scope": "development_rows_and_predictors",
        "holdout_contract_fingerprint": "d" * 64,
        "model_contract_fingerprint": "d" * 64,
        "performance_tier": "preferred",
        "converter_preflight": {"status": "passed"},
    }

    bundle = trainer._freeze_development_candidate(
        "ppg_green",
        {"model": "real-fitted-object"},
        report,
        selected_gold_contract,
    )
    manifest = json.loads((bundle / "candidate_manifest.json").read_text())

    assert manifest["artifact_status"] == "development_candidate"
    assert manifest["evaluation_scope"] == "development_only"
    assert manifest["holdout_evaluated"] is False
    assert "holdout_metrics" not in manifest
    assert manifest["gold_contract"] == selected_gold_contract
    assert manifest["gold_contract"] != trainer._gold_contract
    assert (bundle / "model.joblib").is_file()
    assert (bundle / "calibration.json").is_file()
    assert len(manifest["candidate_artifact_fingerprint"]) == 64


def test_local_training_contract_uses_requested_candidates_and_gates() -> None:
    config = TrainingConfig()

    assert config.ml_execution == "local"
    assert config.candidate_order == (
        "logistic_regression",
        "rbf_svm",
        "extra_trees",
        "lightgbm",
        "xgboost",
    )
    assert config.inner_group_splits == 5
    assert config.optuna_trials >= 50
    assert config.minimum_training_development_group_coverage == 0.90
    assert config.minimum_development_participants == 26
    assert config.minimum_research_balanced_accuracy == 0.70
    assert config.minimum_research_f1 == 0.70
    assert config.minimum_research_roc_auc == 0.70
    assert config.preferred_balanced_accuracy == 0.80
    assert config.preferred_f1 == 0.78
    assert config.preferred_roc_auc == 0.83
    assert _passes(
        {"balanced_accuracy": 0.70, "f1": 0.70, "roc_auc": 0.70},
        type("Pipeline", (), {"training": config})(),
    )


def test_feature_order_excludes_non_predictive_ecg_pat_motion_and_quality() -> None:
    payload = {
        "participant_id": ["ID-01", "ID-02"],
        "phase": ["relax", "task"],
        "sensor_channel": ["ppg_ir", "ppg_ir"],
        "label": [0, 1],
        "quality_score": [0.9, 0.8],
        "motion_score": [0.1, 0.2],
        "ecg_rr_mean": [800.0, 760.0],
        "pat_median_ms": [200.0, 210.0],
    }
    payload.update({feature: [1.0, 2.0] for feature in DEFAULT_PREDICTORS})
    frame = pd.DataFrame(payload)
    trainer = object.__new__(CanonicalClassicalTrainer)
    trainer.config = SimpleNamespace(
        features=SimpleNamespace(promoted_experimental=()),
    )

    assert trainer._feature_order(frame) == DEFAULT_PREDICTORS


def test_below_gate_ml_result_requests_colab_dl_handoff() -> None:
    metrics = {"balanced_accuracy": 0.90, "f1": 0.6999, "roc_auc": 0.90}

    assert not _passes(
        metrics,
        type("Pipeline", (), {"training": TrainingConfig()})(),
    )


def test_blocked_gold_readiness_stops_before_loading_training_rows(
    tmp_path, monkeypatch
) -> None:
    trainer = _governance_trainer(tmp_path)
    monkeypatch.setattr(
        training_canonical,
        "validate_gold_input",
        lambda *_args, **_kwargs: {
            "status": "blocked",
            "blocking_reason": "observed_groups=60/64;coverage_complete=false",
        },
    )
    monkeypatch.setattr(
        trainer,
        "_load_gold",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("blocked readiness must stop before Gold rows are loaded")
        ),
    )
    monkeypatch.setattr(trainer, "_write_report", lambda *_args, **_kwargs: None)

    report = trainer.run("ppg_green")

    assert report["release_status"] == "blocked_insufficient_ppg_coverage"
    assert report["holdout_evaluated"] is False
    assert "coverage_complete=false" in report["blocking_reason"]


def test_release_policy_does_not_change_model_contract_fingerprint() -> None:
    trainer = object.__new__(CanonicalClassicalTrainer)
    trainer.config = SimpleNamespace(
        preprocessing_version="v3.0.0",
        feature_version="v4.0.0_robust_channel_independent",
        training=TrainingConfig(),
    )
    kwargs = {
        "sensor_channel": "ppg_green",
        "selected_model": "logistic_regression",
        "selected_parameters": {"C": 1.0},
        "feature_order": ("ppi_median_ms",),
        "data_fingerprint": "a" * 64,
        "decision_threshold": 0.485,
    }

    model_fingerprint = trainer._holdout_contract_fingerprint(**kwargs)
    policy_fingerprint = trainer._release_policy_fingerprint()
    trainer.config = SimpleNamespace(
        preprocessing_version="v3.0.0",
        feature_version="v4.0.0_robust_channel_independent",
        training=TrainingConfig(performance_policy_version="ml_dl_tiered_v2"),
    )

    assert trainer._holdout_contract_fingerprint(**kwargs) == model_fingerprint
    assert trainer._release_policy_fingerprint() != policy_fingerprint


def test_red_only_change_does_not_change_green_or_ir_fingerprints() -> None:
    frame = pd.DataFrame(
        {
            "participant_id": ["ID-01", "ID-01", "ID-01"],
            "phase": ["relax", "relax", "relax"],
            "sensor_channel": ["ppg_green", "ppg_ir", "ppg_red"],
            "label": [0, 0, 0],
            "ppi_median_ms": [800.0, 810.0, 900.0],
        }
    )
    before = {
        channel: _fingerprint(
            frame[frame["sensor_channel"] == channel], ("ppi_median_ms",)
        )
        for channel in ("ppg_green", "ppg_ir")
    }
    frame.loc[frame["sensor_channel"] == "ppg_red", "ppi_median_ms"] = 750.0
    after = {
        channel: _fingerprint(
            frame[frame["sensor_channel"] == channel], ("ppi_median_ms",)
        )
        for channel in ("ppg_green", "ppg_ir")
    }

    assert after == before


def test_candidate_factory_builds_all_local_model_families() -> None:
    trainer = object.__new__(CanonicalClassicalTrainer)
    trainer.config = SimpleNamespace(training=TrainingConfig())

    assert tuple(trainer._candidate_models()) == (
        "logistic_regression",
        "rbf_svm",
        "extra_trees",
        "lightgbm",
        "xgboost",
    )


def test_reused_test_fingerprint_includes_frozen_oof_threshold() -> None:
    trainer = object.__new__(CanonicalClassicalTrainer)
    trainer.config = SimpleNamespace(
        preprocessing_version="v3.0.0",
        feature_version="v4.0.0_robust_channel_independent",
        training=TrainingConfig(),
    )
    shared = {
        "sensor_channel": "ppg_red",
        "selected_model": "extra_trees",
        "selected_parameters": {"n_estimators": 300},
        "feature_order": ("ppi_median_ms",),
        "data_fingerprint": "a" * 64,
    }

    lower = trainer._holdout_contract_fingerprint(**shared, decision_threshold=0.45)
    higher = trainer._holdout_contract_fingerprint(**shared, decision_threshold=0.55)

    assert lower != higher


def test_legacy_holdout_evidence_is_reused_without_second_access(tmp_path) -> None:
    trainer = _governance_trainer(tmp_path)
    kwargs = {
        "sensor_channel": "ppg_green",
        "selected_model": "logistic_regression",
        "selected_parameters": {"C": 1.0},
        "feature_order": ("ppi_median_ms",),
        "data_fingerprint": "a" * 64,
        "decision_threshold": 0.485,
    }
    model_fingerprint = trainer._holdout_contract_fingerprint(**kwargs)
    legacy_fingerprint = trainer._legacy_holdout_contract_fingerprint(**kwargs)
    historical = tmp_path / "reports" / "model_eval" / "old"
    historical.mkdir(parents=True)
    payload = {
        "sensor_channel": "ppg_green",
        "holdout_contract_fingerprint": legacy_fingerprint,
        "holdout_evaluated": True,
        "holdout_metrics": {
            "balanced_accuracy": 0.75,
            "f1": 0.75,
            "roc_auc": 0.8125,
        },
    }
    (historical / "ppg_green_full.json").write_text(json.dumps(payload))

    reused = trainer._find_prior_holdout_evidence(
        "ppg_green", model_fingerprint, legacy_fingerprint
    )

    assert reused == payload
    assert trainer._has_any_prior_holdout_evidence("ppg_green") is True
    assert (
        trainer.performance_policy.assess(
            reused["holdout_metrics"]
        ).performance_tier.value
        == "research_usable"
    )


def test_unfingerprinted_holdout_report_is_historical_not_governed(tmp_path) -> None:
    trainer = _governance_trainer(tmp_path)
    historical = tmp_path / "reports" / "model_eval" / "legacy"
    historical.mkdir(parents=True)
    report_path = historical / "ppg_ir_full.json"
    report_path.write_text(
        json.dumps(
            {
                "sensor_channel": "ppg_ir",
                "holdout_evaluated": True,
                "holdout_metrics": {"balanced_accuracy": 0.95},
            }
        )
    )

    assert trainer._has_any_prior_holdout_evidence("ppg_ir") is False
    assert trainer._find_prior_holdout_evidence("ppg_ir", "a" * 64) is None
    assert trainer._historical_unverifiable_holdout_reports("ppg_ir") == (
        "reports/model_eval/legacy/ppg_ir_full.json",
    )


def test_current_fingerprinted_evidence_blocks_a_different_holdout_contract(
    tmp_path,
) -> None:
    trainer = _governance_trainer(tmp_path)
    trainer.holdout_evidence.write_once(
        "ppg_ir",
        "a" * 64,
        {
            "sensor_channel": "ppg_ir",
            "contract_fingerprint": "a" * 64,
            "holdout_evaluated": True,
        },
    )

    assert trainer._has_any_prior_holdout_evidence("ppg_ir") is True
    assert trainer._find_prior_holdout_evidence("ppg_ir", "b" * 64) is None


def test_local_ml_trigger_defers_checksum_bound_handoff_to_parent(tmp_path) -> None:
    config = SimpleNamespace(
        artifacts={"reports_dir": "reports"},
        model_version="test-model",
        training=TrainingConfig(),
    )
    paths = PipelinePaths(
        project_root=tmp_path,
        raw_data_root=tmp_path / "raw",
        anchor_csv_path=tmp_path / "anchors.csv",
        artifacts_root=tmp_path / "artifacts",
        bronze_dir=tmp_path / "bronze",
        silver_dir=tmp_path / "silver",
        gold_dir=tmp_path / "gold",
        gold_features_path=tmp_path / "gold" / "gold_features_v3.parquet",
        eda_dir=tmp_path / "eda",
        validation_dir=tmp_path / "validation",
        models_dir=tmp_path / "models",
        export_dir=tmp_path / "export",
    )
    trainer = CanonicalClassicalTrainer(tmp_path, config, paths)

    trainer._write_report(
        "ppg_ir",
        {
            "model_name": "xgboost",
            "development_metrics": {"balanced_accuracy": 0.79},
            "requires_dl": True,
            "trigger_reason": (
                "local_ml_development_metrics_below_minimum_research_gate"
            ),
            "failing_minimum_metrics": ["f1"],
            "holdout_evaluated": False,
            "release_status": "blocked_development_gate",
        },
    )

    trigger = json.loads(
        (
            tmp_path
            / "reports"
            / "model_eval"
            / "ml"
            / "test-model"
            / "ppg_ir"
            / "local_ml_dl_trigger.json"
        ).read_text()
    )
    assert trigger["requires_dl"] is True
    assert trigger["trigger_reason"] == (
        "local_ml_development_metrics_below_minimum_research_gate"
    )
    assert trigger["failing_minimum_metrics"] == ["f1"]
    assert trigger["next_owner"] == "parent_cli_combined_colab_handoff"
    assert "gold_features_path" not in trigger
    assert (
        tmp_path
        / "reports"
        / "model_eval"
        / "ml"
        / "test-model"
        / "ppg_ir"
        / "participant_fold_metrics.parquet"
    ).is_file()
    channel_dir = tmp_path / "reports" / "model_eval" / "ml" / "test-model" / "ppg_ir"
    for artifact in (
        "window_ablation.csv",
        "quality_cohort_ablation.csv",
        "feature_group_ablation.csv",
        "nested_oof_window_predictions.csv",
        "nested_oof_session_predictions.csv",
        "participant_metrics.csv",
        "feature_stability.csv",
        "calibration_report.json",
        "model_selection.json",
    ):
        assert (channel_dir / artifact).is_file()


def test_report_parquet_serializes_nested_parameters_as_canonical_json(
    tmp_path,
) -> None:
    config = SimpleNamespace(
        artifacts={"reports_dir": "reports"},
        model_version="serialization-test",
        training=TrainingConfig(),
    )
    paths = PipelinePaths(
        project_root=tmp_path,
        raw_data_root=tmp_path / "raw",
        anchor_csv_path=tmp_path / "anchors.csv",
        artifacts_root=tmp_path / "artifacts",
        bronze_dir=tmp_path / "bronze",
        silver_dir=tmp_path / "silver",
        gold_dir=tmp_path / "gold",
        gold_features_path=tmp_path / "gold" / "gold_features_v3.parquet",
        eda_dir=tmp_path / "eda",
        validation_dir=tmp_path / "validation",
        models_dir=tmp_path / "models",
        export_dir=tmp_path / "export",
    )
    trainer = CanonicalClassicalTrainer(tmp_path, config, paths)
    trainer._write_report(
        "ppg_green",
        {
            "model_name": "extra_trees",
            "candidate_results": {
                "extra_trees": {
                    "folds": [
                        {
                            "balanced_accuracy": 0.8,
                            "inner_cv_best_parameters": {
                                "max_features": "sqrt",
                                "max_depth": None,
                            },
                            "feature_order": ["ppi_median_ms", "rmssd_ms"],
                        }
                    ]
                }
            },
        },
    )

    frame = pd.read_parquet(
        tmp_path
        / "reports"
        / "model_eval"
        / "ml"
        / "serialization-test"
        / "ppg_green"
        / "participant_fold_metrics.parquet"
    )
    assert frame.loc[0, "inner_cv_best_parameters"] == (
        '{"max_depth":null,"max_features":"sqrt"}'
    )
    assert frame.loc[0, "feature_order"] == '["ppi_median_ms","rmssd_ms"]'


def test_holdout_evidence_store_is_atomic_and_reusable_after_report_failure(
    tmp_path,
) -> None:
    store = HoldoutEvidenceStore(tmp_path / "governance")
    payload = {
        "sensor_channel": "ppg_ir",
        "contract_fingerprint": "f" * 64,
        "holdout_metrics": {"balanced_accuracy": 0.81},
    }

    first = store.write_once("ppg_ir", "f" * 64, payload)
    second = store.load("ppg_ir", "f" * 64)

    assert first == second == payload
    assert store.write_once("ppg_ir", "f" * 64, payload) == payload
    conflicting = {**payload, "holdout_metrics": {"balanced_accuracy": 0.99}}
    try:
        store.write_once("ppg_ir", "f" * 64, conflicting)
    except RuntimeError as exc:
        assert "holdout_evidence_immutability_violation" in str(exc)
    else:
        raise AssertionError("conflicting holdout evidence must fail closed")


def test_holdout_claim_is_written_before_read_and_cannot_be_reopened(tmp_path) -> None:
    store = HoldoutEvidenceStore(tmp_path / "governance")

    store.claim("ppg_red", "a" * 64, {"run_id": "20260829T000000Z"})

    claim = store.claim_path_for("ppg_red")
    assert claim.is_file()
    try:
        store.claim("ppg_red", "b" * 64, {"run_id": "20260829T010000Z"})
    except RuntimeError as exc:
        assert "holdout_access_already_claimed" in str(exc)
    else:
        raise AssertionError("a claimed reused-test channel must fail closed")
