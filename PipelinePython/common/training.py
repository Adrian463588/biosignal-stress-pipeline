from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier

    _LGBM_AVAILABLE = True
except ImportError:  # noqa: BLE001
    _LGBM_AVAILABLE = False

try:
    from xgboost import XGBClassifier

    _XGB_AVAILABLE = True
except ImportError:  # noqa: BLE001
    _XGB_AVAILABLE = False


from .config import MIN_BALANCED_ACCURACY_GATE
from .settings import PipelinePaths, PipelineSettings

logger = logging.getLogger(__name__)

PPG_SENSORS = ("ppg_ir", "ppg_red", "ppg_green")
ALLOWED_PHASES = ("Relax", "Task")
DEVELOPMENT_PARTICIPANTS = tuple(f"ID-{index:02d}" for index in range(1, 29))
HOLDOUT_PARTICIPANTS = tuple(f"ID-{index:02d}" for index in range(29, 33))
# Accept excellent + good + acceptable so we maximise training samples.
# Poor / invalid remain excluded.
ALLOWED_QUALITY_TIERS = ("excellent", "good", "acceptable")
BALANCED_ACC_THRESHOLD = MIN_BALANCED_ACCURACY_GATE
# Models eligible for ONNX export via skl2onnx + onnxmltools.
# LightGBM is excluded — onnxmltools XGBoost support is sufficient.
ONNX_EXPORTABLE_MODELS = {"logreg", "random_forest", "xgboost"}


class StressModelTrainer:
    def __init__(self, paths: PipelinePaths, settings: PipelineSettings) -> None:
        self.paths = paths
        self.settings = settings

    def run(self, sensor_channel: str = "ppg_ir") -> dict:
        """Train leakage-safe stress models for a single PPG sensor channel.

        Steps:
        1. Load gold features (all participants, all phases).
        2. Filter to sensor_channel rows with allowed quality tiers.
        3. Fixed participant holdout split: ID-01..ID-28 development and
           ID-29..ID-32 holdout.
        4. GroupKFold CV on train set for model selection.
        5. Fit best ONNX-exportable model on full train set.
        6. Evaluate on held-out test participants.
        7. Save artefacts: joblib, selected_features.json, split_and_metrics.json.
        """
        if sensor_channel not in PPG_SENSORS:
            raise ValueError(
                f"Invalid sensor_channel={sensor_channel!r}; expected one of {PPG_SENSORS}"
            )

        df = self._load_gold_features()
        diagnostics_all = self._dataset_diagnostics(df)
        df = self._filter_training_rows(df, sensor_channel)
        if df.empty:
            raise ValueError(
                f"Gold dataset empty after filtering to sensor_channel={sensor_channel}"
            )

        y = df["label"].astype(int).to_numpy()
        groups = df["participant_code"].astype(str).to_numpy()
        feature_cols = self._feature_columns(df, sensor_channel)
        X = self._sanitize_features(df[feature_cols])

        logger.info(
            "Dataset ready — channel=%s rows=%d participants=%d features=%d label_dist=%s",
            sensor_channel,
            len(df),
            len(np.unique(groups)),
            len(feature_cols),
            dict(zip(*np.unique(y, return_counts=True))),
        )

        if len(np.unique(y)) < 2:
            return self._train_single_class_fallback(
                X, y, groups, feature_cols, sensor_channel
            )

        train_idx, test_idx = self._participant_holdout_split(groups, y)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        candidates = self._candidate_models(y_train)
        best_name = ""
        best_score = -1.0
        best_pipe: Pipeline | None = None
        cv_scores: Dict[str, List[float | None]] = {}

        for name, clf in candidates.items():
            pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("clf", clf),
                ]
            )
            try:
                fold_scores = self._cross_val_auc(pipe, X_train, y_train, groups_train)
                cv_scores[name] = fold_scores
                finite_scores = [s for s in fold_scores if s is not None]
                mean_score = float(np.mean(finite_scores)) if finite_scores else -1.0
                logger.info(
                    "  [%s] CV AUC=%.4f (%s)",
                    name,
                    mean_score,
                    [f"{s:.3f}" if s else "None" for s in fold_scores],
                )

                # Always fit on the full training set for final evaluation.
                fitted_pipe = Pipeline(steps=pipe.steps)
                fitted_pipe.fit(X_train, y_train)
                self._save_candidate(
                    sensor_channel, name, fitted_pipe, feature_cols, mean_score
                )

                is_exportable = name in ONNX_EXPORTABLE_MODELS
                if is_exportable and mean_score > best_score:
                    best_score = mean_score
                    best_name = name
                    best_pipe = fitted_pipe
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping model %s due to training error: %s", name, exc)
                cv_scores[name] = []

        if best_pipe is None:
            return self._train_single_class_fallback(
                X, y, groups, feature_cols, sensor_channel
            )

        test_prob = self._predict_proba(best_pipe, X_test)
        test_pred = (test_prob >= 0.5).astype(int)
        metrics = self._binary_metrics(y_test, test_pred, test_prob)
        metrics["below_threshold"] = (
            metrics["balanced_accuracy"] < BALANCED_ACC_THRESHOLD
        )

        logger.info(
            "Best model: %s [%s] — test balanced_acc=%.3f f1=%.3f roc_auc=%.3f (threshold=%.2f → %s)",
            best_name,
            sensor_channel,
            metrics["balanced_accuracy"],
            metrics["f1"],
            metrics["roc_auc"],
            BALANCED_ACC_THRESHOLD,
            "PASS" if not metrics["below_threshold"] else "FAIL",
        )

        if metrics["below_threshold"]:
            logger.warning(
                "Model %s [%s] balanced_accuracy=%.3f < threshold %.2f. Export will be skipped.",
                best_name,
                sensor_channel,
                metrics["balanced_accuracy"],
                BALANCED_ACC_THRESHOLD,
            )

        best_dir = self.paths.models_dir / "stress" / sensor_channel
        best_dir.mkdir(parents=True, exist_ok=True)
        model_path = best_dir / "best_model.joblib"
        feature_path = best_dir / "selected_features.json"
        split_path = best_dir / "split_and_metrics.json"
        joblib.dump(best_pipe, model_path)
        self._write_selected_features(
            feature_path, sensor_channel, feature_cols, self._feature_version(df)
        )

        split_manifest = {
            "task": "stress_classification",
            "train_participants": sorted(set(groups_train.tolist())),
            "test_participants": sorted(set(groups[test_idx].tolist())),
            "random_seed": self.settings.random_seed,
            "model_name": best_name,
            "model_selection_policy": "highest_cv_roc_auc_among_onnx_exportable_candidates",
            "onnx_exportable_candidates": sorted(ONNX_EXPORTABLE_MODELS),
            "sensor_channel": sensor_channel,
            "feature_version": self._feature_version(df),
            "model_window_seconds": self.settings.model_window_seconds,
            "model_window_step_seconds": self.settings.model_window_step_seconds,
            "cv_scores": cv_scores,
            "test_metrics": metrics,
            "per_participant_metrics": self._per_participant_metrics(
                y_test, test_pred, test_prob, groups[test_idx]
            ),
            "dataset_diagnostics": diagnostics_all,
            "sensor_rows": int(len(df)),
            "quality_tiers_used": list(ALLOWED_QUALITY_TIERS),
            "export_gate": {
                "metric": "balanced_accuracy",
                "threshold": BALANCED_ACC_THRESHOLD,
                "passed": not metrics["below_threshold"],
            },
        }
        self._write_json(split_path, split_manifest)
        self._write_model_eval_report(sensor_channel, split_manifest)

        return {
            "model_path": str(model_path),
            "feature_path": str(feature_path),
            "split_path": str(split_path),
            "metrics": metrics,
            "model_name": best_name,
        }

    # ── Data loading ───────────────────────────────────────────────────────────

    def _load_gold_features(self) -> pd.DataFrame:
        path = getattr(self.paths, "gold_features_path", None)
        if path is None:
            path = self.paths.gold_dir / "gold_features.csv"
        if not Path(path).exists():
            raise FileNotFoundError(f"Gold features not found: {path}")
        return pd.read_csv(path)

    def _filter_training_rows(
        self, df: pd.DataFrame, sensor_channel: str
    ) -> pd.DataFrame:
        required = {"label", "participant_code", "phase", "sensor_channel"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Gold features missing required columns: {missing}")

        filtered = df[
            df["phase"].isin(ALLOWED_PHASES)
            & df["sensor_channel"].isin(PPG_SENSORS)
            & (df["sensor_channel"] == sensor_channel)
        ].copy()
        if "quality_tier" in filtered.columns:
            filtered = filtered[
                filtered["quality_tier"].isin(ALLOWED_QUALITY_TIERS)
            ].copy()
        return filtered

    def _dataset_diagnostics(self, df: pd.DataFrame) -> dict:
        participants = sorted(
            df["participant_code"].dropna().astype(str).unique().tolist()
        )
        phases_per_participant = (
            df[df["phase"].isin(ALLOWED_PHASES)]
            .groupby("participant_code")["phase"]
            .nunique()
            .to_dict()
        )
        return {
            "participant_count": len(participants),
            "expected_participant_count": 32,
            "participant_count_mismatch": len(participants) != 32,
            "participants": participants,
            "participants_missing_one_phase": sorted(
                str(pid)
                for pid, n_phase in phases_per_participant.items()
                if n_phase < 2
            ),
        }

    # ── Model candidates ───────────────────────────────────────────────────────

    def _candidate_models(self, y_train: np.ndarray) -> dict:
        """Build candidate classifiers, with RandomizedSearchCV for RF and XGBoost."""
        rng = self.settings.random_seed
        pos = max(1, int(np.sum(y_train == 1)))
        neg = max(1, int(np.sum(y_train == 0)))

        # ── Logistic Regression (interpretable baseline) ──
        logreg = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=rng,
            solver="lbfgs",
            C=1.0,
        )

        # ── Random Forest with RandomizedSearchCV ──
        rf_param_dist = {
            "n_estimators": [200, 300, 500],
            "max_depth": [6, 8, 10, 12, None],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", 0.3, 0.5],
        }
        rf_base = RandomForestClassifier(
            class_weight="balanced",
            random_state=rng,
            n_jobs=-1,
        )
        len(np.unique(y_train))
        n_groups_estimate = max(2, int(len(y_train) * 0.05))  # rough lower bound
        rf_cv_splits = min(3, n_groups_estimate)
        random_forest = RandomizedSearchCV(
            estimator=rf_base,
            param_distributions=rf_param_dist,
            n_iter=20,
            scoring="roc_auc",
            cv=rf_cv_splits,
            random_state=rng,
            n_jobs=-1,
            refit=True,
        )

        candidates: dict = {"logreg": logreg, "random_forest": random_forest}

        # ── LightGBM (non-ONNX-exportable, used for comparison only) ──
        if _LGBM_AVAILABLE:
            candidates["lightgbm"] = LGBMClassifier(
                n_estimators=300,
                num_leaves=31,
                learning_rate=0.05,
                class_weight="balanced",
                random_state=rng,
                n_jobs=-1,
                verbosity=-1,
            )

        # ── XGBoost with GPU + RandomizedSearchCV ──
        if _XGB_AVAILABLE:
            xgb_param_dist = {
                "n_estimators": [200, 300, 500, 700],
                "max_depth": [3, 4, 5, 6, 7],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
                "min_child_weight": [1, 3, 5],
                "reg_alpha": [0, 0.1, 0.5],
                "reg_lambda": [1, 2, 5],
            }
            # Try GPU first; fall back to CPU automatically if CUDA unavailable.
            try:
                import xgboost as xgb_mod

                xgb_mod.XGBClassifier(
                    tree_method="hist", device="cuda", n_estimators=1, verbosity=0
                ).fit(np.zeros((2, 3)), [0, 1])
                xgb_device = "cuda"
            except Exception:
                xgb_device = "cpu"
            logger.info("XGBoost device: %s", xgb_device)

            xgb_base = XGBClassifier(
                tree_method="hist",
                device=xgb_device,
                scale_pos_weight=neg / pos,
                eval_metric="logloss",
                verbosity=0,
                random_state=rng,
                n_jobs=-1 if xgb_device == "cpu" else 1,
            )
            xgb_cv_splits = min(3, n_groups_estimate)
            candidates["xgboost"] = RandomizedSearchCV(
                estimator=xgb_base,
                param_distributions=xgb_param_dist,
                n_iter=30,
                scoring="roc_auc",
                cv=xgb_cv_splits,
                random_state=rng,
                n_jobs=1,  # GPU training is already parallel internally
                refit=True,
            )

        return candidates

    # ── Cross-validation ────────────────────────────────────────────────────────

    def _cross_val_auc(
        self,
        pipe: Pipeline,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        groups_train: np.ndarray,
    ) -> List[float | None]:
        unique_groups = np.unique(groups_train)
        if len(unique_groups) < 2:
            raise ValueError("Need at least two participants for GroupKFold")

        fold_scores: List[float | None] = []
        gkf = GroupKFold(n_splits=min(5, len(unique_groups)))
        for tr, va in gkf.split(X_train, y_train, groups_train):
            if len(np.unique(y_train[tr])) < 2 or len(np.unique(y_train[va])) < 2:
                fold_scores.append(None)
                continue
            pipe.fit(X_train.iloc[tr], y_train[tr])
            prob = self._predict_proba(pipe, X_train.iloc[va])
            fold_scores.append(float(roc_auc_score(y_train[va], prob)))
        return fold_scores

    # ── Save candidates ─────────────────────────────────────────────────────────

    def _save_candidate(
        self,
        sensor_channel: str,
        name: str,
        pipe: Pipeline,
        feature_cols: List[str],
        cv_auc: float,
    ) -> None:
        model_subdir = self.paths.models_dir / "stress" / sensor_channel / name
        model_subdir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, model_subdir / "best_model.joblib")
        self._write_selected_features(
            model_subdir / "selected_features.json",
            sensor_channel,
            feature_cols,
            self.settings.feature_version,
        )
        self._write_json(
            model_subdir / "model_metadata.json",
            {
                "model_name": name,
                "sensor_channel": sensor_channel,
                "onnx_exportable": name in ONNX_EXPORTABLE_MODELS,
                "feature_version": self.settings.feature_version,
                "model_window_seconds": self.settings.model_window_seconds,
                "model_window_step_seconds": self.settings.model_window_step_seconds,
                "cv_auc": cv_auc,
            },
        )

    # ── Feature engineering ─────────────────────────────────────────────────────

    def _sanitize_features(self, X: pd.DataFrame) -> pd.DataFrame:
        Xc = X.copy()
        Xc = Xc.replace([np.inf, -np.inf], np.nan)
        for c in Xc.columns:
            Xc[c] = pd.to_numeric(Xc[c], errors="coerce").clip(lower=-1e9, upper=1e9)
        return Xc

    def _feature_columns(self, df: pd.DataFrame, sensor_channel: str) -> List[str]:
        blacklist = {
            "label",
            "participant_code",
            "participant_id",
            "session_id",
            "recording_id",
            "phase",
            "sensor_channel",
            "signal_source",
            "window_id",
            "window_idx",
            "window_start_ms",
            "window_end_ms",
            "window_start_utc",
            "window_end_utc",
            "quality_score",
            "quality_tier",
            "quality_flag",
            "quality_flags",
            "feature_version",
            "split",
            "source_file",
            "exclusion_reason",
        }
        channel_prefixes = tuple(f"{ch}_" for ch in PPG_SENSORS)
        wanted_prefix = f"{sensor_channel}_"
        cols: List[str] = []
        for c in df.columns:
            lower = c.lower()
            if c in blacklist or "ecg" in lower or lower.startswith("ecg_groundtruth_"):
                continue
            if lower.startswith(channel_prefixes) and not lower.startswith(
                wanted_prefix
            ):
                continue
            if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any():
                cols.append(c)
        if not cols:
            raise ValueError(
                f"No numeric PPG feature columns found for {sensor_channel}"
            )
        return cols

    # ── Participant-level split ─────────────────────────────────────────────────

    def _participant_holdout_split(
        self, groups: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        observed = set(groups.tolist())
        expected = set(DEVELOPMENT_PARTICIPANTS) | set(HOLDOUT_PARTICIPANTS)
        unexpected = sorted(observed - expected)
        missing = sorted(expected - observed)
        if unexpected or missing:
            raise ValueError(
                "Fixed participant contract mismatch: "
                f"unexpected={unexpected}, missing={missing}"
            )

        train_idx = np.where(np.isin(groups, DEVELOPMENT_PARTICIPANTS))[0]
        test_idx = np.where(np.isin(groups, HOLDOUT_PARTICIPANTS))[0]
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            raise ValueError(
                "Fixed participant split requires both labels in development and holdout"
            )
        return train_idx, test_idx

    # ── Metrics ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _predict_proba(model, X: pd.DataFrame) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)
            if prob.ndim == 2 and prob.shape[1] >= 2:
                return prob[:, 1]
            return prob.ravel()
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))

    @staticmethod
    def _binary_metrics(
        y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> dict:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:  # noqa: BLE001
            roc_auc = float("nan")
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = float(auc(recall, precision))
        except Exception:  # noqa: BLE001
            pr_auc = float("nan")

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        return {
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "sensitivity_stress": float(
                recall_score(y_true, y_pred, pos_label=1, zero_division=0)
            ),
            "specificity_relax": float(
                recall_score(y_true, y_pred, pos_label=0, zero_division=0)
            ),
            "confusion_matrix": {
                "labels": ["Relax", "Task"],
                "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            },
        }

    def _per_participant_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        groups: np.ndarray,
    ) -> dict:
        out: dict[str, dict] = {}
        for participant in sorted(set(groups.tolist())):
            mask = groups == participant
            out[str(participant)] = self._binary_metrics(
                y_true[mask], y_pred[mask], y_prob[mask]
            )
        return out

    def _feature_version(self, df: pd.DataFrame) -> str:
        if "feature_version" in df.columns and df["feature_version"].notna().any():
            versions = sorted(
                df["feature_version"].dropna().astype(str).unique().tolist()
            )
            return versions[0] if len(versions) == 1 else ",".join(versions)
        return self.settings.feature_version

    # ── Reporting ───────────────────────────────────────────────────────────────

    def _write_model_eval_report(
        self, sensor_channel: str, split_manifest: dict
    ) -> None:
        report_dir = (
            self.paths.project_root
            / "reports"
            / "model_eval"
            / self.settings.model_version
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "stress_metrics.json"
        existing: dict[str, Any] = {}
        if report_path.exists():
            try:
                existing = json.loads(report_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = {}
        existing[sensor_channel] = split_manifest
        self._write_json(report_path, existing)

    # ── Single-class fallback ────────────────────────────────────────────────────

    def _train_single_class_fallback(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        groups: np.ndarray,
        feature_cols: List[str],
        sensor_channel: str,
    ) -> dict:
        logger.warning(
            "Only one class present in gold dataset for %s. Trained dummy fallback model.",
            sensor_channel,
        )
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", DummyClassifier(strategy="most_frequent")),
            ]
        )
        model.fit(self._sanitize_features(X), y)

        best_dir = self.paths.models_dir / "stress" / sensor_channel
        best_dir.mkdir(parents=True, exist_ok=True)
        model_path = best_dir / "best_model.joblib"
        feature_path = best_dir / "selected_features.json"
        split_path = best_dir / "split_and_metrics.json"
        joblib.dump(model, model_path)
        self._write_selected_features(
            feature_path, sensor_channel, feature_cols, self.settings.feature_version
        )

        metrics = {
            "f1": 0.0,
            "roc_auc": None,
            "pr_auc": None,
            "accuracy": 0.0,
            "balanced_accuracy": 0.5,
            "below_threshold": True,
        }
        self._write_json(
            split_path,
            {
                "train_participants": sorted(set(groups.tolist())),
                "test_participants": [],
                "random_seed": self.settings.random_seed,
                "model_name": "dummy_single_class",
                "sensor_channel": sensor_channel,
                "feature_version": self.settings.feature_version,
                "model_window_seconds": self.settings.model_window_seconds,
                "model_window_step_seconds": self.settings.model_window_step_seconds,
                "cv_scores": {},
                "test_metrics": metrics,
            },
        )
        return {
            "model_path": str(model_path),
            "feature_path": str(feature_path),
            "split_path": str(split_path),
            "metrics": metrics,
            "model_name": "dummy_single_class",
        }

    # ── Utilities ────────────────────────────────────────────────────────────────

    @classmethod
    def _write_json(cls, path: Path, payload: Any) -> None:
        path.write_text(json.dumps(cls._json_safe(payload), indent=2), encoding="utf-8")

    def _write_selected_features(
        self,
        path: Path,
        sensor_channel: str,
        feature_cols: List[str],
        feature_version: str,
    ) -> None:
        self._write_json(
            path,
            {
                "selected_features": feature_cols,
                "sensor_channel": sensor_channel,
                "feature_version": feature_version,
                "model_window_seconds": self.settings.model_window_seconds,
                "model_window_step_seconds": self.settings.model_window_step_seconds,
            },
        )

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): cls._json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._json_safe(v) for v in value]
        if isinstance(value, tuple):
            return [cls._json_safe(v) for v in value]
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            value = float(value)
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value


if __name__ == "__main__":
    from PipelinePython.common.cli import main

    main(["train"])
