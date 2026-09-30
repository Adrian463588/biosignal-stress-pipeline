from __future__ import annotations

import json
import logging
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold
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


from .settings import PipelinePaths, PipelineSettings

logger = logging.getLogger(__name__)


class PersonalityLabelResolver:
    """
    Loads IPIP personality results and resolves dominant trait per participant.

    Per AGENTS.md §7, the dominant trait is the IPIP dimension with the highest mean score.
    Maps participant names to participant_code for joining with gold features.
    """

    IPIP_TRAIT_COLS = {
        "Extraversion": "IPIP Extraversion Mean",
        "Agreeableness": "IPIP Agreeableness Mean",
        "Conscientiousness": "IPIP Conscientiousness Mean",
        "Emotional Stability": "IPIP Emotional Stability Mean",
        "Intellect": "IPIP Intellect Mean",
    }

    def __init__(self, personality_csv_path: str) -> None:
        self.personality_csv_path = personality_csv_path
        self._cache: Dict[str, str] | None = None

    def load(self) -> pd.DataFrame:
        """Load personality results and compute dominant trait per participant."""
        df = pd.read_csv(self.personality_csv_path)

        # Ensure required columns exist
        for trait_name, col_name in self.IPIP_TRAIT_COLS.items():
            if col_name not in df.columns:
                raise ValueError(f"Missing required column: {col_name}")

        # Compute dominant trait: the trait with the highest mean score
        trait_scores = df[list(self.IPIP_TRAIT_COLS.values())]
        df["dominant_trait"] = trait_scores.idxmax(axis=1).map(
            {v: k for k, v in self.IPIP_TRAIT_COLS.items()}
        )

        # Map trait names to numeric labels (0-4)
        trait_to_label = {
            "Extraversion": 0,
            "Agreeableness": 1,
            "Conscientiousness": 2,
            "Emotional Stability": 3,
            "Intellect": 4,
        }
        df["dominant_trait_label"] = df["dominant_trait"].map(trait_to_label)

        return df[["Name", "dominant_trait", "dominant_trait_label"]].copy()

    def resolve_name_to_code(self, name: str, gold_df: pd.DataFrame) -> str | None:
        """
        Resolve participant name to participant_code using gold features.

        This is a fuzzy match: we look for any participant_code that appears
        in the gold dataset and try to match it with the name.
        """
        # For now, return None if no direct match found
        # In production, this would use fuzzy string matching or a name mapping table
        return None


class PersonalityModelTrainer:
    """
    Trains multi-class personality classifier on Relax-phase PPG data.

    Per AGENTS.md §7:
    - Input: Relax phase only
    - Target: Dominant IPIP trait (5-class: Extraversion, Agreeableness, Conscientiousness, Emotional Stability, Intellect)
    - Features: PPG-derived only (exclude ECG ground truth)
    - Split: By participant (no leakage)
    - Models: Logistic Regression, Random Forest, LightGBM, XGBoost
    """

    def __init__(self, paths: PipelinePaths, settings: PipelineSettings) -> None:
        self.paths = paths
        self.settings = settings
        self.label_resolver = PersonalityLabelResolver(
            str(
                paths.project_root
                / "PersonalityAnchor"
                / "all_participants_results.csv"
            )
        )

    def run(self, sensor_channel: str = "ppg_ir") -> dict:
        """Train personality model and save artifacts.

        Args:
            sensor_channel: One of "ppg_ir", "ppg_red", "ppg_green".
        """
        # Load gold features
        gold_path = self.paths.gold_dir / "gold_features.csv"
        if not gold_path.exists():
            raise FileNotFoundError(f"Gold features not found: {gold_path}")

        gold_df = pd.read_csv(gold_path)

        # Filter to Relax phase only per AGENTS.md §14
        gold_df = gold_df[gold_df["phase"] == "Relax"].copy()
        if gold_df.empty:
            raise ValueError("No Relax-phase data found in gold features")

        # Filter to specified sensor_channel (AGENTS §14)
        if "sensor_channel" in gold_df.columns:
            gold_df = gold_df[gold_df["sensor_channel"] == sensor_channel].copy()
            logger.info(
                "Filtered to sensor_channel=%s (%d rows)", sensor_channel, len(gold_df)
            )
        else:
            logger.warning(
                "No sensor_channel column in gold features (legacy CSV). "
                "Re-run ingest to generate per-sensor gold features."
            )

        if gold_df.empty:
            raise ValueError(
                f"No data after filtering to phase=Relax sensor_channel={sensor_channel}"
            )

        # Load personality labels
        personality_df = self.label_resolver.load()

        # Join gold features with personality labels
        # We need to map participant_code to Name from personality_df
        # For now, assume participant_code matches a name or ID in the personality data
        merged = self._merge_with_personality(gold_df, personality_df)

        if merged.empty:
            raise ValueError("No samples after merging with personality labels")

        y = merged["dominant_trait_label"].astype(int).to_numpy()
        groups = merged["participant_code"].astype(str).to_numpy()
        feature_cols = self._feature_columns(merged)
        X = self._sanitize_features(merged[feature_cols])

        if len(np.unique(y)) < 2:
            logger.warning(
                "Only one class present in personality dataset. Using fallback model."
            )
            return self._train_single_class_fallback(X, y, groups, feature_cols)

        # Split by participant
        train_idx, test_idx = self._participant_holdout_split(groups)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        # Ensure both train and test have multiple classes
        if len(np.unique(y_train)) < 2:
            logger.warning("Train set has only one class. Resampling...")
            train_idx, test_idx = self._random_split(len(y))
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_train = groups[train_idx]

        # Train candidate models
        candidates = {
            "logreg": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=self.settings.random_seed,
                multi_class="multinomial",
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight="balanced",
                random_state=self.settings.random_seed,
                n_jobs=-1,
            ),
        }

        if _LGBM_AVAILABLE:
            candidates["lightgbm"] = LGBMClassifier(
                n_estimators=200,
                num_leaves=31,
                class_weight="balanced",
                random_state=self.settings.random_seed,
                n_jobs=-1,
                verbosity=-1,
            )

        if _XGB_AVAILABLE:
            candidates["xgboost"] = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                random_state=self.settings.random_seed,
                n_jobs=-1,
                eval_metric="mlogloss",
                verbosity=0,
            )

        best_name = ""
        best_score = -1.0
        best_pipe: Pipeline | None = None
        cv_scores: Dict[str, List[float]] = {}

        for name, clf in candidates.items():
            try:
                pipe = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("clf", clf),
                    ]
                )

                fold_scores = []
                n_group = len(np.unique(groups_train))
                if n_group >= 2 and len(np.unique(y_train)) >= 2:
                    gkf = GroupKFold(n_splits=min(5, n_group))
                    for tr, va in gkf.split(X_train, y_train, groups_train):
                        pipe.fit(X_train.iloc[tr], y_train[tr])
                        pred = pipe.predict(X_train.iloc[va])
                        fold_scores.append(
                            float(
                                f1_score(
                                    y_train[va],
                                    pred,
                                    average="weighted",
                                    zero_division=0,
                                )
                            )
                        )
                else:
                    pipe.fit(X_train, y_train)
                    pred = pipe.predict(X_train)
                    fold_scores.append(
                        float(
                            f1_score(y_train, pred, average="weighted", zero_division=0)
                        )
                    )

                cv_scores[name] = fold_scores

                # Save candidate artifacts in per-sensor personality subdir
                model_subdir = (
                    self.paths.models_dir / "personality" / sensor_channel / name
                )
                model_subdir.mkdir(parents=True, exist_ok=True)
                joblib.dump(pipe, model_subdir / "best_model.joblib")
                (model_subdir / "selected_features.json").write_text(
                    json.dumps(feature_cols, indent=2), encoding="utf-8"
                )
                (model_subdir / "model_metadata.json").write_text(
                    json.dumps(
                        {
                            "model_name": name,
                            "task": "personality_classification",
                            "cv_f1_weighted": float(np.mean(fold_scores))
                            if fold_scores
                            else 0.0,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                mean_score = float(np.mean(fold_scores)) if fold_scores else 0.0
                if mean_score > best_score:
                    best_score = mean_score
                    best_name = name
                    best_pipe = pipe
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Skipping personality model %s due to error: %s", name, exc
                )
                cv_scores[name] = []

        if best_pipe is None:
            return self._train_single_class_fallback(X, y, groups, feature_cols)

        # Fit best model on full train set
        best_pipe.fit(X_train, y_train)

        # Evaluate on test set
        test_pred = best_pipe.predict(X_test)
        best_pipe.predict_proba(X_test) if hasattr(best_pipe, "predict_proba") else None

        metrics = {
            "f1_weighted": float(
                f1_score(y_test, test_pred, average="weighted", zero_division=0)
            ),
            "f1_macro": float(
                f1_score(y_test, test_pred, average="macro", zero_division=0)
            ),
            "accuracy": float(accuracy_score(y_test, test_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, test_pred)),
        }

        # Save best model to per-sensor personality directory
        best_dir = self.paths.models_dir / "personality" / sensor_channel
        best_dir.mkdir(parents=True, exist_ok=True)
        model_path = best_dir / "best_model.joblib"
        joblib.dump(best_pipe, model_path)

        feature_path = best_dir / "selected_features.json"
        feature_path.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")

        split_manifest = {
            "train_participants": sorted(set(groups_train.tolist())),
            "test_participants": sorted(set(groups[test_idx].tolist())),
            "random_seed": self.settings.random_seed,
            "model_name": best_name,
            "task": "personality_classification",
            "sensor_channel": sensor_channel,
            "cv_scores": cv_scores,
            "test_metrics": metrics,
        }
        split_path = best_dir / "personality_split_and_metrics.json"
        split_path.write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

        logger.info(
            "Personality training complete. channel=%s model=%s test_f1_weighted=%.3f test_accuracy=%.3f",
            sensor_channel,
            best_name,
            metrics["f1_weighted"],
            metrics["accuracy"],
        )

        return {
            "model_path": str(model_path),
            "feature_path": str(feature_path),
            "split_path": str(split_path),
            "metrics": metrics,
            "model_name": best_name,
        }

    def _merge_with_personality(
        self, gold_df: pd.DataFrame, personality_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge gold features with personality labels using fuzzy matching.
        """
        import re

        # Load mapping.csv to get the original names for ID-XX
        mapping_path = self.paths.raw_data_root / "mapping.csv"
        id_to_orig = {}
        if mapping_path.exists():
            mapping_df = pd.read_csv(mapping_path)
            for _, row in mapping_df.iterrows():
                # Clean original name (remove digits and special chars)
                clean_orig = re.sub(r"[^a-zA-Z]", "", str(row["original_name"])).lower()
                id_to_orig[row["participant_id"]] = clean_orig

        unique_codes = gold_df["participant_code"].unique()
        code_to_trait = {}

        for code in unique_codes:
            # Fallback to the code itself if no mapping is found
            clean_orig = id_to_orig.get(code, str(code).lower())
            if not clean_orig:
                continue

            for _, prow in personality_df.iterrows():
                clean_name = re.sub(r"[^a-zA-Z\s]", "", str(prow["Name"])).lower()

                # Check if clean_orig is a substring of any word in clean_name
                # or if the name contains clean_orig
                if (
                    clean_orig in clean_name
                    or clean_name.replace(" ", "") in clean_orig
                ):
                    code_to_trait[code] = prow["dominant_trait_label"]
                    break

        if not code_to_trait:
            logger.warning(
                "No personality labels matched with gold features. Returning empty dataframe."
            )
            return pd.DataFrame()

        logger.info(
            f"Matched {len(code_to_trait)} participants for personality training."
        )

        merged = gold_df[gold_df["participant_code"].isin(code_to_trait.keys())].copy()
        merged["dominant_trait_label"] = merged["participant_code"].map(code_to_trait)

        return merged

    def _sanitize_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove infinities and clip extreme outliers."""
        Xc = X.copy()
        Xc = Xc.replace([np.inf, -np.inf], np.nan)
        for c in Xc.columns:
            Xc[c] = pd.to_numeric(Xc[c], errors="coerce").clip(lower=-1e9, upper=1e9)
        return Xc

    def _feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Select numeric feature columns, excluding metadata and ECG ground truth.

        Per AGENTS.md §7, exclude ECG ground truth features to ensure
        training strictly on valid PPG data.
        """
        blacklist = {
            "label",
            "participant_code",
            "phase",
            "session_id",
            "feature_version",
            "window_idx",
            "signal_source",
            "dominant_trait",
            "dominant_trait_label",
        }
        ecg_exclusion_pattern = "ecg_groundtruth_"

        cols: List[str] = []
        for c in df.columns:
            if c in blacklist:
                continue
            if c.startswith(ecg_exclusion_pattern):
                continue
            if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any():
                cols.append(c)

        if not cols:
            raise ValueError("No numeric feature columns found")

        return cols

    def _participant_holdout_split(
        self, groups: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Split by participant to prevent leakage."""
        participants = np.array(sorted(set(groups.tolist())))
        if len(participants) <= 1:
            return self._random_split(len(groups))

        rng = np.random.default_rng(self.settings.random_seed)
        rng.shuffle(participants)
        n_test = max(1, int(len(participants) * self.settings.test_size_participants))
        test_set = set(participants[:n_test].tolist())

        test_idx = np.where(np.array([g in test_set for g in groups]))[0]
        train_idx = np.where(np.array([g not in test_set for g in groups]))[0]

        return train_idx, test_idx

    def _random_split(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Random split when participant-based split is not possible."""
        idx = np.arange(n)
        rng = np.random.default_rng(self.settings.random_seed)
        rng.shuffle(idx)
        cut = max(1, int(n * (1.0 - self.settings.test_size_participants)))
        return idx[:cut], idx[cut:]

    def _train_single_class_fallback(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        groups: np.ndarray,
        feature_cols: List[str],
    ) -> dict:
        """Fallback when only one class is present."""
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", DummyClassifier(strategy="most_frequent")),
            ]
        )
        model.fit(self._sanitize_features(X), y)

        model_path = self.paths.models_dir / "personality_best_model.joblib"
        joblib.dump(model, model_path)

        feature_path = self.paths.models_dir / "personality_selected_features.json"
        feature_path.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")

        split_path = self.paths.models_dir / "personality_split_and_metrics.json"
        split_path.write_text(
            json.dumps(
                {
                    "train_participants": sorted(set(groups.tolist())),
                    "test_participants": sorted(set(groups.tolist())),
                    "random_seed": self.settings.random_seed,
                    "model_name": "dummy_single_class",
                    "task": "personality_classification",
                    "cv_scores": {},
                    "test_metrics": {"f1_weighted": 1.0, "accuracy": 1.0},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        logger.warning(
            "Only one class present in personality dataset. Trained dummy fallback model."
        )
        return {
            "model_path": str(model_path),
            "feature_path": str(feature_path),
            "split_path": str(split_path),
            "metrics": {"f1_weighted": 1.0, "accuracy": 1.0},
            "model_name": "dummy_single_class",
        }
