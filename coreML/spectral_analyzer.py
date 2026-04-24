"""Spectral analyzer for MFCC extraction, training, and inference."""

import json
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from time import perf_counter
from typing import Any, Iterable

import joblib
import librosa
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from config import SETTINGS

from .constants import (
    MFCC_N_COEFFICIENTS,
    SCORE_MAX,
    SCORE_MIN,
    SPECTRAL_MIN_F1,
    SPECTRAL_TIMEOUT_SECONDS,
)
from .errors import AudioProcessingError
from .types import SpectralResult


GENERAL_CONFIG = SETTINGS.general
CORE_CONFIG = SETTINGS.core
PATHS = SETTINGS.paths


class SpectralAnalyzer:
    """Core class for spectral feature extraction, training, and analysis."""

    def __init__(
        self,
        n_mfcc: int = MFCC_N_COEFFICIENTS,
        timeout_seconds: int = SPECTRAL_TIMEOUT_SECONDS,
        model_path: str = str(PATHS.spectral_model_path),
        training_report_path: str = str(PATHS.spectral_training_report_path),
        n_clusters: int = CORE_CONFIG.spectral_cluster_count,
        random_state: int = GENERAL_CONFIG.random_seed,
    ) -> None:
        self.n_mfcc = n_mfcc
        self.timeout_seconds = timeout_seconds
        self.model_path = model_path
        self.training_report_path = training_report_path
        self.n_clusters = n_clusters
        self.random_state = random_state

        self.classifier: RandomForestClassifier | GradientBoostingClassifier | None = None
        self.kmeans_model: KMeans | None = None
        self.cluster_distance_thresholds: dict[int, float] = {}
        self._synthetic_label = CORE_CONFIG.spectral_positive_label

        self._load_artifacts_if_available()

    def _load_artifacts_if_available(self) -> None:
        if not os.path.isfile(self.model_path):
            return
        try:
            artifact = joblib.load(self.model_path)
        except Exception as exc:
            raise AudioProcessingError(
                error_code="SPECTRAL_MODEL_LOAD_FAILED",
                description=f"Unable to load spectral model artifact from {self.model_path}.",
            ) from exc

        if isinstance(artifact, dict):
            self.classifier = artifact.get("classifier")
            self.kmeans_model = artifact.get("kmeans")
            self.cluster_distance_thresholds = artifact.get("cluster_distance_thresholds", {})
            self._synthetic_label = artifact.get(
                "synthetic_label",
                CORE_CONFIG.spectral_positive_label,
            )
        else:
            self.classifier = artifact

    @staticmethod
    def _mean_pool(features: np.ndarray) -> np.ndarray:
        return features.mean(axis=1).astype(np.float32, copy=False)

    @staticmethod
    def _extract_sample_path(sample: dict[str, Any]) -> str:
        for key in ("audio_file_path", "path", "file_path"):
            value = sample.get(key)
            if isinstance(value, str) and value:
                return value
        raise AudioProcessingError(
            error_code="TRAINING_SAMPLE_PATH_MISSING",
            description="Training sample must include audio_file_path/path/file_path.",
        )

    @staticmethod
    def _extract_sample_label(sample: dict[str, Any]) -> str:
        value = sample.get("label")
        if not isinstance(value, str) or not value:
            raise AudioProcessingError(
                error_code="TRAINING_SAMPLE_LABEL_MISSING",
                description="Training sample must include a non-empty string label.",
            )
        return value

    def extract_features(self, audio_file_path: str) -> np.ndarray:
        """Return MFCC feature matrix of shape (40, time_frames), dtype float32."""
        if not audio_file_path:
            raise AudioProcessingError(
                error_code="AUDIO_PATH_MISSING",
                description="Audio file path must be provided.",
            )

        if not os.path.isfile(audio_file_path):
            raise AudioProcessingError(
                error_code="AUDIO_FILE_NOT_FOUND",
                description=f"Audio file not found: {audio_file_path}",
            )

        try:
            audio_signal, sample_rate = librosa.load(audio_file_path, sr=None, mono=True)
        except Exception as exc:
            raise AudioProcessingError(
                error_code="AUDIO_LOAD_FAILED",
                description="Audio is corrupt or format is unsupported.",
            ) from exc

        if audio_signal.size == 0:
            raise AudioProcessingError(
                error_code="AUDIO_EMPTY",
                description="Audio file contains no samples.",
            )

        try:
            mfcc_features = librosa.feature.mfcc(
                y=audio_signal,
                sr=sample_rate,
                n_mfcc=self.n_mfcc,
            )
        except Exception as exc:
            raise AudioProcessingError(
                error_code="MFCC_EXTRACTION_FAILED",
                description="Failed to extract MFCC features from audio.",
            ) from exc

        if mfcc_features.ndim != 2 or 0 in mfcc_features.shape:
            raise AudioProcessingError(
                error_code="MFCC_INVALID_SHAPE",
                description="Extracted MFCC features are empty or malformed.",
            )

        return mfcc_features.astype(np.float32, copy=False)

    def train(
        self,
        labeled_samples: Iterable[dict[str, Any]],
        model_type: str = CORE_CONFIG.spectral_model_type,
        test_size: float = CORE_CONFIG.spectral_test_split_ratio,
    ) -> float:
        samples = list(labeled_samples)
        if len(samples) < 2:
            raise AudioProcessingError(
                error_code="TRAINING_DATA_INSUFFICIENT",
                description="At least 2 labeled samples are required to train spectral model.",
            )

        vectors: list[np.ndarray] = []
        labels: list[str] = []

        for sample in samples:
            path = self._extract_sample_path(sample)
            label = self._extract_sample_label(sample)
            features = self.extract_features(path)
            vectors.append(self._mean_pool(features))
            labels.append(label)

        x = np.vstack(vectors)
        y = np.array(labels)

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y if len(set(y.tolist())) > 1 else None,
        )

        if model_type == "gradient_boosting":
            self.classifier = GradientBoostingClassifier(random_state=self.random_state)
        else:
            self.classifier = RandomForestClassifier(
                n_estimators=CORE_CONFIG.spectral_random_forest_n_estimators,
                random_state=self.random_state,
                n_jobs=CORE_CONFIG.spectral_random_forest_n_jobs,
            )

        self.classifier.fit(x_train, y_train)
        predictions = self.classifier.predict(x_test)
        model_f1 = float(f1_score(y_test, predictions, pos_label=self._synthetic_label))

        if model_f1 < SPECTRAL_MIN_F1:
            raise AudioProcessingError(
                error_code="SPECTRAL_MIN_F1_NOT_MET",
                description=(
                    f"Spectral training F1 ({model_f1:.3f}) is below required "
                    f"minimum ({SPECTRAL_MIN_F1:.2f})."
                ),
            )

        self.kmeans_model = KMeans(
            n_clusters=min(self.n_clusters, max(1, len(x_train))),
            random_state=self.random_state,
            n_init=CORE_CONFIG.spectral_kmeans_n_init,
        )
        self.kmeans_model.fit(x_train)

        train_cluster_ids = self.kmeans_model.predict(x_train)
        self.cluster_distance_thresholds = {}
        for cluster_id in np.unique(train_cluster_ids):
            cluster_mask = train_cluster_ids == cluster_id
            cluster_vectors = x_train[cluster_mask]
            center = self.kmeans_model.cluster_centers_[cluster_id]
            distances = np.linalg.norm(cluster_vectors - center, axis=1)
            self.cluster_distance_thresholds[int(cluster_id)] = float(
                distances.mean()
                + CORE_CONFIG.spectral_anomaly_std_multiplier * distances.std(ddof=0)
            )

        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        artifact = {
            "classifier": self.classifier,
            "kmeans": self.kmeans_model,
            "cluster_distance_thresholds": self.cluster_distance_thresholds,
            "synthetic_label": self._synthetic_label,
        }
        joblib.dump(artifact, self.model_path)

        report_payload = {
            "kmeans": {
                "n_clusters": int(getattr(self.kmeans_model, "n_clusters", self.n_clusters)),
                "assignment_method": "KMeans (Euclidean distance)",
            },
            "spectral_training": {
                "model_type": model_type,
                "f1_score": model_f1,
                "sample_count": len(samples),
            },
        }
        os.makedirs(os.path.dirname(self.training_report_path) or ".", exist_ok=True)
        with open(self.training_report_path, "w", encoding="utf-8") as report_file:
            json.dump(report_payload, report_file, indent=2)

        return model_f1

    def _predict_synthetic_probability(self, feature_vector: np.ndarray) -> float:
        if self.classifier is None:
            raise AudioProcessingError(
                error_code="SPECTRAL_MODEL_NOT_TRAINED",
                description="Spectral model is not trained or loaded yet.",
            )

        probabilities = self.classifier.predict_proba(feature_vector.reshape(1, -1))[0]
        classes = getattr(self.classifier, "classes_", np.array(["real", self._synthetic_label]))
        classes_list = [str(item) for item in classes.tolist()]
        if self._synthetic_label in classes_list:
            synthetic_index = classes_list.index(self._synthetic_label)
        else:
            synthetic_index = int(np.argmax(probabilities))
        return float(probabilities[synthetic_index])

    def _compute_anomaly_flag(self, feature_vector: np.ndarray) -> bool:
        if self.kmeans_model is None:
            return False

        cluster_id = int(self.kmeans_model.predict(feature_vector.reshape(1, -1))[0])
        center = self.kmeans_model.cluster_centers_[cluster_id]
        distance = float(np.linalg.norm(feature_vector - center))
        threshold = self.cluster_distance_thresholds.get(cluster_id)
        if threshold is None:
            return True
        return distance > threshold

    def _analyze_sync(self, audio_file_path: str) -> SpectralResult:
        start_time = perf_counter()
        features = self.extract_features(audio_file_path)
        pooled = self._mean_pool(features)
        synthetic_prob = self._predict_synthetic_probability(pooled)
        spectral_score = float(np.clip(synthetic_prob * 100.0, SCORE_MIN, SCORE_MAX))
        anomaly_flag = self._compute_anomaly_flag(pooled)
        processing_time_ms = int((perf_counter() - start_time) * 1000)
        return SpectralResult(
            spectral_score=spectral_score,
            anomaly_flag=anomaly_flag,
            processing_time_ms=processing_time_ms,
        )

    def analyze(self, audio_file_path: str) -> SpectralResult:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._analyze_sync, audio_file_path)
            try:
                return future.result(timeout=self.timeout_seconds)
            except TimeoutError as exc:
                future.cancel()
                raise AudioProcessingError(
                    error_code="SPECTRAL_TIMEOUT",
                    description=(
                        f"Spectral analysis exceeded timeout of {self.timeout_seconds} seconds."
                    ),
                ) from exc

    def serialize_features(self, features: np.ndarray, output_path: str) -> None:
        """Persist MFCC features to disk as .npy."""
        if not output_path:
            raise AudioProcessingError(
                error_code="FEATURE_OUTPUT_PATH_MISSING",
                description="Output path for features must be provided.",
            )

        if not isinstance(features, np.ndarray):
            raise AudioProcessingError(
                error_code="FEATURES_INVALID_TYPE",
                description="Features must be a NumPy ndarray.",
            )

        if features.ndim != 2 or 0 in features.shape:
            raise AudioProcessingError(
                error_code="FEATURES_INVALID_SHAPE",
                description="Features must be a non-empty 2D array.",
            )

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        try:
            np.save(output_path, features.astype(np.float32, copy=False))
        except Exception as exc:
            raise AudioProcessingError(
                error_code="FEATURE_SERIALIZATION_FAILED",
                description="Failed to serialize MFCC features to .npy.",
            ) from exc

    def deserialize_features(self, input_path: str) -> np.ndarray:
        """Load MFCC features from .npy with original shape and dtype."""
        if not input_path:
            raise AudioProcessingError(
                error_code="FEATURE_INPUT_PATH_MISSING",
                description="Input path for features must be provided.",
            )

        if not os.path.isfile(input_path):
            raise AudioProcessingError(
                error_code="FEATURE_FILE_NOT_FOUND",
                description=f"Feature file not found: {input_path}",
            )

        try:
            features = np.load(input_path, allow_pickle=False)
        except Exception as exc:
            raise AudioProcessingError(
                error_code="FEATURE_DESERIALIZATION_FAILED",
                description="Failed to deserialize MFCC features from .npy.",
            ) from exc

        if not isinstance(features, np.ndarray):
            raise AudioProcessingError(
                error_code="FEATURES_INVALID_TYPE",
                description="Deserialized features are not a NumPy ndarray.",
            )

        if features.ndim != 2 or 0 in features.shape:
            raise AudioProcessingError(
                error_code="FEATURES_INVALID_SHAPE",
                description="Deserialized features are empty or malformed.",
            )

        return features.astype(np.float32, copy=False)
