"""Intent analyzer contracts for transcription and urgency scoring."""

import csv
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from time import perf_counter
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from config import SETTINGS

from .constants import INTENT_MIN_F1, INTENT_TIMEOUT_SECONDS, SCORE_MAX, SCORE_MIN
from .errors import AudioProcessingError
from .types import IntentResult


GENERAL_CONFIG = SETTINGS.general
CORE_CONFIG = SETTINGS.core
PATHS = SETTINGS.paths


class IntentAnalyzer:
    """Core class for intent transcription and scoring workflow."""

    _KEYWORD_WEIGHTS = dict(CORE_CONFIG.intent_keyword_weights)
    _TIME_PRESSURE_PHRASES = CORE_CONFIG.intent_time_pressure_phrases

    def __init__(
        self,
        timeout_seconds: int = INTENT_TIMEOUT_SECONDS,
        keywords_csv_path: str = str(PATHS.intent_keywords_csv_path),
        model_path: str = str(PATHS.intent_model_path),
        training_report_path: str = str(PATHS.intent_training_report_path),
        random_state: int = GENERAL_CONFIG.random_seed,
        n_clusters: int = CORE_CONFIG.intent_cluster_count,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.keywords_csv_path = keywords_csv_path
        self.model_path = model_path
        self.random_state = random_state
        self.n_clusters = n_clusters
        self.training_report_path = training_report_path

        self._whisper_model = self._load_whisper_model()
        self._nltk = None
        self._nlp = self._safe_load_spacy_model()
        self._ensure_nltk_tokenizer()
        self.keyword_weights = self._load_keyword_weights()

        self.vectorizer: TfidfVectorizer | None = None
        self.classifier: RandomForestClassifier | None = None
        self.kmeans_model: KMeans | None = None
        self.cluster_distance_thresholds: dict[int, float] = {}
        self._coercion_label = CORE_CONFIG.intent_positive_label

        self._load_artifacts_if_available()

    @staticmethod
    def _load_whisper_model() -> Any:
        try:
            import whisper

            return whisper.load_model(CORE_CONFIG.intent_whisper_model_name)
        except Exception as exc:
            raise AudioProcessingError(
                error_code="WHISPER_INIT_FAILED",
                description=(
                    f"Failed to initialize Whisper {CORE_CONFIG.intent_whisper_model_name} model."
                ),
            ) from exc

    def _safe_load_spacy_model(self) -> Any:
        try:
            import spacy

            return spacy.load("en_core_web_sm")
        except Exception:
            try:
                import spacy

                return spacy.blank("en")
            except Exception:
                return None

    def _ensure_nltk_tokenizer(self) -> None:
        try:
            import nltk

            self._nltk = nltk
            nltk.data.find("tokenizers/punkt")
        except Exception:
            try:
                import nltk

                self._nltk = nltk
                nltk.download("punkt", quiet=True)
            except Exception:
                self._nltk = None

    def _load_keyword_weights(self) -> dict[str, float]:
        if not os.path.isfile(self.keywords_csv_path):
            return dict(self._KEYWORD_WEIGHTS)

        loaded_weights: dict[str, float] = {}
        try:
            with open(self.keywords_csv_path, "r", encoding="utf-8") as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    keyword = str(row.get("keyword", "")).strip().lower()
                    if not keyword:
                        continue
                    weight_raw = row.get("weight", "0")
                    loaded_weights[keyword] = float(weight_raw)
        except Exception as exc:
            raise AudioProcessingError(
                error_code="KEYWORD_CSV_LOAD_FAILED",
                description=f"Unable to parse keyword CSV at {self.keywords_csv_path}.",
            ) from exc

        return loaded_weights or dict(self._KEYWORD_WEIGHTS)

    def _load_artifacts_if_available(self) -> None:
        if not os.path.isfile(self.model_path):
            return
        try:
            artifact = joblib.load(self.model_path)
        except Exception as exc:
            raise AudioProcessingError(
                error_code="INTENT_MODEL_LOAD_FAILED",
                description=f"Unable to load intent model artifact from {self.model_path}.",
            ) from exc

        if not isinstance(artifact, dict):
            return

        self.vectorizer = artifact.get("vectorizer")
        self.classifier = artifact.get("classifier")
        self.kmeans_model = artifact.get("kmeans")
        self.cluster_distance_thresholds = artifact.get("cluster_distance_thresholds", {})
        self._coercion_label = artifact.get("coercion_label", CORE_CONFIG.intent_positive_label)

    def _validate_audio_file(self, audio_file_path: str) -> None:
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

    def _transcribe_audio(self, audio_file_path: str) -> str:
        try:
            transcription = self._whisper_model.transcribe(audio_file_path)
        except Exception as exc:
            raise AudioProcessingError(
                error_code="TRANSCRIPTION_FAILED",
                description="Failed to transcribe audio with Whisper.",
            ) from exc
        return str(transcription.get("text", "")).strip()

    def _keyword_score(self, text: str) -> float:
        score = 0.0
        for keyword, weight in self.keyword_weights.items():
            if keyword in text:
                score += weight
        return score

    def _urgency_score(self, text: str) -> float:
        score = 0.0
        for phrase, bonus in self._TIME_PRESSURE_PHRASES:
            if phrase in text:
                score += bonus

        try:
            if self._nltk is None:
                raise RuntimeError("nltk unavailable")
            tokens = self._nltk.word_tokenize(text)
            if any(token in {"now", "urgent", "immediately", "quickly"} for token in tokens):
                score += CORE_CONFIG.intent_nltk_urgency_bonus
        except Exception:
            tokens = text.split()
            if any(token in {"now", "urgent", "immediately", "quickly"} for token in tokens):
                score += CORE_CONFIG.intent_fallback_urgency_bonus

        try:
            if self._nlp is None:
                raise RuntimeError("spacy unavailable")
            doc = self._nlp(text)
            imperative_bonus = sum(
                CORE_CONFIG.intent_imperative_verb_bonus
                for token in doc
                if token.pos_ == "VERB" and token.tag_ == "VB"
            )
            score += imperative_bonus
        except Exception:
            pass

        return score

    @staticmethod
    def _normalize_score(
        raw_score: float,
        min_raw: float = CORE_CONFIG.intent_normalize_min_raw,
        max_raw: float = CORE_CONFIG.intent_normalize_max_raw,
    ) -> float:
        clipped_raw = max(min_raw, min(raw_score, max_raw))
        scaled = (clipped_raw - min_raw) / (max_raw - min_raw)
        return float(np.clip(scaled * 100.0, SCORE_MIN, SCORE_MAX))

    def _score_transcript(self, transcript: str) -> float:
        text = transcript.lower().strip()
        if not text:
            return 0.0

        raw_score = self._keyword_score(text) + self._urgency_score(text)
        return self._normalize_score(raw_score)

    @staticmethod
    def _extract_transcript(sample: dict[str, Any]) -> str:
        text = sample.get("transcript")
        if not isinstance(text, str) or not text.strip():
            raise AudioProcessingError(
                error_code="TRAINING_TRANSCRIPT_MISSING",
                description="Training sample must include a non-empty transcript.",
            )
        return text.strip()

    @staticmethod
    def _extract_label(sample: dict[str, Any]) -> str:
        label = sample.get("label")
        if not isinstance(label, str) or not label:
            raise AudioProcessingError(
                error_code="TRAINING_LABEL_MISSING",
                description="Training sample must include a non-empty label.",
            )
        return label

    def train(
        self,
        labeled_transcripts: Iterable[dict[str, Any]],
        test_size: float = CORE_CONFIG.intent_test_split_ratio,
    ) -> float:
        samples = list(labeled_transcripts)
        if len(samples) < 2:
            raise AudioProcessingError(
                error_code="INTENT_TRAINING_DATA_INSUFFICIENT",
                description="At least 2 labeled transcript samples are required for training.",
            )

        texts = [self._extract_transcript(sample) for sample in samples]
        labels = [self._extract_label(sample) for sample in samples]

        self.vectorizer = TfidfVectorizer(
            ngram_range=CORE_CONFIG.intent_tfidf_ngram_range,
            min_df=CORE_CONFIG.intent_tfidf_min_df,
        )
        x = self.vectorizer.fit_transform(texts)
        y = np.array(labels)

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y if len(set(y.tolist())) > 1 else None,
        )

        self.classifier = RandomForestClassifier(
            n_estimators=CORE_CONFIG.intent_random_forest_n_estimators,
            random_state=self.random_state,
            n_jobs=CORE_CONFIG.intent_random_forest_n_jobs,
        )
        self.classifier.fit(x_train, y_train)

        predictions = self.classifier.predict(x_test)
        model_f1 = float(f1_score(y_test, predictions, pos_label=self._coercion_label))

        if model_f1 < INTENT_MIN_F1:
            raise AudioProcessingError(
                error_code="INTENT_MIN_F1_NOT_MET",
                description=(
                    f"Intent training F1 ({model_f1:.3f}) is below required "
                    f"minimum ({INTENT_MIN_F1:.2f})."
                ),
            )

        dense_train = x_train.toarray()
        self.kmeans_model = KMeans(
            n_clusters=min(self.n_clusters, max(1, dense_train.shape[0])),
            random_state=self.random_state,
            n_init=CORE_CONFIG.intent_kmeans_n_init,
        )
        self.kmeans_model.fit(dense_train)

        train_cluster_ids = self.kmeans_model.predict(dense_train)
        self.cluster_distance_thresholds = {}
        for cluster_id in np.unique(train_cluster_ids):
            cluster_mask = train_cluster_ids == cluster_id
            cluster_vectors = dense_train[cluster_mask]
            center = self.kmeans_model.cluster_centers_[cluster_id]
            distances = np.linalg.norm(cluster_vectors - center, axis=1)
            self.cluster_distance_thresholds[int(cluster_id)] = float(
                distances.mean()
                + CORE_CONFIG.intent_anomaly_std_multiplier * distances.std(ddof=0)
            )

        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        artifact = {
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "kmeans": self.kmeans_model,
            "cluster_distance_thresholds": self.cluster_distance_thresholds,
            "coercion_label": self._coercion_label,
        }
        joblib.dump(artifact, self.model_path)
        # Write intent training report with KMeans metadata
        try:
            report_payload = {
                "kmeans": {
                    "n_clusters": int(getattr(self.kmeans_model, "n_clusters", self.n_clusters)),
                    "assignment_method": "KMeans (Euclidean distance)",
                },
                "intent_training": {
                    "model_type": "random_forest",
                    "f1_score": model_f1,
                    "sample_count": len(samples),
                },
            }
            os.makedirs(os.path.dirname(self.training_report_path) or ".", exist_ok=True)
            with open(self.training_report_path, "w", encoding="utf-8") as report_file:
                json.dump(report_payload, report_file, indent=2)
        except Exception:
            # Do not fail training if report writing fails; log silently.
            pass
        return model_f1

    def _compute_anomaly_flag(self, transcript: str) -> bool:
        if self.vectorizer is None or self.kmeans_model is None:
            return False

        sparse_vector = self.vectorizer.transform([transcript])
        if hasattr(sparse_vector, "toarray"):
            vector = getattr(sparse_vector, "toarray")()[0]
        else:
            vector = np.asarray(sparse_vector)[0]
        cluster_id = int(self.kmeans_model.predict(vector.reshape(1, -1))[0])
        center = self.kmeans_model.cluster_centers_[cluster_id]
        distance = float(np.linalg.norm(vector - center))
        threshold = self.cluster_distance_thresholds.get(cluster_id)
        if threshold is None:
            return True
        return distance > threshold

    def _analyze_sync(self, audio_file_path: str) -> IntentResult:
        start_time = perf_counter()
        self._validate_audio_file(audio_file_path)

        transcript = self._transcribe_audio(audio_file_path)
        no_speech_detected = not transcript
        if no_speech_detected:
            intent_score = 0.0
        else:
            intent_score = self._score_transcript(transcript)
        anomaly_flag = self._compute_anomaly_flag(transcript) if transcript else False

        processing_time_ms = int((perf_counter() - start_time) * 1000)
        return IntentResult(
            intent_score=intent_score,
            transcript=transcript,
            no_speech_detected=no_speech_detected,
            anomaly_flag=anomaly_flag,
            processing_time_ms=processing_time_ms,
        )

    def analyze(self, audio_file_path: str) -> IntentResult:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._analyze_sync, audio_file_path)
            try:
                return future.result(timeout=self.timeout_seconds)
            except TimeoutError as exc:
                future.cancel()
                raise AudioProcessingError(
                    error_code="INTENT_TIMEOUT",
                    description=f"Intent analysis exceeded timeout of {self.timeout_seconds} seconds.",
                ) from exc
