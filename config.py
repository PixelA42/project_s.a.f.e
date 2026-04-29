"""Centralized configuration for Project S.A.F.E. ML codepaths."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


@dataclass(frozen=True)
class GeneralConfig:
    random_seed: int = 42


@dataclass(frozen=True)
class PathConfig:
    project_root: Path = PROJECT_ROOT
    audio_dataset_dir: Path = _resolve("Audios")
    spectrogram_dir: Path = _resolve("spectrograms")
    outputs_dir: Path = _resolve("outputs")
    experiment_log_path: Path = _resolve("outputs", "experiment_log.csv")
    supervised_outputs_dir: Path = _resolve("outputs", "supervised")
    unsupervised_outputs_dir: Path = _resolve("outputs", "unsupervised")
    labels_csv_path: Path = _resolve("labels.csv")
    dataset_stats_path: Path = _resolve("dataset_stats.txt")
    models_dir: Path = _resolve("models")
    training_report_path: Path = _resolve("models", "training_report.json")
    embedding_csv_path: Path = _resolve("models", "feature_embedding_2d.csv")
    embedding_plot_path: Path = _resolve("models", "feature_embedding_2d.png")
    supervised_model_path: Path = _resolve("models", "spectrogram_supervised_model.joblib")
    semisupervised_model_path: Path = _resolve(
        "models", "spectrogram_semisupervised_model.joblib"
    )
    unsupervised_model_path: Path = _resolve("models", "spectrogram_kmeans_model.joblib")
    feature_pipeline_path: Path = _resolve("models", "spectrogram_feature_pipeline.joblib")
    temp_spectrogram_path: Path = _resolve("tmp_uploaded_spec.png")
    spectral_model_path: Path = _resolve("models", "spectral_model.joblib")
    spectral_training_report_path: Path = _resolve("models", "spectral_training_report.json")
    intent_model_path: Path = _resolve("models", "intent_model.joblib")
    intent_training_report_path: Path = _resolve("models", "intent_training_report.json")
    intent_keywords_csv_path: Path = _resolve("data_pipeline", "distress_keywords_v1.csv")


@dataclass(frozen=True)
class AudioDataConfig:
    supported_extensions: tuple[str, ...] = (".mp3", ".wav", ".flac", ".ogg", ".m4a")
    categories: tuple[str, ...] = ("human", "ai")
    label_map: dict[str, int] = field(default_factory=lambda: {"human": 0, "ai": 1})
    clip_duration_seconds: int = 5
    sample_rate: int = 22050
    n_mels: int = 128
    mel_fmax: int = 8000
    spectrogram_image_size: int = 224
    feature_image_size: int = 64
    figure_size_inches: float = 2.24
    colormap: str = "magma"
    augmentation_enabled: bool = True
    augmentation_copies: int = 4
    augmentation_types: tuple[str, ...] = (
        "noise",
        "pitch_up",
        "pitch_down",
        "stretch",
        "volume_up",
        "volume_down",
    )
    augmentation_noise_scale: float = 0.005
    augmentation_pitch_up_steps: int = 2
    augmentation_pitch_down_steps: int = -2
    augmentation_time_stretch_rate: float = 0.85
    augmentation_volume_up_gain: float = 1.5
    augmentation_volume_down_gain: float = 0.5

    @property
    def spectrogram_dpi(self) -> float:
        return self.spectrogram_image_size / self.figure_size_inches


@dataclass(frozen=True)
class TrainingConfig:
    test_split_ratio: float = 0.4
    pca_components: int = 40
    embedding_method: str = "tsne"
    pseudo_threshold: float = 0.85
    decision_threshold: float = 0.5
    semi_unlabeled_ratio: float = 0.3
    random_forest_n_estimators: int = 350
    random_forest_n_jobs: int = -1
    positive_class_weight_multiplier: float = 1.5
    kmeans_max_clusters: int = 8
    kmeans_n_init: str = "auto"
    tsne_perplexity_cap: int = 30
    embedding_plot_size: tuple[int, int] = (8, 6)
    embedding_plot_dpi: int = 180
    embedding_plot_alpha: float = 0.75
    embedding_plot_marker_size: int = 18
    embedding_grid_alpha: float = 0.2


@dataclass(frozen=True)
class DataLoadingConfig:
    test_split_ratio: float = 0.2
    numeric_fill_threshold: float = 0.10
    numeric_interpolate_threshold: float = 0.40
    categorical_fill_threshold: float = 0.20
    categorical_missing_bucket_threshold: float = 0.50
    identifier_unique_ratio_threshold: float = 0.95
    supported_tabular_extensions: tuple[str, ...] = (
        ".csv",
        ".json",
        ".parquet",
        ".xlsx",
        ".xls",
    )
    unlabeled_directory_markers: tuple[str, ...] = (
        "unlabeled",
        "unknown",
        "unsorted",
        "misc",
    )
    real_directory_markers: tuple[str, ...] = (
        "real",
        "human",
        "genuine",
        "organic",
    )
    synthetic_directory_markers: tuple[str, ...] = (
        "ai",
        "synthetic",
        "tts",
        "xtts",
        "openai",
        "voicebox",
        "valle",
        "flashspeech",
        "seedtts",
        "naturalspeech",
    )
    target_candidate_names: tuple[str, ...] = (
        "target",
        "label",
        "class",
        "y",
        "output",
        "risk_label",
    )


@dataclass(frozen=True)
class CoreMLConfig:
    score_min: float = 0.0
    score_max: float = 100.0
    spectral_mfcc_n_coefficients: int = 40
    spectral_timeout_seconds: int = 5
    intent_timeout_seconds: int = 10
    api_timeout_seconds: int = 15
    spectral_min_f1: float = 0.75
    intent_min_f1: float = 0.70
    spectral_model_type: str = "random_forest"
    spectral_test_split_ratio: float = 0.2
    spectral_random_forest_n_estimators: int = 300
    spectral_random_forest_n_jobs: int = -1
    spectral_cluster_count: int = 8
    spectral_kmeans_n_init: str = "auto"
    spectral_anomaly_std_multiplier: float = 2.0
    spectral_positive_label: str = "synthetic"
    intent_test_split_ratio: float = 0.2
    intent_random_forest_n_estimators: int = 300
    intent_random_forest_n_jobs: int = -1
    intent_cluster_count: int = 8
    intent_kmeans_n_init: str = "auto"
    intent_anomaly_std_multiplier: float = 2.0
    intent_tfidf_ngram_range: tuple[int, int] = (1, 2)
    intent_tfidf_min_df: int = 1
    intent_whisper_model_name: str = "base"
    intent_keyword_weights: dict[str, float] = field(
        default_factory=lambda: {
            "bail": 1.0,
            "transfer": 1.0,
            "otp": 0.9,
            "upi": 1.0,
            "money": 0.9,
            "accident": 0.8,
            "hospital": 0.8,
            "help": 0.6,
            "police": 0.7,
        }
    )
    intent_time_pressure_phrases: tuple[tuple[str, float], ...] = (
        ("right now", 0.8),
        ("asap", 0.8),
        ("now", 0.4),
        ("please hurry", 0.9),
        ("urgent", 0.9),
        ("call me back", 0.5),
        ("send money", 1.0),
    )
    intent_nltk_urgency_bonus: float = 0.5
    intent_fallback_urgency_bonus: float = 0.3
    intent_imperative_verb_bonus: float = 0.2
    intent_normalize_min_raw: float = 0.0
    intent_normalize_max_raw: float = 5.0
    intent_positive_label: str = "coercive"


@dataclass(frozen=True)
class FusionConfig:
    spectral_weight: float = 0.70
    intent_weight: float = 0.30
    high_risk_final_score_threshold: float = 75.0
    high_risk_intent_score_threshold: float = 60.0
    prank_final_score_threshold: float = 75.0
    prank_intent_score_threshold: float = 40.0


@dataclass(frozen=True)
class MLConfig:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    audio: AudioDataConfig = field(default_factory=AudioDataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataLoadingConfig = field(default_factory=DataLoadingConfig)
    core: CoreMLConfig = field(default_factory=CoreMLConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)


SETTINGS = MLConfig()
