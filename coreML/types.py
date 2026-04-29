"""Shared result contracts for core ML analyzers."""

from dataclasses import dataclass


@dataclass(slots=True)
class SpectralResult:
    spectral_score: float
    anomaly_flag: bool
    processing_time_ms: int


@dataclass(slots=True)
class IntentResult:
    intent_score: float
    transcript: str
    no_speech_detected: bool
    anomaly_flag: bool
    processing_time_ms: int


@dataclass(slots=True)
class InferenceResult:
    """Inference result with support for uncertainty flagging."""

    final_score: float  # Fused score [0, 1]
    predicted_label: int  # 0=human, 1=ai
    confidence: float  # Confidence score
    is_uncertain: bool  # True if in uncertainty zone [0.35, 0.65]
    spectral_score: float | None = None  # Raw spectral score
    intent_score: float | None = None  # Raw intent score
    processing_time_ms: int = 0
    model_name: str = "baseline_ensemble"
    risk_label: str = "SAFE"  # HIGH_RISK, PRANK, or SAFE
