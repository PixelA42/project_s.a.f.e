"""Core ML package for Project S.A.F.E."""

from .constants import (
    API_TIMEOUT_SECONDS,
    INTENT_MIN_F1,
    INTENT_TIMEOUT_SECONDS,
    MFCC_N_COEFFICIENTS,
    SCORE_MAX,
    SCORE_MIN,
    SPECTRAL_MIN_F1,
    SPECTRAL_TIMEOUT_SECONDS,
)
from .errors import AudioProcessingError
from .intent_analyzer import IntentAnalyzer
from .risk_classifier import RiskClassifier, RiskLabel
from .score_fuser import ScoreFuser
from .spectral_analyzer import SpectralAnalyzer
from .types import IntentResult, SpectralResult
from .utils import verify_serialization

__all__ = [
    "API_TIMEOUT_SECONDS",
    "AudioProcessingError",
    "INTENT_MIN_F1",
    "INTENT_TIMEOUT_SECONDS",
    "IntentAnalyzer",
    "IntentResult",
    "MFCC_N_COEFFICIENTS",
    "RiskClassifier",
    "RiskLabel",
    "SCORE_MAX",
    "SCORE_MIN",
    "ScoreFuser",
    "SPECTRAL_MIN_F1",
    "SpectralAnalyzer",
    "SPECTRAL_TIMEOUT_SECONDS",
    "SpectralResult",
    "verify_serialization",
]
