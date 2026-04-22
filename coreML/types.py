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
