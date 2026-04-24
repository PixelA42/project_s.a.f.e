"""Shared constants backed by the centralized ML config."""

from __future__ import annotations

from config import SETTINGS


CORE_CONFIG = SETTINGS.core
FUSION_CONFIG = SETTINGS.fusion

SCORE_MIN: float = CORE_CONFIG.score_min
SCORE_MAX: float = CORE_CONFIG.score_max

MFCC_N_COEFFICIENTS: int = CORE_CONFIG.spectral_mfcc_n_coefficients

SPECTRAL_TIMEOUT_SECONDS: int = CORE_CONFIG.spectral_timeout_seconds
INTENT_TIMEOUT_SECONDS: int = CORE_CONFIG.intent_timeout_seconds
API_TIMEOUT_SECONDS: int = CORE_CONFIG.api_timeout_seconds

SPECTRAL_MIN_F1: float = CORE_CONFIG.spectral_min_f1
INTENT_MIN_F1: float = CORE_CONFIG.intent_min_f1

SCORE_FUSION_SPECTRAL_WEIGHT: float = FUSION_CONFIG.spectral_weight
SCORE_FUSION_INTENT_WEIGHT: float = FUSION_CONFIG.intent_weight
HIGH_RISK_FINAL_SCORE_THRESHOLD: float = FUSION_CONFIG.high_risk_final_score_threshold
HIGH_RISK_INTENT_SCORE_THRESHOLD: float = FUSION_CONFIG.high_risk_intent_score_threshold
PRANK_FINAL_SCORE_THRESHOLD: float = FUSION_CONFIG.prank_final_score_threshold
PRANK_INTENT_SCORE_THRESHOLD: float = FUSION_CONFIG.prank_intent_score_threshold
