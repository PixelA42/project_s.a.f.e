"""Risk classification rules for fused fraud scores."""

from enum import StrEnum

from .constants import (
    HIGH_RISK_FINAL_SCORE_THRESHOLD,
    HIGH_RISK_INTENT_SCORE_THRESHOLD,
    PRANK_FINAL_SCORE_THRESHOLD,
    PRANK_INTENT_SCORE_THRESHOLD,
)


class RiskLabel(StrEnum):
    """Enumerated risk labels exposed to API and UI layers."""

    HIGH_RISK = "HIGH_RISK"
    PRANK = "PRANK"
    SAFE = "SAFE"


class RiskClassifier:
    """Classify fused scores into one actionable risk label."""

    high_risk_final_score_threshold: float = HIGH_RISK_FINAL_SCORE_THRESHOLD
    high_risk_intent_score_threshold: float = HIGH_RISK_INTENT_SCORE_THRESHOLD
    prank_final_score_threshold: float = PRANK_FINAL_SCORE_THRESHOLD
    prank_intent_score_threshold: float = PRANK_INTENT_SCORE_THRESHOLD

    def classify(self, final_score: float, intent_score: float) -> RiskLabel:
        if (
            final_score > self.high_risk_final_score_threshold
            and intent_score > self.high_risk_intent_score_threshold
        ):
            return RiskLabel.HIGH_RISK
        if (
            final_score > self.prank_final_score_threshold
            and intent_score < self.prank_intent_score_threshold
        ):
            return RiskLabel.PRANK
        return RiskLabel.SAFE
