"""Risk classification rules for fused fraud scores."""

from enum import StrEnum


class RiskLabel(StrEnum):
    """Enumerated risk labels exposed to API and UI layers."""

    HIGH_RISK = "HIGH_RISK"
    PRANK = "PRANK"
    SAFE = "SAFE"


class RiskClassifier:
    """Classify fused scores into one actionable risk label."""

    def classify(self, final_score: float, intent_score: float) -> RiskLabel:
        if final_score > 75 and intent_score > 60:
            return RiskLabel.HIGH_RISK
        if final_score > 75 and intent_score < 40:
            return RiskLabel.PRANK
        return RiskLabel.SAFE
