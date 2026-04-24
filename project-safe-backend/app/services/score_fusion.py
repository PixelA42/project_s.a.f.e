"""
Score Fusion — decision layer.
Combines spectral_score + intent_score → final_score + risk_label.
No external API calls — pure arithmetic and threshold logic.
"""
from dataclasses import dataclass
from flask import current_app


@dataclass
class FusionResult:
    final_score: float
    spectral_score: float
    intent_score: float
    risk_label: str       # "HIGH_RISK" | "PRANK" | "SAFE"
    reasoning: str


def fuse(spectral_score: float, intent_score: float) -> FusionResult:
    """
    Weighted fusion with dual-gate classification.

    HIGH_RISK gate: final_score >= HIGH_RISK_THRESHOLD AND intent_score >= 50
       → Requires both layers to fire. Prevents synthetic-but-harmless false positives.

    PRANK gate: spectral_score >= PRANK_THRESHOLD AND intent_score < 50
       → AI voice detected but no coercion in content.

    SAFE: everything else.
    """
    spectral_w: float      = current_app.config.get("SPECTRAL_WEIGHT", 0.6)
    intent_w: float        = current_app.config.get("INTENT_WEIGHT", 0.4)
    high_risk_threshold    = current_app.config.get("HIGH_RISK_THRESHOLD", 70)
    prank_threshold        = current_app.config.get("PRANK_THRESHOLD", 35)

    final_score = round(
        (spectral_score * spectral_w) + (intent_score * intent_w), 1
    )

    if final_score >= high_risk_threshold and intent_score >= 50:
        label = "HIGH_RISK"
        reasoning = (
            f"Dual threat confirmed: synthetic voice (spectral={spectral_score}) "
            f"and coercion language (intent={intent_score}) both exceed thresholds."
        )
    elif spectral_score >= prank_threshold and intent_score < 50:
        label = "PRANK"
        reasoning = (
            f"AI-generated voice detected (spectral={spectral_score}) "
            f"but content is harmless (intent={intent_score}). "
            f"Likely synthetic but non-threatening."
        )
    else:
        label = "SAFE"
        reasoning = (
            f"No significant threat: spectral={spectral_score}, "
            f"intent={intent_score}. Voice and content within safe parameters."
        )

    return FusionResult(
        final_score=final_score,
        spectral_score=round(spectral_score, 1),
        intent_score=round(intent_score, 1),
        risk_label=label,
        reasoning=reasoning,
    )