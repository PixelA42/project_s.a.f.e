"""
Score Fusion — the decision layer.
Combines spectral_score and intent_score into a final_score and risk_label.

This is where the "dual-layer prevents cry-wolf" logic lives:
  - HIGH_RISK requires BOTH layers to be elevated
  - PRANK fires when spectral is high but intent is low (synthetic but harmless)
  - SAFE is the default when neither layer is alarmed
"""
from dataclasses import dataclass
from flask import current_app


@dataclass
class FusionResult:
    final_score: float
    spectral_score: float
    intent_score: float
    risk_label: str  # "HIGH_RISK" | "PRANK" | "SAFE"
    reasoning: str


def fuse(spectral_score: float, intent_score: float) -> FusionResult:
    """
    Weighted fusion of both scoring layers.

    Weights are configurable via .env:
        SPECTRAL_WEIGHT=0.6
        INTENT_WEIGHT=0.4

    Thresholds are configurable via .env:
        HIGH_RISK_THRESHOLD=70
        PRANK_THRESHOLD=35
    """
    spectral_w: float = current_app.config.get("SPECTRAL_WEIGHT", 0.6)
    intent_w: float = current_app.config.get("INTENT_WEIGHT", 0.4)
    high_risk_threshold: int = current_app.config.get("HIGH_RISK_THRESHOLD", 70)
    prank_threshold: int = current_app.config.get("PRANK_THRESHOLD", 35)

    # Weighted average
    final_score = round(
        (spectral_score * spectral_w) + (intent_score * intent_w), 1
    )

    # ── Classification logic ───────────────────────────────────
    # HIGH_RISK: overall score is elevated AND intent confirms coercion.
    # This dual-gate prevents single-layer false positives.
    if final_score >= high_risk_threshold and intent_score >= 50:
        label = "HIGH_RISK"
        reasoning = (
            f"Dual threat: spectral={spectral_score} (synthetic voice), "
            f"intent={intent_score} (coercion signals). "
            f"Both layers exceeded thresholds simultaneously."
        )

    # PRANK: synthetic voice detected but NO coercion in the content.
    elif spectral_score >= prank_threshold and intent_score < 50:
        label = "PRANK"
        reasoning = (
            f"Synthetic voice detected (spectral={spectral_score}) "
            f"but content is harmless (intent={intent_score}). "
            f"AI-generated voice, non-threatening."
        )

    # SAFE: no significant threat signal from either layer.
    else:
        label = "SAFE"
        reasoning = (
            f"No significant threat. "
            f"spectral={spectral_score}, intent={intent_score}. "
            f"Voice and content both within safe parameters."
        )

    return FusionResult(
        final_score=final_score,
        spectral_score=round(spectral_score, 1),
        intent_score=round(intent_score, 1),
        risk_label=label,
        reasoning=reasoning,
    )