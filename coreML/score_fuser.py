"""Weighted score fusion for spectral and intent outputs."""

from .constants import SCORE_MAX, SCORE_MIN


class ScoreFuser:
    """Fuse spectral and intent signals into a final risk score."""

    spectral_weight: float = 0.70
    intent_weight: float = 0.30

    @staticmethod
    def _validate_score(name: str, value: float) -> None:
        if value < SCORE_MIN or value > SCORE_MAX:
            raise ValueError(f"{name} must be in the range [{SCORE_MIN}, {SCORE_MAX}].")

    def fuse(self, spectral_score: float, intent_score: float) -> float:
        self._validate_score("spectral_score", spectral_score)
        self._validate_score("intent_score", intent_score)

        final_score = (self.spectral_weight * spectral_score) + (
            self.intent_weight * intent_score
        )
        return float(final_score)
