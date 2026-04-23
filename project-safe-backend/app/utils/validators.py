"""Request validation schemas for API routes."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AnalyzeRequest(BaseModel):
    """Validated payload for POST /api/v1/analyze."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    audio_b64: str | None = None
    transcript: str | None = None
    mock_label: Literal["HIGH_RISK", "PRANK", "SAFE"] | None = None
    caller_id: str | None = Field(default=None, max_length=120)
    caller_number: str | None = Field(default=None, max_length=40)

    @model_validator(mode="after")
    def validate_content(self):
        """Require at least one analysis source unless forced mock label is set."""
        if self.mock_label:
            return self

        has_audio = bool(self.audio_b64)
        has_transcript = bool(self.transcript)
        if not has_audio and not has_transcript:
            raise ValueError(
                "Provide at least one of: audio_b64 or transcript"
            )
        return self
