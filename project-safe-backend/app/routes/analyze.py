"""
POST /api/v1/analyze — the main endpoint.

Accepts audio bytes and/or a transcript, runs both scoring layers,
fuses the results, and returns a ScoreResponse matching the frontend's
TypeScript type definition.
"""
import base64
import json
from datetime import datetime, timezone

from flask import Blueprint, request, jsonify, current_app
from pydantic import ValidationError

from app import limiter
from app.utils.validators import AnalyzeRequest
from app.services import spectral_engine, intent_engine, score_fusion

analyze_bp = Blueprint("analyze", __name__)

# ── Mock data for forced-label dev mode ───────────────────────────────
_MOCK_SCORES = {
    "HIGH_RISK": {"spectral": 94.0, "intent": 88.0},
    "PRANK": {"spectral": 78.0, "intent": 12.0},
    "SAFE": {"spectral": 8.0, "intent": 5.0},
}


@analyze_bp.route("/analyze", methods=["POST"])
@limiter.limit("20 per minute")
def analyze():
    """
    Analyze an incoming call and return a risk assessment.

    Request body (JSON):
    {
        "audio_b64":     "<base64 audio>",   // optional
        "transcript":    "<call text>",       // optional
        "mock_label":    "HIGH_RISK",         // dev-only
        "caller_id":     "John Doe",          // optional
        "caller_number": "+1 555 1234"        // optional
    }

    Response (200):
    {
        "risk_label":     "HIGH_RISK" | "PRANK" | "SAFE",
        "final_score":    92.0,
        "spectral_score": 94.0,
        "intent_score":   88.0,
        "caller_id":      "John Doe",
        "caller_number":  "+1 555 1234",
        "reasoning":      "...",
        "timestamp":      "2026-04-23T12:00:00Z"
    }
    """
    # ── Parse and validate ─────────────────────────────────────
    try:
        body = AnalyzeRequest.model_validate(request.get_json(force=True) or {})
    except ValidationError as e:
        return jsonify({"error": "Invalid request", "details": e.errors()}), 422
    except Exception:
        return jsonify({"error": "Could not parse request body"}), 400

    # ── Dev mode: forced mock label ────────────────────────────
    if body.mock_label:
        scores = _MOCK_SCORES[body.mock_label]
        result = score_fusion.fuse(
            spectral_score=scores["spectral"],
            intent_score=scores["intent"],
        )
        current_app.logger.info(
            f"[MOCK] label={result.risk_label} | "
            f"final={result.final_score} | "
            f"caller={body.caller_id}"
        )
        return jsonify(_build_response(result, body)), 200

    # ── Decode audio if provided ───────────────────────────────
    audio_bytes: bytes | None = None
    if body.audio_b64:
        try:
            audio_bytes = base64.b64decode(body.audio_b64)
        except Exception:
            return jsonify({"error": "Invalid base64 audio data"}), 422

    # ── Run both scoring layers in parallel (simple sequential for now) ─
    # TODO: switch to asyncio or threading.Thread for true parallelism
    spectral_score = spectral_engine.analyze(
        audio_bytes=audio_bytes,
        transcript=body.transcript,
    )
    intent_score = intent_engine.analyze(
        transcript=body.transcript,
    )

    # ── Fuse scores → risk label ───────────────────────────────
    result = score_fusion.fuse(spectral_score, intent_score)

    current_app.logger.info(
        f"label={result.risk_label} | "
        f"final={result.final_score} | "
        f"spectral={result.spectral_score} | "
        f"intent={result.intent_score} | "
        f"caller={body.caller_id}"
    )

    return jsonify(_build_response(result, body)), 200


def _build_response(result: score_fusion.FusionResult, body: AnalyzeRequest) -> dict:
    return {
        "risk_label": result.risk_label,
        "final_score": result.final_score,
        "spectral_score": result.spectral_score,
        "intent_score": result.intent_score,
        "caller_id": body.caller_id,
        "caller_number": body.caller_number,
        "reasoning": result.reasoning,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }