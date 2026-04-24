import base64
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from flask import Blueprint, request, jsonify, current_app, g
from flasgger import swag_from
from pydantic import ValidationError

from app import limiter
from app.utils.validators import AnalyzeRequest
from app.utils.auth_helpers import optional_auth
from app.services import spectral_engine, intent_engine, score_fusion
from app.models import user_store

analyze_bp = Blueprint("analyze", __name__)

_MOCK_SCORES = {
    "HIGH_RISK": {"spectral": 94.0, "intent": 88.0},
    "PRANK":     {"spectral": 78.0, "intent": 12.0},
    "SAFE":      {"spectral":  8.0, "intent":  5.0},
}


@analyze_bp.route("/analyze", methods=["POST"])
@limiter.limit("20 per minute")
@optional_auth
def analyze():
    """
    Analyze an incoming call for synthetic voice and coercion signals.
    ---
    tags:
      - Analyze
    summary: Score a call — returns HIGH_RISK, PRANK, or SAFE
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            audio_b64:
              type: string
              description: Base64-encoded audio file (WAV or MP3). Used by the spectral engine.
            transcript:
              type: string
              description: Call transcript text. Used by the intent engine.
            mock_label:
              type: string
              enum: [HIGH_RISK, PRANK, SAFE]
              description: Dev-only. Forces a specific risk label without running the engines.
            caller_id:
              type: string
              example: John Doe
            caller_number:
              type: string
              example: "+1 555 234 5678"
    responses:
      200:
        description: Analysis complete
        schema:
          type: object
          properties:
            risk_label:
              type: string
              enum: [HIGH_RISK, PRANK, SAFE]
              example: HIGH_RISK
            final_score:
              type: number
              example: 92.0
              description: Weighted fusion of spectral and intent scores (0–100)
            spectral_score:
              type: number
              example: 94.0
              description: Acoustic/voice forensics score (0–100)
            intent_score:
              type: number
              example: 88.0
              description: NLP coercion detection score (0–100)
            caller_id:
              type: string
              example: Unknown Caller
            caller_number:
              type: string
              example: "+1 800 555 0199"
            reasoning:
              type: string
              example: "Dual threat confirmed: synthetic voice and coercion language both exceed thresholds."
            timestamp:
              type: string
              example: "2026-04-24T12:00:00+00:00"
      400:
        description: Could not parse request body
      422:
        description: Validation error
      429:
        description: Rate limit exceeded
    """
    try:
        body = AnalyzeRequest.model_validate(request.get_json(force=True) or {})
    except ValidationError as e:
        return jsonify({"error": "Invalid request", "details": e.errors()}), 422
    except Exception:
        return jsonify({"error": "Could not parse request body"}), 400

    if body.mock_label:
        scores = _MOCK_SCORES[body.mock_label]
        result = score_fusion.fuse(
            spectral_score=scores["spectral"],
            intent_score=scores["intent"],
        )
        _update_user_stats(result.risk_label)
        current_app.logger.info(
            f"[Analyze] MOCK label={result.risk_label} caller={body.caller_id}"
        )
        return jsonify(_build_response(result, body)), 200

    audio_bytes: bytes | None = None
    if body.audio_b64:
        try:
            audio_bytes = base64.b64decode(body.audio_b64)
        except Exception:
            return jsonify({"error": "Invalid base64 audio data"}), 422

    transcript = body.transcript
    if audio_bytes and not transcript and current_app.config.get("USE_REAL_TRANSCRIPTION"):
        dg_result = spectral_engine.transcribe_with_deepgram(audio_bytes)
        if dg_result:
            transcript = dg_result.get("transcript", "")

    spectral_score = 0.0
    intent_score   = 0.0

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_spectral = pool.submit(spectral_engine.analyze, audio_bytes, transcript)
        future_intent   = pool.submit(intent_engine.analyze, transcript)
        for future in as_completed([future_spectral, future_intent]):
            if future is future_spectral:
                spectral_score = future.result()
            else:
                intent_score = future.result()

    result = score_fusion.fuse(spectral_score, intent_score)
    _update_user_stats(result.risk_label)

    current_app.logger.info(
        f"[Analyze] label={result.risk_label} final={result.final_score} "
        f"spectral={result.spectral_score} intent={result.intent_score}"
    )
    return jsonify(_build_response(result, body)), 200


def _build_response(result: score_fusion.FusionResult, body: AnalyzeRequest) -> dict:
    return {
        "risk_label":     result.risk_label,
        "final_score":    result.final_score,
        "spectral_score": result.spectral_score,
        "intent_score":   result.intent_score,
        "caller_id":      body.caller_id,
        "caller_number":  body.caller_number,
        "reasoning":      result.reasoning,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
    }


def _update_user_stats(risk_label: str) -> None:
    user = getattr(g, "current_user", None)
    if user:
        user_store.increment_stats(
            user["id"],
            threat_blocked=(risk_label == "HIGH_RISK"),
        )