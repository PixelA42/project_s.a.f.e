import base64
import hashlib
import io
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone

from flask import Blueprint, current_app, g, jsonify, request
from pydantic import ValidationError

from app import limiter
from app.database import get_session, record_call_log
from app.models import user_store
from app.services import intent_engine, score_fusion, spectral_engine
from app.utils.auth_helpers import optional_auth
from app.utils.validators import (
  AnalyzeRequest,
  ErrorResponse,
  EvaluateRiskRequest,
  EvaluateRiskResponse,
)

analyze_bp = Blueprint("analyze", __name__)
MAX_AUDIO_BYTES = 25 * 1024 * 1024
ALLOWED_AUDIO_EXTENSIONS = {"wav", "mp3"}
SPECTRAL_TIMEOUT_SECONDS = 5
INTENT_TIMEOUT_SECONDS = 10
API_TIMEOUT_SECONDS = 15

_MOCK_SCORES = {
    "HIGH_RISK": {"spectral": 94.0, "intent": 88.0},
    "PRANK":     {"spectral": 78.0, "intent": 12.0},
    "SAFE":      {"spectral":  8.0, "intent":  5.0},
}


def _request_id() -> str:
  return getattr(g, "request_id", None) or str(uuid.uuid4())


def _json_error(error: str, status_code: int, description: str | None = None, field: str | None = None):
  payload = ErrorResponse(error=error, description=description, field=field).model_dump(exclude_none=True)
  return jsonify(payload), status_code


def _extract_audio_upload():
  upload = request.files.get("file") or request.files.get("audio") or request.files.get("audio_file")
  if upload is None or not upload.filename:
    return None, _json_error("validation_error", 422, "Audio file is required", "file")

  filename = upload.filename
  extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
  if extension not in ALLOWED_AUDIO_EXTENSIONS:
    return None, _json_error(
      "validation_error",
      422,
      "Only WAV and MP3 audio files are supported",
      "file",
    )

  audio_bytes = upload.read()
  if len(audio_bytes) > MAX_AUDIO_BYTES:
    return None, _json_error(
      "payload_too_large",
      413,
      "Audio file exceeds the 25 MB limit",
      "file",
    )

  if not audio_bytes:
    return None, _json_error("validation_error", 422, "Audio file is empty", "file")

  return (audio_bytes, filename), None


def _run_with_timeout(func, timeout_seconds: int, *args, **kwargs):
  app = current_app._get_current_object()

  def _call():
    with app.app_context():
      return func(*args, **kwargs)

  with ThreadPoolExecutor(max_workers=1) as pool:
    future = pool.submit(_call)
    return future.result(timeout=timeout_seconds)


def _attempt_transcription(audio_bytes: bytes, filename: str) -> str:
  if current_app.config.get("USE_REAL_TRANSCRIPTION"):
    result = spectral_engine.transcribe_with_deepgram(audio_bytes)
    if result and result.get("transcript"):
      return str(result.get("transcript", "")).strip()

  whisper_model = getattr(current_app, "whisper_model", None)
  if whisper_model is not None:
    try:
      import numpy as np
      import soundfile as sf

      audio, sample_rate = sf.read(io.BytesIO(audio_bytes))
      if audio.ndim > 1:
        audio = audio.mean(axis=1)

      if hasattr(whisper_model, "transcribe"):
        result = whisper_model.transcribe(audio, fp16=False)
        transcript = result.get("text", "")
        if transcript:
          return str(transcript).strip()
    except Exception:
      current_app.logger.exception("[Intent] local transcription failed for %s", filename)

  return ""


def _classify_risk(spectral_score: float, intent_score: float) -> tuple[float, str]:
  final_score = round((0.70 * spectral_score) + (0.30 * intent_score), 1)
  if final_score > 75 and intent_score > 60:
    return final_score, "HIGH_RISK"
  if final_score > 75 and intent_score < 40:
    return final_score, "PRANK"
  return final_score, "SAFE"


def _fast_spectral_score(audio_bytes: bytes, filename: str) -> float:
  digest = hashlib.blake2b(audio_bytes + filename.encode("utf-8"), digest_size=8).digest()
  normalized = int.from_bytes(digest, byteorder="big") / float(2**64 - 1)
  return round(10.0 + (normalized * 80.0), 1)


def _persist_call_log(
  *,
  caller_id: str | None,
  audio_file_path: str,
  transcript: str | None,
  spectral_score: float,
  intent_score: float,
  final_score: float,
  risk_label: str,
  anomaly_flag: bool,
) -> tuple[bool, str | None]:
  request_id = _request_id()
  try:
    with get_session() as session:
      record_call_log(
        session,
        caller_id=caller_id,
        audio_file_path=audio_file_path,
        transcript=transcript,
        spectral_score=spectral_score,
        intent_score=intent_score,
        final_score=final_score,
        risk_label=risk_label,
        anomaly_flag=anomaly_flag,
      )
    return True, request_id
  except Exception:
    current_app.logger.exception("[Analyze] failed to persist call log request_id=%s", request_id)
    return False, request_id


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
      return jsonify({"error": "validation_error", "description": "Invalid request", "field": "body", "details": e.errors()}), 422
    except Exception:
      return jsonify({"error": "invalid_json", "description": "Could not parse request body"}), 400

    if body.mock_label:
      scores = _MOCK_SCORES[body.mock_label]
      result = score_fusion.fuse(spectral_score=scores["spectral"], intent_score=scores["intent"])
      _update_user_stats(result.risk_label)
      current_app.logger.info(
        f"[Analyze] MOCK label={result.risk_label} caller={body.caller_id}"
      )
      return _persist_and_respond(result, body, transcript=body.transcript)

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
    return _persist_and_respond(result, body, transcript=transcript)


@analyze_bp.route("/analyze-audio", methods=["POST"])
@limiter.limit("20 per minute")
def analyze_audio():
    """Analyze a multipart audio upload and return spectral scoring."""

    extracted, error = _extract_audio_upload()
    if error:
        return error

    audio_bytes, filename = extracted

    try:
        if current_app.config.get("USE_REAL_SPECTRAL_MODEL", False):
            spectral_score = _run_with_timeout(
                spectral_engine.analyze,
                SPECTRAL_TIMEOUT_SECONDS,
                audio_bytes,
                None,
            )
        else:
            spectral_score = _fast_spectral_score(audio_bytes, filename)
    except FuturesTimeoutError:
        return _json_error(
            "timeout",
            504,
            f"Spectral analysis exceeded {SPECTRAL_TIMEOUT_SECONDS} seconds",
            "file",
        )
    except Exception as exc:
        current_app.logger.exception("[AnalyzeAudio] spectral analysis failed")
        return _json_error("analysis_error", 500, str(exc), "file")

    return jsonify(
        {
            "spectral_score": round(float(spectral_score), 1),
            "anomaly_flag": False,
            "processing_time_ms": 0,
            "filename": filename,
        }
    ), 200
@analyze_bp.route("/analyze-intent", methods=["POST"])
@limiter.limit("20 per minute")
def analyze_intent():
    """Analyze a multipart audio upload and return transcript and intent scoring."""

    extracted, error = _extract_audio_upload()
    if error:
        return error

    audio_bytes, filename = extracted
    transcript = _attempt_transcription(audio_bytes, filename)
    no_speech_detected = not bool(transcript.strip())

    try:
        intent_score = _run_with_timeout(
            intent_engine.analyze,
            INTENT_TIMEOUT_SECONDS,
            transcript,
        )
    except FuturesTimeoutError:
        return _json_error(
            "timeout",
            504,
            f"Intent analysis exceeded {INTENT_TIMEOUT_SECONDS} seconds",
            "file",
        )
    except Exception as exc:
        current_app.logger.exception("[AnalyzeIntent] intent analysis failed")
        return _json_error("analysis_error", 500, str(exc), "file")

    if no_speech_detected:
        intent_score = 0.0

    return jsonify(
        {
            "intent_score": round(float(intent_score), 1),
            "transcript": transcript,
            "no_speech_detected": no_speech_detected,
            "anomaly_flag": False,
            "processing_time_ms": 0,
            "filename": filename,
        }
    ), 200
@analyze_bp.route("/evaluate-risk", methods=["POST"])
@limiter.limit("20 per minute")
def evaluate_risk():
    """Fuse spectral and intent scores and persist the call evaluation."""

    try:
        body = EvaluateRiskRequest.model_validate(request.get_json(force=True) or {})
    except ValidationError as e:
        return jsonify({"error": "validation_error", "description": "Invalid request", "field": "body", "details": e.errors()}), 422
    except Exception:
        return jsonify({"error": "invalid_json", "description": "Could not parse request body"}), 400

    final_score, risk_label = _classify_risk(body.spectral_score, body.intent_score)
    audio_file_path = body.audio_file_path or f"request://{_request_id()}"

    ok, request_id = _persist_call_log(
      caller_id=body.caller_id,
      audio_file_path=audio_file_path,
      transcript=None,
      spectral_score=float(body.spectral_score),
      intent_score=float(body.intent_score),
      final_score=final_score,
      risk_label=risk_label,
      anomaly_flag=bool(getattr(body, "anomaly_flag", False)),
    )
    if not ok:
        return jsonify({"error": "Failed to persist analysis result", "request_id": request_id}), 500

    response = EvaluateRiskResponse(
        final_score=final_score,
        risk_label=risk_label,
        spectral_score=round(float(body.spectral_score), 1),
        intent_score=round(float(body.intent_score), 1),
        caller_id=body.caller_id,
        audio_file_path=audio_file_path,
    ).model_dump(exclude_none=True)
    response["request_id"] = request_id
    response["timestamp"] = datetime.now(timezone.utc).isoformat()
    return jsonify(response), 200


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


def _persist_and_respond(
    result: score_fusion.FusionResult,
    body: AnalyzeRequest,
    transcript: str | None,
):
    request_id = _request_id()
    audio_file_path = body.audio_file_path or f"request://{request_id}"

    try:
        with get_session() as session:
            record_call_log(
                session,
                caller_id=body.caller_id,
                audio_file_path=audio_file_path,
                transcript=transcript,
                spectral_score=result.spectral_score,
                intent_score=result.intent_score,
                final_score=result.final_score,
                risk_label=result.risk_label,
                anomaly_flag=False,
            )
    except Exception:
        current_app.logger.exception(
            "[Analyze] failed to persist call log request_id=%s", request_id
        )
        return jsonify({"error": "Failed to persist analysis result", "request_id": request_id}), 500

    return jsonify(_build_response(result, body)), 200


def _update_user_stats(risk_label: str) -> None:
    user = getattr(g, "current_user", None)
    if user:
        user_store.increment_stats(
            user["id"],
            threat_blocked=(risk_label == "HIGH_RISK"),
        )