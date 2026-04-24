from datetime import datetime, timezone
from flask import Blueprint, jsonify, current_app
from flasgger import swag_from

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health():
    """
    Server and engine health check.
    ---
    tags:
      - Health
    summary: Check API status
    description: Returns the current status of the server, both scoring engines, and external API configuration.
    responses:
      200:
        description: Server is healthy
        schema:
          type: object
          properties:
            status:
              type: string
              example: ok
            timestamp:
              type: string
              example: "2026-04-24T06:58:35.004323+00:00"
            engines:
              type: object
              properties:
                spectral:
                  type: string
                  example: librosa_local
                intent:
                  type: string
                  example: keyword_classifier
            external_apis:
              type: object
              properties:
                deepgram:
                  type: string
                  example: not_configured
                openai:
                  type: string
                  example: not_configured
            config:
              type: object
              properties:
                spectral_weight:
                  type: number
                  example: 0.6
                intent_weight:
                  type: number
                  example: 0.4
                high_risk_threshold:
                  type: integer
                  example: 70
                prank_threshold:
                  type: integer
                  example: 35
    """
    spectral_loaded = getattr(current_app, "spectral_model", None) is not None
    intent_loaded   = getattr(current_app, "intent_model", None) is not None

    deepgram_configured = bool(
        current_app.config.get("DEEPGRAM_API_KEY") and
        current_app.config["DEEPGRAM_API_KEY"] not in ("your_deepgram_api_key_here", "not_configured_yet", "")
    )
    openai_configured = bool(
        current_app.config.get("OPENAI_API_KEY") and
        current_app.config["OPENAI_API_KEY"] not in ("your_openai_api_key_here", "not_configured_yet", "")
    )

    return jsonify({
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "engines": {
            "spectral": "teammate_model" if spectral_loaded else "librosa_local",
            "intent":   "teammate_model" if intent_loaded else (
                "openai_api" if openai_configured else "keyword_classifier"
            ),
        },
        "external_apis": {
            "deepgram": "configured" if deepgram_configured else "not_configured",
            "openai":   "configured" if openai_configured   else "not_configured",
        },
        "config": {
            "spectral_weight":     current_app.config.get("SPECTRAL_WEIGHT"),
            "intent_weight":       current_app.config.get("INTENT_WEIGHT"),
            "high_risk_threshold": current_app.config.get("HIGH_RISK_THRESHOLD"),
            "prank_threshold":     current_app.config.get("PRANK_THRESHOLD"),
        },
    }), 200