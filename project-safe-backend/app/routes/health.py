"""
GET /api/v1/health — system status check.
Used by the frontend to verify the backend is reachable.
"""
from datetime import datetime, timezone
from flask import Blueprint, jsonify, current_app

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health():
    spectral_loaded = getattr(current_app, "spectral_model", None) is not None
    intent_loaded = getattr(current_app, "intent_model", None) is not None

    return jsonify({
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": {
            "spectral": "real" if spectral_loaded else "mock",
            "intent": "real" if intent_loaded else "mock",
        },
        "config": {
            "spectral_weight": current_app.config.get("SPECTRAL_WEIGHT"),
            "intent_weight": current_app.config.get("INTENT_WEIGHT"),
            "high_risk_threshold": current_app.config.get("HIGH_RISK_THRESHOLD"),
            "prank_threshold": current_app.config.get("PRANK_THRESHOLD"),
        },
    }), 200