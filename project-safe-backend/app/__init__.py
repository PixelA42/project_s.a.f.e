"""
Flask application factory.
"""
import uuid

from flask import Flask, jsonify, g, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flasgger import Swagger

from config import get_config
from app.database import configure_database
from app.models.loaders import load_all_models
from app.utils.logger import setup_logger

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[],
)

# ── Swagger / OpenAPI config ───────────────────────────────────────────
SWAGGER_CONFIG = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec",
            "route": "/api/v1/docs/apispec.json",
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/api/v1/docs",
}

SWAGGER_TEMPLATE = {
    "swagger": "2.0",
    "info": {
        "title": "Project S.A.F.E. API",
        "description": (
            "**Synthetic Audio Fraud Engine** — dual-layer AI voice scam detection.\n\n"
            "Two independent scoring layers:\n"
            "- **Spectral Engine** — acoustic/MFCC analysis via librosa\n"
            "- **Intent Engine** — NLP coercion detection via keyword classifier or OpenAI\n\n"
            "Scores are fused into a final `risk_label`: `HIGH_RISK`, `PRANK`, or `SAFE`."
        ),
        "version": "1.0.0",
        "contact": {
            "name": "Project S.A.F.E. Team",
        },
    },
    "basePath": "/",
    "schemes": ["http", "https"],
    "securityDefinitions": {
        "BearerAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "Authorization",
            "description": "Enter: **Bearer &lt;your_token&gt;**",
        }
    },
    "tags": [
        {"name": "Health",  "description": "Server and engine status"},
        {"name": "Analyze", "description": "Call risk analysis"},
        {"name": "Auth",    "description": "User registration and login"},
    ],
    "consumes": ["application/json"],
    "produces": ["application/json"],
}


def create_app() -> Flask:
    app = Flask(__name__)
    cfg = get_config()
    app.config.from_object(cfg)

    setup_logger(app)

    # ── CORS — allow all origins ───────────────────────────────
    CORS(
        app,
        resources={r"/api/*": {"origins": "*"}},
        supports_credentials=False,
    )

    limiter.init_app(app)
    load_all_models(app)
    configure_database(app.config.get("DATABASE_URL"))

    # ── Swagger UI ─────────────────────────────────────────────
    Swagger(app, config=SWAGGER_CONFIG, template=SWAGGER_TEMPLATE)

    @app.before_request
    def assign_request_id():
        g.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

    # ── Blueprints ─────────────────────────────────────────────
    from app.routes.analyze import analyze_bp, analyze_audio, analyze_intent, evaluate_risk
    from app.routes.health import health_bp
    from app.routes.auth import auth_bp

    app.register_blueprint(analyze_bp, url_prefix="/api/v1")
    app.register_blueprint(health_bp,  url_prefix="/api/v1")
    app.register_blueprint(auth_bp,    url_prefix="/api/v1/auth")

    app.add_url_rule("/api/analyze-audio", endpoint="api_analyze_audio", view_func=analyze_audio, methods=["POST"])
    app.add_url_rule("/api/analyze-intent", endpoint="api_analyze_intent", view_func=analyze_intent, methods=["POST"])
    app.add_url_rule("/api/evaluate-risk", endpoint="api_evaluate_risk", view_func=evaluate_risk, methods=["POST"])

    # ── Global error handlers ──────────────────────────────────
    @app.errorhandler(404)
    def not_found(_e):
        return jsonify({"error": "endpoint not found"}), 404

    @app.errorhandler(405)
    def method_not_allowed(_e):
        return jsonify({"error": "method not allowed"}), 405

    @app.errorhandler(429)
    def rate_limited(_e):
        return jsonify({"error": "rate limit exceeded — slow down"}), 429

    @app.errorhandler(500)
    def server_error(e):
        app.logger.exception("Unhandled server error")
        return jsonify({"error": "internal server error"}), 500

    return app