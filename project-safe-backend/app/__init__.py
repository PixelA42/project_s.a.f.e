"""
Flask application factory.
Creates the app, registers blueprints, sets up CORS and rate limiting.
"""
from flask import Flask
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from config import get_config
from app.models.loaders import load_all_models
from app.utils.logger import setup_logger

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[],
)


def create_app() -> Flask:
    app = Flask(__name__)
    cfg = get_config()
    app.config.from_object(cfg)

    # ── Logging ────────────────────────────────────────────────
    setup_logger(app)

    # ── CORS ───────────────────────────────────────────────────
    CORS(
        app,
        resources={r"/api/*": {"origins": cfg.CORS_ORIGINS}},
        supports_credentials=False,
    )

    # ── Rate limiting ──────────────────────────────────────────
    limiter.init_app(app)

    # ── ML models (loaded once at startup) ────────────────────
    load_all_models(app)

    # ── Blueprints ─────────────────────────────────────────────
    from app.routes.analyze import analyze_bp
    from app.routes.health import health_bp

    app.register_blueprint(analyze_bp, url_prefix="/api/v1")
    app.register_blueprint(health_bp, url_prefix="/api/v1")

    # ── Global error handlers ──────────────────────────────────
    @app.errorhandler(404)
    def not_found(_e):
        return {"error": "endpoint not found"}, 404

    @app.errorhandler(429)
    def rate_limited(_e):
        return {"error": "rate limit exceeded — slow down"}, 429

    @app.errorhandler(500)
    def server_error(_e):
        return {"error": "internal server error"}, 500

    return app