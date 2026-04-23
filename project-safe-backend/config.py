"""
Central configuration — every setting comes from environment variables.
Add new keys here before using them anywhere in the app.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env into os.environ


class Config:
    # ── Flask ──────────────────────────────────────────────────
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-change-in-production")
    DEBUG: bool = os.getenv("FLASK_DEBUG", "0") == "1"
    TESTING: bool = False

    # ── CORS ───────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = [
        o.strip()
        for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
    ]

    # ── Spectral engine ────────────────────────────────────────
    DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "")
    ASSEMBLYAI_API_KEY: str = os.getenv("ASSEMBLYAI_API_KEY", "")
    SPECTRAL_MODEL_PATH: str = os.getenv(
        "SPECTRAL_MODEL_PATH", "app/models/weights/spectral_model.pkl"
    )
    USE_REAL_SPECTRAL_MODEL: bool = (
        os.getenv("USE_REAL_SPECTRAL_MODEL", "0") == "1"
    )
    USE_REAL_TRANSCRIPTION: bool = (
        os.getenv("USE_REAL_TRANSCRIPTION", "0") == "1"
    )

    # ── Intent engine ──────────────────────────────────────────
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    INTENT_MODEL_PATH: str = os.getenv(
        "INTENT_MODEL_PATH", "app/models/weights/intent_model.pkl"
    )
    USE_REAL_INTENT_MODEL: bool = (
        os.getenv("USE_REAL_INTENT_MODEL", "0") == "1"
    )

    # ── Score fusion ───────────────────────────────────────────
    SPECTRAL_WEIGHT: float = float(os.getenv("SPECTRAL_WEIGHT", "0.6"))
    INTENT_WEIGHT: float = float(os.getenv("INTENT_WEIGHT", "0.4"))
    HIGH_RISK_THRESHOLD: int = int(os.getenv("HIGH_RISK_THRESHOLD", "70"))
    PRANK_THRESHOLD: int = int(os.getenv("PRANK_THRESHOLD", "35"))

    # ── Rate limiting ──────────────────────────────────────────
    RATELIMIT_DEFAULT: str = os.getenv("RATELIMIT_DEFAULT", "60 per minute")
    RATELIMIT_ANALYZE: str = os.getenv("RATELIMIT_ANALYZE", "20 per minute")


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False
    TESTING = False


class TestingConfig(Config):
    TESTING = True
    DEBUG = True
    USE_REAL_SPECTRAL_MODEL = False
    USE_REAL_INTENT_MODEL = False


config_map = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}


def get_config() -> type[Config]:
    env = os.getenv("FLASK_ENV", "development")
    return config_map.get(env, DevelopmentConfig)