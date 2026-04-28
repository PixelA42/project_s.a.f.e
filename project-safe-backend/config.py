"""
Central configuration — every setting loaded from environment variables.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── Flask ──────────────────────────────────────────────────
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-CHANGE-IN-PRODUCTION")
    DEBUG: bool = os.getenv("FLASK_DEBUG", "0") == "1"
    TESTING: bool = False

    # ── Database ───────────────────────────────────────────────
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///project_safe_backend.db")

    # ── CORS ───────────────────────────────────────────────────
    CORS_ORIGINS: str = "*"

    # ── JWT ────────────────────────────────────────────────────
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", SECRET_KEY)
    JWT_EXPIRY_HOURS: int = int(os.getenv("JWT_EXPIRY_HOURS", "24"))

    # ── Deepgram ───────────────────────────────────────────────
    # Real endpoint: https://api.deepgram.com/v1/listen
    DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "")
    DEEPGRAM_ENDPOINT: str = "https://api.deepgram.com/v1/listen"
    DEEPGRAM_PARAMS: dict = {
        "model": "nova-2",
        "language": "en",
        "smart_format": "true",
        "diarize": "true",
        "punctuate": "true",
    }
    USE_REAL_TRANSCRIPTION: bool = os.getenv("USE_REAL_TRANSCRIPTION", "0") == "1"

    # ── OpenAI ─────────────────────────────────────────────────
    # Real endpoint: https://api.openai.com/v1/chat/completions
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_ENDPOINT: str = "https://api.openai.com/v1/chat/completions"
    USE_REAL_INTENT_MODEL: bool = os.getenv("USE_REAL_INTENT_MODEL", "0") == "1"

    # ── Teammate ML models ─────────────────────────────────────
    SPECTRAL_MODEL_PATH: str = os.getenv(
        "SPECTRAL_MODEL_PATH", "app/models/weights/spectral_model.pkl"
    )
    INTENT_MODEL_PATH: str = os.getenv(
        "INTENT_MODEL_PATH", "app/models/weights/intent_model.pkl"
    )
    USE_REAL_SPECTRAL_MODEL: bool = os.getenv("USE_REAL_SPECTRAL_MODEL", "0") == "1"

    # ── Score fusion ───────────────────────────────────────────
    SPECTRAL_WEIGHT: float = float(os.getenv("SPECTRAL_WEIGHT", "0.6"))
    INTENT_WEIGHT: float = float(os.getenv("INTENT_WEIGHT", "0.4"))
    HIGH_RISK_THRESHOLD: int = int(os.getenv("HIGH_RISK_THRESHOLD", "70"))
    PRANK_THRESHOLD: int = int(os.getenv("PRANK_THRESHOLD", "35"))

    # ── Rate limiting ──────────────────────────────────────────
    RATELIMIT_DEFAULT: str = os.getenv("RATELIMIT_DEFAULT", "60 per minute")
    RATELIMIT_ANALYZE: str = os.getenv("RATELIMIT_ANALYZE", "20 per minute")
    RATELIMIT_AUTH: str = os.getenv("RATELIMIT_AUTH", "10 per minute")


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
    USE_REAL_TRANSCRIPTION = False
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///:memory:")


config_map = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}


def get_config() -> type[Config]:
    env = os.getenv("FLASK_ENV", "development")
    return config_map.get(env, DevelopmentConfig)