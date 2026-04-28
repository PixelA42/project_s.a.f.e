from __future__ import annotations

import io
import math
import struct
import sys
import wave
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "project-safe-backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import create_app
from Database.models import reset_database


@pytest.fixture(autouse=True)
def backend_env(monkeypatch):
    monkeypatch.setenv("FLASK_ENV", "testing")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    reset_database()
    yield
    reset_database()


@pytest.fixture()
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def make_wav_bytes(duration_seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    total_samples = int(duration_seconds * sample_rate)
    amplitude = 1000
    frequency = 440.0

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        frames = bytearray()
        for index in range(total_samples):
            sample = int(amplitude * math.sin(2.0 * math.pi * frequency * index / sample_rate))
            frames.extend(struct.pack("<h", sample))
        wav_file.writeframes(bytes(frames))

    return buffer.getvalue()


def test_analyze_audio_accepts_wav_upload(client):
    audio_bytes = make_wav_bytes()

    response = client.post(
        "/api/analyze-audio",
        data={"file": (io.BytesIO(audio_bytes), "sample.wav")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert "spectral_score" in payload
    assert "anomaly_flag" in payload
    assert 0 <= payload["spectral_score"] <= 100


def test_analyze_audio_accepts_mp3_extension(client):
    audio_bytes = make_wav_bytes()

    response = client.post(
        "/api/analyze-audio",
        data={"file": (io.BytesIO(audio_bytes), "sample.mp3")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert "spectral_score" in payload
    assert "anomaly_flag" in payload


def test_analyze_audio_rejects_missing_file(client):
    response = client.post("/api/analyze-audio", data={}, content_type="multipart/form-data")

    assert response.status_code == 422
    payload = response.get_json()
    assert payload["error"] == "validation_error"
    assert payload["field"] == "file"


def test_analyze_audio_rejects_oversized_file(client):
    oversized = b"0" * (25 * 1024 * 1024 + 1)

    response = client.post(
        "/api/analyze-audio",
        data={"file": (io.BytesIO(oversized), "too_large.wav")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 413
    payload = response.get_json()
    assert payload["error"] == "payload_too_large"


def test_analyze_intent_returns_transcript_fields(client):
    audio_bytes = make_wav_bytes()

    response = client.post(
        "/api/analyze-intent",
        data={"file": (io.BytesIO(audio_bytes), "sample.wav")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert "intent_score" in payload
    assert "transcript" in payload
    assert "no_speech_detected" in payload
    assert "anomaly_flag" in payload
    assert 0 <= payload["intent_score"] <= 100


def test_evaluate_risk_returns_fused_scores(client):
    response = client.post(
        "/api/evaluate-risk",
        json={
            "spectral_score": 92,
            "intent_score": 88,
            "caller_id": "caller-1",
            "audio_file_path": "/tmp/sample.wav",
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["risk_label"] == "HIGH_RISK"
    assert payload["spectral_score"] == 92
    assert payload["intent_score"] == 88
    assert payload["final_score"] == 90.8


def test_evaluate_risk_rejects_invalid_payload(client):
    response = client.post(
        "/api/evaluate-risk",
        json={
            "spectral_score": 120,
            "intent_score": -1,
        },
    )

    assert response.status_code == 422
    payload = response.get_json()
    assert payload["error"] == "validation_error"
    assert payload["field"] == "body"
