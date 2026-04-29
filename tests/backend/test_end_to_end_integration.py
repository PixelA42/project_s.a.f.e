from __future__ import annotations

import io
import math
import struct
import sys
import time
import wave
from pathlib import Path
from uuid import UUID

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "project-safe-backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import create_app
from Database.models import CallLog, configure_database, get_session, reset_database


@pytest.fixture(autouse=True)
def sqlite_database():
    configure_database("sqlite:///:memory:")
    yield
    reset_database()


@pytest.fixture()
def app_client():
    app = create_app()
    configure_database("sqlite:///:memory:")
    app.config.update(TESTING=True)
    return app.test_client()


def make_wav_bytes(duration_seconds: float = 1.0, sample_rate: int = 8000) -> bytes:
    buffer = io.BytesIO()
    total_samples = int(duration_seconds * sample_rate)
    amplitude = 1000
    frequency = 220.0

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        frames = bytearray()
        for idx in range(total_samples):
            sample = int(amplitude * math.sin(2.0 * math.pi * frequency * idx / sample_rate))
            frames.extend(struct.pack("<h", sample))
        wav_file.writeframes(bytes(frames))

    return buffer.getvalue()


def test_full_pipeline_audio_to_db_round_trip(app_client):
    audio_bytes = make_wav_bytes(duration_seconds=1.0)

    spectral_resp = app_client.post(
        "/api/analyze-audio",
        data={"file": (io.BytesIO(audio_bytes), "sample.wav")},
        content_type="multipart/form-data",
    )
    assert spectral_resp.status_code == 200
    spectral_payload = spectral_resp.get_json()
    assert "spectral_score" in spectral_payload

    intent_resp = app_client.post(
        "/api/analyze-intent",
        data={"file": (io.BytesIO(audio_bytes), "sample.wav")},
        content_type="multipart/form-data",
    )
    assert intent_resp.status_code == 200
    intent_payload = intent_resp.get_json()
    assert "intent_score" in intent_payload
    assert "transcript" in intent_payload

    evaluate_resp = app_client.post(
        "/api/evaluate-risk",
        json={
            "spectral_score": spectral_payload["spectral_score"],
            "intent_score": intent_payload["intent_score"],
            "caller_id": "E2E Caller",
            "audio_file_path": "/tmp/e2e_sample.wav",
            "anomaly_flag": bool(
                spectral_payload.get("anomaly_flag", False) or intent_payload.get("anomaly_flag", False)
            ),
        },
    )
    assert evaluate_resp.status_code == 200
    evaluate_payload = evaluate_resp.get_json()
    assert "id" in evaluate_payload
    assert evaluate_payload["spectral_score"] == round(float(spectral_payload["spectral_score"]), 1)
    assert evaluate_payload["intent_score"] == round(float(intent_payload["intent_score"]), 1)

    call_log_id = UUID(evaluate_payload["id"])
    with get_session() as session:
        record = session.get(CallLog, call_log_id)

    assert record is not None
    assert record.id == call_log_id
    assert record.caller_id == "E2E Caller"
    assert record.audio_file_path == "/tmp/e2e_sample.wav"
    assert record.spectral_score == evaluate_payload["spectral_score"]
    assert record.intent_score == evaluate_payload["intent_score"]
    assert record.final_score == evaluate_payload["final_score"]
    assert record.risk_label.value == evaluate_payload["risk_label"]


def test_endpoint_timing_sla_for_60_second_audio(app_client):
    audio_bytes = make_wav_bytes(duration_seconds=60.0)

    start = time.perf_counter()
    spectral_resp = app_client.post(
        "/api/analyze-audio",
        data={"file": (io.BytesIO(audio_bytes), "long.wav")},
        content_type="multipart/form-data",
    )
    spectral_elapsed = time.perf_counter() - start
    assert spectral_resp.status_code == 200
    assert spectral_elapsed <= 5.0
    spectral_score = spectral_resp.get_json()["spectral_score"]

    start = time.perf_counter()
    intent_resp = app_client.post(
        "/api/analyze-intent",
        data={"file": (io.BytesIO(audio_bytes), "long.wav")},
        content_type="multipart/form-data",
    )
    intent_elapsed = time.perf_counter() - start
    assert intent_resp.status_code == 200
    assert intent_elapsed <= 10.0
    intent_score = intent_resp.get_json()["intent_score"]

    start = time.perf_counter()
    evaluate_resp = app_client.post(
        "/api/evaluate-risk",
        json={
            "spectral_score": spectral_score,
            "intent_score": intent_score,
            "caller_id": "Timing Caller",
            "audio_file_path": "/tmp/long.wav",
        },
    )
    evaluate_elapsed = time.perf_counter() - start
    assert evaluate_resp.status_code == 200
    assert evaluate_elapsed <= 15.0
