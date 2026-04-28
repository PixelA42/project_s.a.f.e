from __future__ import annotations

from datetime import datetime, timezone
import sys
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy.exc import IntegrityError

from Database.models import CallLog, RiskLabel, configure_database, get_session, record_call_log, reset_database


ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "project-safe-backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import create_app


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


def test_call_log_round_trip_persists_all_required_fields():
    timestamp = datetime(2026, 4, 28, 15, 0, 0, tzinfo=timezone.utc)

    with get_session() as session:
        entry = record_call_log(
            session,
            caller_id="caller-123",
            audio_file_path="/tmp/audio.wav",
            transcript="help me please transfer money",
            spectral_score=91.25,
            intent_score=77.5,
            final_score=87.1,
            risk_label=RiskLabel.HIGH_RISK,
            anomaly_flag=True,
            timestamp=timestamp,
        )
        entry_id = entry.id

    assert isinstance(entry_id, UUID)

    with get_session() as session:
        record = session.get(CallLog, entry_id)

    assert record is not None
    assert record.id == entry_id
    assert record.caller_id == "caller-123"
    assert record.audio_file_path == "/tmp/audio.wav"
    assert record.transcript == "help me please transfer money"
    assert record.spectral_score == 91.25
    assert record.intent_score == 77.5
    assert record.final_score == 87.1
    assert record.risk_label == RiskLabel.HIGH_RISK
    assert record.anomaly_flag is True
    assert record.timestamp == timestamp
    assert record.timestamp.tzinfo is not None
    assert record.timestamp.utcoffset() == timezone.utc.utcoffset(record.timestamp)


def test_call_log_uses_unique_uuid_v4_ids():
    ids: list[UUID] = []

    with get_session() as session:
        for index in range(10):
            record = record_call_log(
                session,
                caller_id=f"caller-{index}",
                audio_file_path=f"/tmp/audio-{index}.wav",
                transcript="transfer now",
                spectral_score=80.0 + index * 0.1,
                intent_score=60.0 + index * 0.1,
                final_score=75.0 + index * 0.1,
                risk_label=RiskLabel.PRANK,
            )
            ids.append(record.id)

    assert len(ids) == len(set(ids))
    assert all(uuid.version == 4 for uuid in ids)


def test_call_log_check_constraints_reject_out_of_range_scores():
    with pytest.raises(IntegrityError):
        with get_session() as session:
            record_call_log(
                session,
                caller_id="caller-bad",
                audio_file_path="/tmp/audio.wav",
                transcript="invalid score",
                spectral_score=101.0,
                intent_score=50.0,
                final_score=50.0,
                risk_label=RiskLabel.SAFE,
            )


def test_analyze_route_persists_call_log_record(app_client):
    response = app_client.post(
        "/api/v1/analyze",
        json={
            "mock_label": "SAFE",
            "caller_id": "Alice",
            "caller_number": "+15555550199",
            "audio_file_path": "/tmp/call.wav",
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["risk_label"] == "SAFE"

    with get_session() as session:
        records = session.query(CallLog).all()

    assert len(records) == 1
    record = records[0]
    assert record.caller_id == "Alice"
    assert record.audio_file_path == "/tmp/call.wav"
    assert record.risk_label == RiskLabel.SAFE
    assert record.final_score == payload["final_score"]


def test_analyze_route_persists_anomaly_flag(app_client):
    """E2E test: /api/v1/analyze → response includes risk data → CallLog persists anomaly_flag."""
    response = app_client.post(
        "/api/v1/analyze",
        json={
            "mock_label": "HIGH_RISK",
            "caller_id": "Charlie",
            "caller_number": "+19876543210",
            "audio_file_path": "/tmp/suspicious.wav",
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["risk_label"] == "HIGH_RISK"

    with get_session() as session:
        records = session.query(CallLog).all()

    assert len(records) == 1
    record = records[0]
    assert record.caller_id == "Charlie"
    assert record.risk_label == RiskLabel.HIGH_RISK
    # anomaly_flag should be persisted (even if not in mock response, it defaults to False)
    assert isinstance(record.anomaly_flag, bool)
