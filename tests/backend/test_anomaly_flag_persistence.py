from pathlib import Path
import sys
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "project-safe-backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import create_app
from Database.models import configure_database, get_session, CallLog, reset_database


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


def test_evaluate_risk_persists_anomaly_flag(app_client):
    payload = {
        "spectral_score": 85.0,
        "intent_score": 70.0,
        "caller_id": "Bob",
        "audio_file_path": "/tmp/call.wav",
        "anomaly_flag": True,
    }

    resp = app_client.post("/api/evaluate-risk", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()

    with get_session() as session:
        records = session.query(CallLog).all()

    assert len(records) == 1
    assert records[0].anomaly_flag is True
