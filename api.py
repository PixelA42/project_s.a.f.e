from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Project SAFE API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024
ALLOWED_EXTENSIONS = {"mp3", "wav", "flac", "m4a"}
RISK_LABELS = {"SAFE", "PRANK", "HIGH_RISK"}

SCENARIO_RESPONSES: dict[str, dict[str, Any]] = {
    "SAFE": {
        "risk_label": "SAFE",
        "final_score": 12,
        "spectral_score": 8,
        "intent_score": 5,
        "caller_id": "John Doe",
        "caller_number": "+1 (555) 234-5678",
    },
    "PRANK": {
        "risk_label": "PRANK",
        "final_score": 55,
        "spectral_score": 78,
        "intent_score": 12,
        "caller_id": "Unknown Caller",
        "caller_number": "+1 (555) 867-5309",
    },
    "HIGH_RISK": {
        "risk_label": "HIGH_RISK",
        "final_score": 92,
        "spectral_score": 94,
        "intent_score": 88,
        "caller_id": "Unknown Caller",
        "caller_number": "+1 (800) 555-0199",
    },
}


class EvaluateRiskRequest(BaseModel):
    scenario: str = Field(..., description="One of SAFE, PRANK, HIGH_RISK")


class ScoreResponse(BaseModel):
    risk_label: str
    final_score: int
    spectral_score: int
    intent_score: int
    caller_id: str | None = None
    caller_number: str | None = None
    timestamp: str


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "timestamp": _iso_now()}


@app.post("/api/evaluate-risk", response_model=ScoreResponse)
def evaluate_risk(payload: EvaluateRiskRequest) -> dict[str, Any]:
    scenario = payload.scenario.strip().upper()
    if scenario not in RISK_LABELS:
        raise HTTPException(
            status_code=400,
            detail="scenario must be one of SAFE, PRANK, or HIGH_RISK.",
        )

    return {
        **SCENARIO_RESPONSES[scenario],
        "timestamp": _iso_now(),
    }


@app.post("/detect-audio")
async def detect_audio(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = file.filename or ""
    if not filename:
        raise HTTPException(status_code=400, detail="Bad Request: No file selected")

    if not _allowed_file(filename):
        raise HTTPException(
            status_code=415,
            detail="Unsupported Media Type: Only .mp3, .wav, .flac, or .m4a allowed",
        )

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Payload Too Large: File exceeds the 5MB limit.",
        )

    return {
        "status": "success",
        "message": "Audio received and validated successfully.",
        "filename": filename,
        "prediction": "AI_Generated",
        "confidence_score": 0.94,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=5000, reload=False)
