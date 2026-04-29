"""Project S.A.F.E. — FastAPI backend.

Endpoints
---------
GET  /api/health            — liveness probe
POST /api/evaluate-risk     — mock scenario-based risk evaluation (UI demo)
POST /detect-audio          — real hybrid inference (supervised + unsupervised)

The /detect-audio endpoint implements the full hybrid pipeline:
  1. Supervised PyTorch spectral classifier  →  positive_probability
  2. Unsupervised anomaly detector           →  reconstruction_error, isolation_score
  3. Smart uncertainty routing               →  final routing decision

If the supervised model outputs a low-confidence score (0.35–0.65), the
unsupervised anomaly score is used to decide whether to flag the file for
manual review or pass it as HUMAN.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import SETTINGS

app = FastAPI(title="Project SAFE API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PATHS = SETTINGS.paths
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024   # 25 MB per spec
ALLOWED_EXTENSIONS = {"mp3", "wav", "flac", "m4a"}
RISK_LABELS = {"SAFE", "PRANK", "HIGH_RISK"}

# Uncertainty zone boundaries (must match quick_predict.py)
UNCERTAINTY_LOWER: float = 0.35
UNCERTAINTY_UPPER: float = 0.65

# ---------------------------------------------------------------------------
# Lazy-loaded ML singletons (loaded once on first request, not at import time)
# ---------------------------------------------------------------------------
_supervised_model = None
_unsupervised_detector = None
_uncertainty_queue = None


def _get_supervised_model():
    global _supervised_model
    if _supervised_model is None:
        try:
            import torch
            from coreML.torch_inference import load_torch_spectral_model
            _supervised_model = load_torch_spectral_model(
                PATHS.supervised_torch_weights_path, device="cpu"
            )
        except Exception as exc:
            print(f"[WARN] Could not load supervised model: {exc}")
    return _supervised_model


def _get_unsupervised_detector():
    global _unsupervised_detector
    if _unsupervised_detector is None:
        try:
            from train_unsupervised import UnsupervisedAnomalyDetector
            _unsupervised_detector = UnsupervisedAnomalyDetector(device="cpu")
        except Exception as exc:
            print(f"[WARN] Could not load unsupervised detector: {exc}")
    return _unsupervised_detector


def _get_uncertainty_queue():
    global _uncertainty_queue
    if _uncertainty_queue is None:
        from coreML.uncertainty_queue import UncertaintyQueue
        _uncertainty_queue = UncertaintyQueue(
            queue_dir=PATHS.outputs_dir / "uncertainty_queue",
            lower_bound=UNCERTAINTY_LOWER,
            upper_bound=UNCERTAINTY_UPPER,
        )
    return _uncertainty_queue


# ---------------------------------------------------------------------------
# Mock scenario responses (used by /api/evaluate-risk for UI demo)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

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


class HybridDetectionResponse(BaseModel):
    """Response schema for the real hybrid inference endpoint."""

    # Core result
    prediction: str = Field(..., description="AI or HUMAN")
    routing_decision: str = Field(..., description="Final routing: AI, HUMAN, UNCERTAIN, UNCERTAIN_ANOMALY")
    routing_reason: str

    # Supervised scores
    supervised_probability: float = Field(..., ge=0.0, le=1.0)
    supervised_label: str
    is_uncertain: bool

    # Unsupervised scores
    reconstruction_error: float
    isolation_score: float
    anomaly_flag: bool
    ae_anomaly: bool
    if_anomaly: bool
    unsupervised_ready: bool

    # Metadata
    filename: str
    timestamp: str
    queue_item_id: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _audio_to_temp_spectrogram(audio_path: Path) -> Path:
    """Convert audio to a temporary spectrogram PNG for the unsupervised model."""
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import numpy as np

    _AUDIO = SETTINGS.audio
    signal, sr = librosa.load(
        str(audio_path),
        sr=_AUDIO.sample_rate,
        mono=True,
        duration=_AUDIO.clip_duration_seconds,
    )
    mel = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_mels=_AUDIO.n_mels, fmax=_AUDIO.mel_fmax
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    fig = plt.figure(
        figsize=(_AUDIO.figure_size_inches, _AUDIO.figure_size_inches),
        dpi=_AUDIO.spectrogram_dpi,
    )
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.axis("off")
    librosa.display.specshow(mel_db, sr=sr, cmap=_AUDIO.colormap, ax=ax)
    plt.savefig(tmp.name, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "timestamp": _iso_now()}


@app.post("/api/evaluate-risk", response_model=ScoreResponse)
def evaluate_risk(payload: EvaluateRiskRequest) -> dict[str, Any]:
    """Mock scenario-based risk evaluation for UI demo purposes."""
    scenario = payload.scenario.strip().upper()
    if scenario not in RISK_LABELS:
        raise HTTPException(
            status_code=400,
            detail="scenario must be one of SAFE, PRANK, or HIGH_RISK.",
        )
    return {**SCENARIO_RESPONSES[scenario], "timestamp": _iso_now()}


@app.post("/detect-audio", response_model=HybridDetectionResponse)
async def detect_audio(file: UploadFile = File(...)) -> dict[str, Any]:
    """Hybrid supervised + unsupervised audio deepfake detection.

    Pipeline
    --------
    1. Validate and save the uploaded audio to a temp file.
    2. Run the supervised PyTorch spectral classifier.
    3. Run the unsupervised anomaly detector (Autoencoder + Isolation Forest).
    4. Apply smart uncertainty routing:
       - Confident supervised score  →  use supervised label directly.
       - Uncertain score [0.35–0.65] →  use unsupervised anomaly signal to
         decide between HUMAN, AI fail-closed routing, or manual review.
    5. Return all scores and the routing decision.
    """
    # ------------------------------------------------------------------ #
    # Validate upload                                                      #
    # ------------------------------------------------------------------ #
    filename = file.filename or ""
    if not filename:
        raise HTTPException(status_code=400, detail="No file selected.")

    if not _allowed_file(filename):
        raise HTTPException(
            status_code=415,
            detail="Unsupported format. Only .mp3, .wav, .flac, or .m4a allowed.",
        )

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the 25 MB limit.",
        )

    # ------------------------------------------------------------------ #
    # Save to temp file                                                    #
    # ------------------------------------------------------------------ #
    suffix = Path(filename).suffix.lower()
    tmp_audio = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp_audio.write(file_bytes)
        tmp_audio.close()
        audio_path = Path(tmp_audio.name)

        # ---------------------------------------------------------------- #
        # Step 1 — Supervised classification                                #
        # ---------------------------------------------------------------- #
        supervised_model = _get_supervised_model()
        if supervised_model is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Supervised model not available. "
                    "Ensure models/spectral_model.pt exists."
                ),
            )

        try:
            from coreML.torch_inference import infer_audio_probability
            positive_probability = infer_audio_probability(
                supervised_model, str(audio_path), device="cpu"
            )
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Supervised inference failed: {exc}",
            ) from exc

        supervised_prediction = 1 if positive_probability >= SETTINGS.training.decision_threshold else 0
        supervised_label = "AI" if supervised_prediction == 1 else "HUMAN"
        is_uncertain = UNCERTAINTY_LOWER <= positive_probability <= UNCERTAINTY_UPPER

        # ---------------------------------------------------------------- #
        # Step 2 — Unsupervised anomaly scoring                             #
        # ---------------------------------------------------------------- #
        unsupervised_detector = _get_unsupervised_detector()
        unsupervised_scores: dict[str, Any] = {
            "reconstruction_error": 0.0,
            "isolation_score": 0.0,
            "anomaly_flag": False,
            "ae_anomaly": False,
            "if_anomaly": False,
            "unsupervised_ready": False,
        }

        temp_spec_path: Path | None = None
        if unsupervised_detector is not None and unsupervised_detector.is_ready:
            try:
                temp_spec_path = _audio_to_temp_spectrogram(audio_path)
                unsupervised_scores = unsupervised_detector.score(str(temp_spec_path))
            except Exception as exc:
                print(f"[WARN] Unsupervised scoring failed: {exc}")
            finally:
                if temp_spec_path:
                    try:
                        temp_spec_path.unlink(missing_ok=True)
                    except Exception:
                        pass

        # ---------------------------------------------------------------- #
        # Step 3 — Smart uncertainty routing                                #
        # ---------------------------------------------------------------- #
        uncertainty_queue = _get_uncertainty_queue()
        queue_item_id: str | None = None

        if not is_uncertain:
            routing_decision = supervised_label
            routing_reason = (
                f"Supervised model confident "
                f"(p={positive_probability:.3f}, "
                f"threshold={SETTINGS.training.decision_threshold})"
            )
        else:
            anomaly_flag = unsupervised_scores.get("anomaly_flag", False)
            unsupervised_ready = unsupervised_scores.get("unsupervised_ready", False)

            if not unsupervised_ready:
                routing_decision = "UNCERTAIN"
                routing_reason = (
                    f"Supervised uncertain (p={positive_probability:.3f}) and "
                    "unsupervised model not available — queued for manual review"
                )
            elif anomaly_flag:
                routing_decision = "UNCERTAIN_ANOMALY"
                routing_reason = (
                    f"Supervised uncertain (p={positive_probability:.3f}); "
                    f"unsupervised anomaly detected "
                    f"(recon_err={unsupervised_scores['reconstruction_error']:.4f}, "
                    f"iso_score={unsupervised_scores['isolation_score']:.4f}) "
                    "— flagged for manual review"
                )
            elif supervised_prediction == 1:
                routing_decision = "AI"
                routing_reason = (
                    f"Supervised uncertain but above fraud threshold "
                    f"(p={positive_probability:.3f}, "
                    f"threshold={SETTINGS.training.decision_threshold}); "
                    "unsupervised model sees no anomaly, but the system fails closed as AI"
                )
            else:
                routing_decision = "HUMAN"
                routing_reason = (
                    f"Supervised uncertain (p={positive_probability:.3f}); "
                    "unsupervised model sees no anomaly — classified as HUMAN"
                )

            if (not unsupervised_ready) or anomaly_flag:
                queue_item = uncertainty_queue.add_to_queue(
                    audio_file_path=filename,
                    predicted_probability=positive_probability,
                    predicted_label=supervised_prediction,
                    confidence_score=max(positive_probability, 1.0 - positive_probability),
                    model_name="hybrid_spectral_model",
                    tags=[
                        "uncertainty_zone",
                        "api_routed",
                        "anomaly_detected" if anomaly_flag else "no_anomaly",
                        "unsupervised_used" if unsupervised_ready else "unsupervised_unavailable",
                    ],
                )
                queue_item_id = queue_item.item_id

        # Final human-readable prediction
        prediction = (
            "AI"
            if routing_decision in ("AI", "UNCERTAIN_ANOMALY")
            else "HUMAN"
        )

        return {
            "prediction": prediction,
            "routing_decision": routing_decision,
            "routing_reason": routing_reason,
            "supervised_probability": positive_probability,
            "supervised_label": supervised_label,
            "is_uncertain": is_uncertain,
            "reconstruction_error": unsupervised_scores["reconstruction_error"],
            "isolation_score": unsupervised_scores["isolation_score"],
            "anomaly_flag": unsupervised_scores["anomaly_flag"],
            "ae_anomaly": unsupervised_scores["ae_anomaly"],
            "if_anomaly": unsupervised_scores["if_anomaly"],
            "unsupervised_ready": unsupervised_scores["unsupervised_ready"],
            "filename": filename,
            "timestamp": _iso_now(),
            "queue_item_id": queue_item_id,
        }

    finally:
        # Always clean up the temp audio file
        try:
            os.unlink(tmp_audio.name)
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=5000, reload=False)
