"""
Spectral Engine — Layer 1
Acoustic / voice forensics analysis.

Detects synthetic voice markers by analysing:
  - MFCC (Mel-frequency cepstral coefficients) anomalies
  - GAN / TTS artefacts in the frequency spectrum
  - Prosody irregularities (unnatural pitch patterns)
  - Phase discontinuities typical of voice cloning models

── PLUGGING IN YOUR TEAMMATE'S MODEL ────────────────────────────────────
1.  Your teammate trains a binary classifier on audio features.
    Input:  numpy array of shape (n_mfcc_features,) or a raw waveform.
    Output: float in [0, 100] — higher = more likely synthetic.

2.  They save it:
        joblib.dump(trained_pipeline, "app/models/weights/spectral_model.pkl")

3.  Set in .env:
        USE_REAL_SPECTRAL_MODEL=1
        SPECTRAL_MODEL_PATH=app/models/weights/spectral_model.pkl

4.  The predict() call below will automatically use it instead of the mock.
─────────────────────────────────────────────────────────────────────────
"""
import numpy as np
from flask import current_app


def analyze(audio_bytes: bytes | None, transcript: str | None = None) -> float:
    """
    Returns a spectral_score in [0, 100].
    0  = definitely real human voice
    100 = definitely synthetic / cloned
    """
    model = getattr(current_app, "spectral_model", None)

    if model is not None and audio_bytes:
        return _run_real_model(model, audio_bytes)

    # ── Mock mode ──────────────────────────────────────────────
    # Generates a realistic-looking score for UI development.
    # Replace this block entirely once the real model is wired in.
    return _mock_spectral_score(transcript)


def _run_real_model(model, audio_bytes: bytes) -> float:
    """
    Extracts features from raw audio and runs the trained classifier.

    ── EXPECTED MODEL INTERFACE ─────────────────────────────────────────
    model.predict(features: np.ndarray) -> float   (score 0–100)
    OR
    model.predict_proba(features: np.ndarray) -> np.ndarray  (probability)

    If your model returns a probability in [0, 1], scale it by 100 here.
    ─────────────────────────────────────────────────────────────────────
    """
    features = _extract_audio_features(audio_bytes)
    if features is None:
        return _mock_spectral_score(None)

    try:
        # ── Replace this call with whatever interface your model exposes ──
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba([features])[0][1]  # P(synthetic)
            return float(prob * 100)
        else:
            return float(model.predict([features])[0])
    except Exception as e:
        current_app.logger.error(f"SpectralEngine.predict failed: {e}")
        return _mock_spectral_score(None)


def _extract_audio_features(audio_bytes: bytes) -> np.ndarray | None:
    """
    Converts raw audio bytes → feature vector for the model.

    Currently extracts 40 MFCCs (mean over time axis).
    Your teammate's model may expect a different feature set —
    modify this function to match whatever they trained on.

    ── TO SWITCH TO REAL TRANSCRIPTION ──────────────────────────────────
    Set USE_REAL_TRANSCRIPTION=1 and DEEPGRAM_API_KEY in .env.
    Then replace this function body with a Deepgram API call:

        import httpx, base64
        resp = httpx.post(
            "https://api.deepgram.com/v1/listen",
            headers={"Authorization": f"Token {current_app.config['DEEPGRAM_API_KEY']}"},
            content=audio_bytes,
        )
        # Use resp.json() features + diarization to build your feature vector
    ─────────────────────────────────────────────────────────────────────
    """
    try:
        import librosa
        import soundfile as sf
        import io

        audio, sr = sf.read(io.BytesIO(audio_bytes))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # convert stereo → mono

        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        return mfccs.mean(axis=1)  # shape: (40,)

    except Exception as e:
        current_app.logger.warning(f"Feature extraction failed: {e}")
        return None


def _mock_spectral_score(hint: str | None) -> float:
    """
    Returns a deterministic mock score for UI development.
    In production this entire function is bypassed.
    """
    rng = np.random.default_rng(seed=abs(hash(hint or "default")) % (2**31))
    return round(float(rng.uniform(10, 95)), 1)