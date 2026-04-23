# Project S.A.F.E. — Spectral Analysis for Fake Audio Detection

**Status:** Pipeline built, API connected, ready for model training.

---

## Overview

Building a system to detect AI-generated voice clones, specifically targeting Indian scam and prank calls. Generic models fail in this context due to local accents, Hinglish, and background noise — so the system is built specifically for the Indian context.

---

## What's Working Right Now

- **Dataset Setup:** Created a local folder structure separating real Indian speech (Project Vaani) from synthetic AI audio (ElevenLabs).
- **Audio Pipeline (`audio_pipeline.py`):** Converts raw `.mp3`/`.wav` files into Mel Spectrogram images.
- **Data Augmentation:** Automatic noise injection and pitch shifting built into the pipeline.
- **Smart-Skip Optimization:** Script only processes new audio files instead of regenerating the whole dataset every run.
- **Backend API (`api.py`):** Flask API connecting frontend to backend. Handles edge cases — blocks non-audio files, caps file size at 5MB, handles bad data, and returns clean JSON responses.

---

## Tech Stack

| Library | Purpose |
|---|---|
| `librosa` | Audio loading and feature extraction |
| `numpy` | Numerical array operations |
| `matplotlib` | Spectrogram image generation |
| `Flask` | REST API backend |
| `pandas` | Dataset mapping via `labels.csv` |
| `scipy` | Used internally by librosa |

---

## Dataset Summary (After Augmentation)

| Metric | Count |
|---|---|
| Total Samples | 490 |
| Human (label = 0) | 275 |
| AI (label = 1) | 215 |
| Original Samples | 98 |
| Augmented Samples | 392 |

**Augmentation Breakdown (98 copies each):**
- None (original)
- Noise injection
- Pitch shift up
- Pitch shift down
- Time stretch

**Output folders:**
- Spectrograms → `spectrograms/`
- Labels → `labels.csv`

**Train/Test Split:** 60% training / 40% testing

---

## Key Concepts

### What is Data Augmentation?

Data augmentation means making slightly altered copies of original data to improve model robustness. It's the equivalent of throwing curveballs at the AI during training.

Techniques used:
- Adding background noise
- Pitch shifting (up and down)
- Time stretching
- Volume control

**Goal:** Prevent overfitting and increase model robustness.

---

### Why Look at Audio Frequencies Instead of Text/NLP?

Scammers say the exact same things a real person would (e.g., *"your UPI failed"*). The words aren't fake — the digital artifacts in the voice are. Additionally, transcribing mixed Hinglish accurately is unreliable. Detecting acoustic anomalies is far more robust.

---

### What is a Spectrogram?

Audio is normally a **1D signal** — a wave moving up and down over time. A spectrogram transforms that into a **2D image**, acting as a visual photograph of sound that shows which frequencies are active at each millisecond.

**How to read a spectrogram:**
- **X-axis (left → right):** Time — left is the start, right is the end of the audio.
- **Y-axis (bottom → top):** Frequency — bottom is deep/bass sounds, top is high-pitched sounds.
- **Color (Magma colormap):** Dark purple/black = silence at that frequency. Bright yellow = loud sound at that frequency.

---

### Why Convert Audio to Spectrograms?

Feeding raw 1D audio directly into a neural network is computationally expensive and inefficient. Converting to a 2D spectrogram image turns an audio problem into an **image classification problem**, allowing a standard CNN to find visual patterns — which it is much better at.

**Key insight:** In AI-generated voice, the spectrogram lacks the natural variations and characteristics present in human speech. The model learns to spot these differences visually.

---

### Why Inject Fake Background Noise?

AI voice generators produce clean, studio-quality audio. Real Indian phone calls almost always have background noise. Without noise injection, the model learns a shortcut: *"Clean audio = AI, Noisy audio = Human"*. Adding synthetic noise forces the model to look for actual AI voice artifacts instead.

---

### What is an API?

**API** stands for **Application Programming Interface**. When a user uploads audio in the frontend, the API carries it to the backend for processing and returns the result.

**How it's built:**
1. Choose a framework (Flask in this case).
2. Define specific routes/endpoints for each action (e.g., `/predict`).

---

### Librosa Library

`librosa` is a powerful Python library built on `scipy` and `numpy`.

**Core function — `librosa.load()`:**
- `y` → NumPy array of the audio signal (1D floating-point time series)
- `sr` → Sample rate — the number of audio samples per second

**Key transform:**
- `librosa.stft()` → Short-Time Fourier Transform (STFT) — converts the audio into frequency data over time, which is then visualized as a spectrogram.

---

## Database — PostgreSQL (pgAdmin 4)

### Table: `call_logs`

| Column | Type | Description |
|---|---|---|
| `id` | UUID | Unique identifier for each call |
| `timestamp` | TIMESTAMPTZ | Exact date and time of the call (timezone-aware) |
| `caller_id` | TEXT | Phone number that made the call |
| `audio_file_path` | TEXT | File path to audio (stored as reference, not raw data) |
| `transcript` | TEXT | Speech-to-text conversion of the call |
| `matched_keywords` | TEXT[] | Array of flagged red-flag words |
| `keyword_count` | INT | Total number of red-flag keywords |
| `urgency_score` | FLOAT | Measures panic/urgency in the call |
| `coercion_score` | FLOAT | Measures how threatening the caller is |
| `spectral_score` | FLOAT | Layer 1 result — how AI-like the voice sounds (0–100) |
| `intent_score` | FLOAT | Layer 2 result — how scam-like the content is (0–100) |
| `final_score` | FLOAT | Fusion formula result (0–100) |
| `risk_label` | ENUM | Human-readable tag: `HIGH_RISK`, `PRANK`, or `SAFE` |
| `anomaly_flag` | BOOLEAN | True if human review is needed |

---

### Database Vocabulary

- **UUID (Universally Unique Identifier):** A unique alphanumeric barcode (e.g., `550e8400-e29b-41d4-a716-446655440000`) — prevents ID collisions even across multiple servers.
- **TIMESTAMPTZ (Timestamp with Time Zone):** Records exact time *and* timezone — critical for security tools where "9:00 AM" means different things in different regions.
- **TEXT[] (Array of Text):** A list of text values, e.g., `["urgent", "bitcoin", "police"]`.
- **FLOAT:** Decimal number — necessary because AI scores are rarely whole numbers (e.g., 80.459).
- **BOOLEAN:** Simple True/False switch (e.g., `anomaly_flag = True`).
- **ENUM (Enumeration):** A strict multiple-choice field. `risk_label` only accepts `HIGH_RISK`, `PRANK`, or `SAFE` — any other value is rejected by the database.

---

### Constraints

- **Score Range Checks:** Enforces that `spectral_score`, `intent_score`, and `final_score` must all fall strictly between 0 and 100.
- **Fusion Formula Check:** Database verifies that `final_score = (0.70 × spectral_score) + (0.30 × intent_score)`.
- **Risk Label Logic Check:** Enforces business rules — e.g., if `final_score > 75` AND `intent_score > 60`, the label *must* be `HIGH_RISK`.

---

### Fusion Formula

```
final_score = (0.70 × spectral_score) + (0.30 × intent_score)
```

**Risk label rules:**
- `final_score > 75` AND `intent_score > 60` → **HIGH_RISK**
- Otherwise → **PRANK** or **SAFE** based on thresholds

---

### Walkthrough: A Scam Call Is Detected

1. **The Call:** Caller says, *"Mom, I lost my phone, I need ₹500 right now!"*
2. **Layer 1 — Audio (Spectral):** AI detects robotic voice artifacts → `spectral_score = 85.0`
3. **Layer 2 — Text (Intent):** NLP model flags keywords: `["need", "money", "now"]` → `intent_score = 90.0`
4. **Fusion Formula:** `(0.70 × 85) + (0.30 × 90) = 59.5 + 27.0 = 86.5` → `final_score = 86.5`
5. **Risk Logic:** `86.5 > 75` AND `90 > 60` → label = **HIGH_RISK**
6. **Database Insert:** Complete row saved to `call_logs`.

---

## Viva Defense Notes (Design Decisions)

### Why not use text/NLP alone?
Scammers use the same words as real callers — the fraud is in the voice, not the script. Acoustic anomaly detection is more reliable than transcription, especially with mixed Hinglish.

### Why spectrograms over raw audio?
Raw 1D waveforms are computationally expensive for neural networks. 2D spectrogram images allow the use of CNNs, which are purpose-built for visual pattern recognition.

### Why inject background noise?
Prevents the model from learning a lazy shortcut (clean = AI). Forces it to learn genuine voice artifacts.

---

## Post-Viva To-Do List

1. Scrape more audio data (target: 500–1000 files per category to prevent overfitting).
2. Train the CNN on the spectrogram image dataset.
3. Update `api.py` to replace the mock JSON response with live model predictions.