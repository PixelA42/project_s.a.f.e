# Requirements Document

## Introduction

Project S.A.F.E. (Synthetic Audio Fraud Engine) is a dual-layer contextual risk engine that detects AI voice cloning scams while filtering out harmless pranks and memes. The system analyzes incoming audio calls through two independent scoring layers — a spectral/acoustic layer that identifies synthetic voice origins, and an NLP/intent layer that detects urgency and financial coercion signals — then fuses the scores into a final risk classification. This dual-layer approach prevents "cry wolf" fatigue caused by single-layer detectors that flag harmless AI-generated content as threats.

---

## Technology Stack

### Backend
- **Language:** Python 3.11.9
- **API Framework:** FastAPI with Uvicorn as the ASGI server — chosen for native async support, which is critical when handling heavy ML inference without blocking concurrent requests
- **Database:** PostgreSQL 15, accessed via SQLAlchemy ORM — provides enterprise-grade reliability and future-proofs the project for vector embedding storage
- **Infrastructure:** Docker Compose to spin up a local PostgreSQL instance so all team members share an identical database environment

### ML / AI Pipeline
- **Audio Feature Extraction:** Librosa — industry-standard Python library for extracting Mel-Frequency Cepstral Coefficients (MFCCs) and other acoustic features from audio signals
- **Speech-to-Text Transcription:** OpenAI Whisper (base model) — robust offline transcription that handles accented speech and noisy audio without requiring a cloud API
- **NLP / Intent Scoring:** spaCy and NLTK — used together for Named Entity Recognition, keyword matching, and urgency linguistic pattern detection (imperative verbs, time-pressure phrases)
- **Supervised / Unsupervised ML:** scikit-learn — provides the binary classifiers (Random Forest or Gradient Boosting) and K-Means clustering used across both analysis layers
- **Model Serialization:** Joblib — serializes trained model artifacts to disk for reproducible loading at API startup
- **Numerical Computing:** NumPy — array manipulation and MFCC feature storage in `.npy` format

### Frontend
- **Framework:** React 18 with TypeScript — component-based architecture that maps cleanly to the four distinct UI states (Loading, Safe, Prank, Scam Likely)
- **Styling:** Tailwind CSS — utility-first framework that allows the UI Architect to build responsive, high-contrast alert screens rapidly without writing custom CSS
- **Build Tool:** Vite — fast development server and bundler for the React frontend

### Testing
- **Test Runner:** pytest with pytest-asyncio for async FastAPI endpoint tests
- **Property-Based Testing:** Hypothesis — generates random inputs to verify correctness invariants across all components
- **In-Memory DB for Tests:** SQLite (swapped via `DATABASE_URL` environment variable) — eliminates the need for a live PostgreSQL instance during unit and property tests

---

## Glossary

- **Engine**: The S.A.F.E. system as a whole, encompassing all layers, APIs, and UI components.
- **Spectral_Analyzer**: The Layer 1 component responsible for acoustic feature extraction and synthetic voice detection.
- **Intent_Analyzer**: The Layer 2 component responsible for audio transcription and urgency/coercion intent scoring.
- **Score_Fuser**: The component that combines spectral_score and intent_score into a final_score using the weighted formula.
- **Risk_Classifier**: The component that maps a final_score and intent_score pair to a risk_label.
- **API_Server**: The FastAPI/Uvicorn backend that exposes audio analysis endpoints.
- **Call_Log_Store**: The PostgreSQL database layer (via SQLAlchemy) that persists call analysis records.
- **UI**: The React + Tailwind CSS frontend that displays call state and risk alerts to the user.
- **spectral_score**: A numeric value in the range [0, 100] representing the probability that an audio sample originates from a synthetic (AI-generated) voice. Higher values indicate higher likelihood of synthetic origin.
- **intent_score**: A numeric value in the range [0, 100] representing the degree of urgency and financial coercion detected in the transcribed speech content. Higher values indicate stronger coercion signals.
- **final_score**: The weighted fusion score computed as `(0.70 × spectral_score) + (0.30 × intent_score)`.
- **risk_label**: One of three classification outcomes: `HIGH_RISK`, `PRANK`, or `SAFE`.
- **Distress_Keyword_Set**: A versioned resource file containing coercion-related keywords (e.g., Bail, Transfer, OTP, Accident, Hospital, Police, UPI, Money, Help) with associated categories and weights. Maintained by the Data Pipeline team and loaded by the Intent_Analyzer at startup.
- **MFCC**: Mel-Frequency Cepstral Coefficients — acoustic features extracted from audio using Librosa, used as input to the Spectral_Analyzer's classifier.
- **Whisper**: The OpenAI Whisper model used by the Intent_Analyzer for offline audio-to-text transcription.
- **ASVspoof_Dataset**: The labeled dataset of real and synthetic audio samples used for supervised training of the Spectral_Analyzer.
- **Pseudo_Label**: A programmatically assigned label applied to unlabeled audio samples during semi-supervised training when the model confidence exceeds a defined threshold.

---

## Requirements

### Requirement 1: Spectral Analysis — Synthetic Voice Detection

**User Story:** As a Core ML Architect, I want the system to analyze acoustic features of incoming audio and produce a spectral score, so that I can quantify the likelihood that a voice is AI-generated.

**Technology used:** Librosa for MFCC extraction, scikit-learn for binary classification, Joblib for model serialization, NumPy for feature array handling.

#### Acceptance Criteria

1. WHEN an audio file is submitted to the Spectral_Analyzer, THE Spectral_Analyzer SHALL extract MFCC features from the audio signal using Librosa, producing a 2D feature array with 40 MFCC coefficients across all time frames.
2. WHEN MFCC features are extracted, THE Spectral_Analyzer SHALL produce a spectral_score in the range [0, 100] representing the probability of synthetic voice origin, derived from a trained scikit-learn binary classifier.
3. IF the submitted audio file is corrupt, empty, or in an unsupported format, THEN THE Spectral_Analyzer SHALL return a structured error response containing an error code and a human-readable description.
4. THE Spectral_Analyzer SHALL be trained using the ASVspoof_Dataset with supervised binary classification (synthetic vs. real labels) and SHALL achieve a minimum F1 score of 0.75 on the held-out test split.
5. WHEN the Spectral_Analyzer produces a spectral_score, THE Spectral_Analyzer SHALL complete the scoring computation within 5 seconds for audio samples up to 60 seconds in duration.

---

### Requirement 2: Intent Analysis — Urgency and Coercion Detection

**User Story:** As a Core ML Architect, I want the system to transcribe spoken audio and score the urgency and coercion level of the content, so that I can distinguish distress scams from harmless synthetic speech.

**Technology used:** OpenAI Whisper for transcription, spaCy and NLTK for NLP scoring, scikit-learn for supervised intent classification, Joblib for model serialization.

#### Acceptance Criteria

1. WHEN an audio file is submitted to the Intent_Analyzer, THE Intent_Analyzer SHALL transcribe the spoken content into text using the Whisper base model running locally.
2. WHEN a transcript is produced, THE Intent_Analyzer SHALL analyze the transcript against the Distress_Keyword_Set and produce an intent_score in the range [0, 100].
3. THE Intent_Analyzer SHALL assign higher intent_score values to transcripts containing a greater density of Distress_Keyword_Set terms combined with urgency linguistic patterns (e.g., imperative verbs, time-pressure phrases) detected via spaCy and NLTK.
4. IF the audio file contains no intelligible speech, THEN THE Intent_Analyzer SHALL return an intent_score of 0 and include a `no_speech_detected` flag in the response.
5. IF the submitted audio file is corrupt, empty, or in an unsupported format, THEN THE Intent_Analyzer SHALL return a structured error response containing an error code and a human-readable description.
6. WHEN the Intent_Analyzer produces an intent_score, THE Intent_Analyzer SHALL complete the transcription and scoring within 10 seconds for audio samples up to 60 seconds in duration.

---

### Requirement 3: Score Fusion and Risk Classification

**User Story:** As a Core ML Architect, I want the spectral and intent scores to be fused into a single final score and mapped to a risk label, so that the system can produce a single, actionable risk decision.

**Technology used:** Pure Python arithmetic for the fusion formula; no external ML library required for this component.

#### Acceptance Criteria

1. WHEN a spectral_score and an intent_score are available, THE Score_Fuser SHALL compute the final_score using the formula: `final_score = (0.70 × spectral_score) + (0.30 × intent_score)`.
2. THE Score_Fuser SHALL produce a final_score in the range [0, 100].
3. WHEN a final_score and intent_score are available, THE Risk_Classifier SHALL assign the risk_label `HIGH_RISK` if final_score > 75 AND intent_score > 60.
4. WHEN a final_score and intent_score are available, THE Risk_Classifier SHALL assign the risk_label `PRANK` if final_score > 75 AND intent_score < 40.
5. WHEN a final_score and intent_score are available, THE Risk_Classifier SHALL assign the risk_label `SAFE` if the conditions for `HIGH_RISK` and `PRANK` are not met.
6. FOR ALL valid (spectral_score, intent_score) pairs in [0, 100] × [0, 100], THE Score_Fuser SHALL produce a final_score that satisfies: `final_score = (0.70 × spectral_score) + (0.30 × intent_score)` (fusion formula invariant).

---

### Requirement 4: Backend API — Audio Analysis Endpoints

**User Story:** As a Data Pipeline & Backend API developer, I want a set of REST API endpoints that accept audio input and return computed scores and risk labels, so that the UI and downstream consumers can integrate with the Engine without direct model access.

**Technology used:** FastAPI with Uvicorn, Pydantic for request/response validation, python-multipart for audio file uploads.

#### Acceptance Criteria

1. THE API_Server SHALL expose a `POST /api/analyze-audio` endpoint that accepts an audio file and returns a JSON response containing spectral_score and anomaly_flag.
2. THE API_Server SHALL expose a `POST /api/analyze-intent` endpoint that accepts an audio file and returns a JSON response containing intent_score, transcript, no_speech_detected, and anomaly_flag.
3. THE API_Server SHALL expose a `POST /api/evaluate-risk` endpoint that accepts a JSON body with spectral_score and intent_score and returns a JSON response containing final_score, risk_label, spectral_score, and intent_score.
4. WHEN a request is received at any endpoint, THE API_Server SHALL validate that all required fields are present and within acceptable ranges before processing.
5. IF a required field is missing or out of range, THEN THE API_Server SHALL return an HTTP 422 response with a structured JSON error body identifying the invalid field and the reason for rejection.
6. WHEN a valid request is processed successfully, THE API_Server SHALL return an HTTP 200 response within 15 seconds for audio samples up to 60 seconds in duration.
7. THE API_Server SHALL accept audio files in WAV and MP3 formats with a maximum file size of 25 MB.

---

### Requirement 5: Call Log Persistence

**User Story:** As a Data Pipeline & Backend API developer, I want each analyzed call to be persisted in a database, so that analysis results are auditable and available for model retraining.

**Technology used:** PostgreSQL 15 as the database, SQLAlchemy ORM for Python-to-database interaction, psycopg2-binary as the database adapter.

#### Acceptance Criteria

1. WHEN a risk evaluation is completed, THE Call_Log_Store SHALL persist a record containing: id (UUID), timestamp (UTC), caller_id, audio_file_path, transcript, spectral_score, intent_score, final_score, and risk_label.
2. THE Call_Log_Store SHALL assign a unique UUID v4 to each persisted record, guaranteed to be distinct across all records.
3. THE Call_Log_Store SHALL record the timestamp of the analysis at the time the evaluation completes, stored in UTC with timezone awareness.
4. IF a database write fails, THEN THE API_Server SHALL return an HTTP 500 response and log the failure with the associated request identifier — no internal error details SHALL be exposed in the response body.
5. THE Call_Log_Store SHALL enforce database-level CHECK constraints ensuring that spectral_score, intent_score, and final_score values are in the range [0, 100].

---

### Requirement 6: Frontend Risk Display — UI States

**User Story:** As a UI & System Architect, I want the call interface to display the correct visual state based on the risk label, so that users can immediately understand the threat level of an incoming call.

**Technology used:** React 18 with TypeScript, Tailwind CSS for styling, Vite as the build tool. During development, mock JSON responses are used in place of live API calls so the UI can be built and tested independently.

#### Acceptance Criteria

1. WHEN the UI receives a risk_label of `HIGH_RISK`, THE UI SHALL display the "SCAM LIKELY" alert state — a full-screen red background with a bold warning header, a pulsing alert icon, and the message "Synthetic Voice + Coercion Detected" to communicate immediate danger.
2. WHEN the UI receives a risk_label of `PRANK`, THE UI SHALL display the "Digital Avatar" state — an amber-toned screen with a playful indicator and the message "Synthetic Voice Detected — Harmless Content" to reassure the user without causing panic.
3. WHEN the UI receives a risk_label of `SAFE`, THE UI SHALL display the standard incoming-call screen — a clean, neutral design with a green indicator and no fraud warning, matching the appearance of a normal phone call UI.
4. WHEN the UI is awaiting a score response from the API, THE UI SHALL display a loading state with a spinner animation and SHALL NOT render any risk_label from a prior call, ensuring stale data is never shown.
5. THE UI SHALL display the final_score, spectral_score, and intent_score as numeric values (e.g., "Final Score: 82 / 100") alongside the risk_label in all non-loading states, giving the user full transparency into how the decision was reached.
6. WHERE mock score data is used during development, THE UI SHALL render all three risk states correctly using hardcoded mock responses before live API integration is complete, allowing the UI Architect to work independently of the backend.

---

### Requirement 7: Data Pipeline — Dataset Preparation

**User Story:** As a Data Pipeline & Backend API developer, I want the training datasets to be structured and split correctly, so that all three learning paradigms (supervised, unsupervised, semi-supervised) can be executed reliably.

**Technology used:** Python scripts for dataset splitting and manifest generation, the ASVspoof dataset as the primary labeled audio source, a versioned CSV file for the Distress_Keyword_Set.

#### Acceptance Criteria

1. THE Data_Pipeline SHALL partition the ASVspoof_Dataset into a labeled training split (80%), a labeled validation split (10%), and a held-out test split (10%) before model training begins, with no sample appearing in more than one split.
2. THE Data_Pipeline SHALL maintain a separate unlabeled audio batch for use in semi-supervised learning, kept entirely separate from the labeled splits.
3. THE Data_Pipeline SHALL maintain the Distress_Keyword_Set as a versioned CSV file (e.g., `distress_keywords_v1.csv`) containing keyword, category, and weight columns, accessible to the Intent_Analyzer at runtime.
4. WHEN the semi-supervised training phase runs, THE Data_Pipeline SHALL apply Pseudo_Labels to unlabeled samples where the Spectral_Analyzer or Intent_Analyzer confidence score exceeds 0.85.
5. IF a Pseudo_Label confidence score is below 0.85, THEN THE Data_Pipeline SHALL exclude that sample from the labeled training set and retain it in the unlabeled batch.
6. THE Data_Pipeline SHALL produce a JSON dataset manifest file after each pipeline run, documenting the run ID, timestamp, and counts of labeled, unlabeled, pseudo-labeled samples, and per-split sizes.

---

### Requirement 8: Model Training — Supervised Classification

**User Story:** As a Core ML Architect, I want supervised binary classifiers trained on labeled data, so that the Spectral_Analyzer and Intent_Analyzer produce reliable baseline scores.

**Technology used:** scikit-learn for classifier training (Random Forest or Gradient Boosting), Joblib for saving and loading trained model artifacts.

#### Acceptance Criteria

1. THE Spectral_Analyzer SHALL be trained as a binary classifier using mean-pooled MFCC features extracted from the ASVspoof_Dataset labeled split, with class labels `synthetic` and `real`.
2. WHEN training completes, THE Spectral_Analyzer SHALL achieve a minimum F1 score of 0.75 on the held-out test split.
3. THE Intent_Analyzer scoring model SHALL be trained using labeled transcripts annotated with coercion intent labels derived from the Distress_Keyword_Set.
4. WHEN training completes, THE Intent_Analyzer scoring model SHALL achieve a minimum F1 score of 0.70 on the held-out test split.
5. WHEN a trained model artifact is produced, THE Engine SHALL serialize the model to disk as a `.joblib` file using Joblib, enabling reproducible loading at API startup without retraining.

---

### Requirement 9: Model Training — Unsupervised Anomaly Detection

**User Story:** As a Core ML Architect, I want unsupervised clustering applied to acoustic and text-derived features, so that the system can surface anomalous patterns not captured by the supervised classifiers.

**Technology used:** scikit-learn K-Means clustering for both acoustic and text-derived feature vectors.

#### Acceptance Criteria

1. THE Spectral_Analyzer SHALL apply K-Means clustering to MFCC feature vectors to identify acoustic anomaly clusters that fall outside the supervised training distribution.
2. THE Intent_Analyzer SHALL apply K-Means clustering to text-derived feature vectors (TF-IDF or spaCy embeddings) to identify linguistic anomaly patterns.
3. WHEN an audio sample falls into an anomaly cluster, THE Engine SHALL include an `anomaly_flag` field set to `true` in the API response for that sample.
4. THE Engine SHALL document the number of clusters and the cluster assignment method used in a training report JSON file stored alongside the model artifacts.

---

### Requirement 10: Parser and Serializer — Audio Feature Round-Trip

**User Story:** As a Core ML Architect, I want MFCC feature extraction and serialization to be verifiable end-to-end, so that feature data is not corrupted between extraction, storage, and model inference.

**Technology used:** NumPy for feature array storage in `.npy` binary format.

#### Acceptance Criteria

1. WHEN MFCC features are extracted from an audio file, THE Spectral_Analyzer SHALL serialize the feature array to NumPy `.npy` format on disk for caching and reuse.
2. WHEN a serialized feature array is loaded from disk, THE Spectral_Analyzer SHALL deserialize it into a NumPy array with the same shape (40 × time_frames) and dtype (float32) as the original.
3. FOR ALL valid audio files processed by the Spectral_Analyzer, extracting features then serializing then deserializing SHALL produce a feature array numerically equivalent to the original extracted array — no precision loss or shape change is acceptable.

---
