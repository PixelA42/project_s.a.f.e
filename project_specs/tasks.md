# Implementation Plan: Project S.A.F.E. (Synthetic Audio Fraud Engine)

## Overview

Implement the dual-layer contextual risk engine in four parallel workstreams:
1. **Data Pipeline** — dataset prep, keyword CSV, manifest generation
2. **Core ML** — Spectral_Analyzer (Librosa + scikit-learn), Intent_Analyzer (Whisper + spaCy/NLTK), Score_Fuser, Risk_Classifier
3. **Backend API & Persistence** — FastAPI endpoints, SQLAlchemy models, Docker
4. **React UI** — mock incoming-call screen, risk state transitions, live API integration

Workstreams A through D can be executed in parallel up to Task 23. Integration wiring happens in the final tasks.

---

## Tasks

- [ ] 1. Project scaffolding and shared infrastructure
  - Create the top-level directory layout: `backend/`, `ml/`, `data_pipeline/`, `frontend/`, `tests/`, `models/`
  - Add `pytest.ini` configuring `asyncio_mode = auto` and pointing to the `tests/` directory
  - Add `tests/conftest.py` with three shared fixtures: an in-memory SQLite engine that swaps out PostgreSQL for tests, a temp audio file factory that generates short WAV files, and a mock score factory that returns hardcoded spectral/intent score pairs
  - Update `requirements.txt` to include all runtime and test dependencies: fastapi, uvicorn, sqlalchemy, psycopg2-binary, librosa, soundfile, joblib, numpy, scikit-learn, openai-whisper, spacy, nltk, hypothesis, pytest, pytest-asyncio, python-multipart, httpx
  - _Requirements: 4, 5, 7, 8_

---

### Workstream A — Data Pipeline

- [ ] 2. Implement DataPipeline — dataset splitting
  - Create `data_pipeline/pipeline.py` containing a DataPipeline class
  - Implement a `split_dataset` method that reads the ASVspoof_Dataset directory, shuffles the file list with a fixed random seed for reproducibility, and partitions it into train (80%), validation (10%), and test (10%) lists with no overlap
  - Define a DatasetSplit dataclass holding `train`, `validation`, and `test` lists, plus a `full` property that returns the concatenation of all three
  - _Requirements: 7.1_

  - [ ]* 2.1 Write property test for dataset split disjointness and completeness
    - **Property 14: Dataset Splits Are Disjoint and Complete**
    - **Validates: Requirements 7.1**
    - Generate random lists of integers representing file IDs; run split_dataset; assert that the three output lists share no elements and their combined contents equal the input list exactly

- [ ] 3. Implement DataPipeline — pseudo-labeling
  - Implement an `apply_pseudo_labels` method that accepts a list of unlabeled samples (each with an associated confidence score) and a threshold defaulting to 0.85
  - Samples whose confidence score is at or above 0.85 are moved to a labeled list with their pseudo-label assigned; samples below the threshold remain in an unlabeled list
  - Define a PseudoLabelResult dataclass with `labeled` and `unlabeled` list fields
  - _Requirements: 7.2, 7.4, 7.5_

  - [ ]* 3.1 Write property test for pseudo-label threshold invariant
    - **Property 15: Pseudo-Label Threshold Invariant**
    - **Validates: Requirements 7.4, 7.5**
    - Generate random lists of floats between 0 and 1 as confidence scores; run apply_pseudo_labels; assert every sample with confidence ≥ 0.85 appears in the labeled output and every sample with confidence < 0.85 appears in the unlabeled output

- [ ] 4. Implement DataPipeline — keyword CSV and manifest
  - Create `data_pipeline/distress_keywords_v1.csv` with three columns (keyword, category, weight) and nine seed rows: Bail/financial/1.0, Transfer/financial/1.0, OTP/financial/0.9, UPI/financial/1.0, Money/financial/0.9, Accident/distress/0.8, Hospital/distress/0.8, Help/distress/0.6, Police/authority/0.7
  - Implement a `generate_manifest` method that accepts a DatasetSplit and a PseudoLabelResult and writes a JSON file to `data_pipeline/manifest.json` containing: a UUID run_id, an ISO-8601 UTC timestamp, labeled_count, unlabeled_count, pseudo_labeled_count, pseudo_label_threshold (0.85), and a splits object with train/validation/test counts
  - _Requirements: 7.3, 7.6_

- [ ] 5. Checkpoint — Data Pipeline
  - Run all data pipeline tests and confirm they pass. Raise any questions or blockers before proceeding.

---

### Workstream B — Core ML

- [ ] 6. Implement SpectralAnalyzer — MFCC extraction and serialization
  - Create `ml/spectral_analyzer.py` containing a SpectralAnalyzer class
  - Implement an `extract_features` method that uses Librosa to load the audio file and extract 40 MFCC coefficients across all time frames, returning a 2D NumPy array of shape (40, time_frames) with dtype float32. Raise an AudioProcessingError with a non-empty error_code and description for corrupt, empty, or unsupported-format inputs
  - Implement a `serialize_features` method that saves the feature array to a `.npy` file using NumPy's save function
  - Implement a `deserialize_features` method that loads a `.npy` file and returns the array with the same shape and dtype as the original
  - Define AudioProcessingError as a custom exception class with error_code and description string attributes
  - _Requirements: 1.1, 1.3, 10.1, 10.2_

  - [ ]* 6.1 Write property test for MFCC extraction output shape and dtype
    - **Property 1: MFCC Extraction Produces Valid Output**
    - **Validates: Requirements 1.1**
    - Generate synthetic audio signals using numpy with random durations and sample rates; run extract_features; assert the returned array is 2D, has dtype float32, and both dimensions are greater than zero

  - [ ]* 6.2 Write property test for invalid audio raising structured error
    - **Property 3: Invalid Audio Produces Structured Error**
    - **Validates: Requirements 1.3, 2.5**
    - Generate random binary data and empty byte strings as fake audio inputs; assert that AudioProcessingError is raised and that both error_code and description are non-empty strings

  - [ ]* 6.3 Write property test for MFCC feature array round-trip
    - **Property 17: MFCC Feature Array Round-Trip**
    - **Validates: Requirements 10.2, 10.3**
    - For random valid audio signals, run extract_features then serialize_features then deserialize_features; assert the loaded array passes numpy's allclose check against the original and has identical shape and dtype

- [ ] 7. Implement SpectralAnalyzer — supervised binary classifier
  - Implement a `train` method that iterates over all labeled samples in a DatasetSplit, calls extract_features on each file, mean-pools the resulting (40, time_frames) array to a (40,) vector, and trains a scikit-learn RandomForestClassifier or GradientBoostingClassifier with labels `synthetic` and `real`
  - Implement an `analyze` method that calls extract_features, mean-pools the result, runs classifier inference, maps the synthetic-class probability (0.0–1.0) to a 0–100 score, and returns a SpectralResult containing spectral_score, anomaly_flag, and processing_time_ms. Enforce the 5-second timeout using a ThreadPoolExecutor
  - Save the trained classifier to `models/spectral_model.joblib` using Joblib; load it from disk at class instantiation
  - _Requirements: 1.2, 1.4, 1.5, 8.1, 8.2, 8.5_

  - [ ]* 7.1 Write property test for spectral score range invariant
    - **Property 2: Spectral Score Range Invariant**
    - **Validates: Requirements 1.2**
    - For random valid audio inputs, run analyze; assert the returned spectral_score satisfies 0 ≤ spectral_score ≤ 100

- [ ] 8. Implement SpectralAnalyzer — unsupervised anomaly detection
  - After training the supervised classifier, fit a scikit-learn KMeans model on the same mean-pooled MFCC vectors to identify acoustic anomaly clusters
  - During inference in the `analyze` method, assign the input vector to a cluster; set anomaly_flag = True if the cluster falls outside the distribution of clusters seen during supervised training
  - Write a `models/training_report.json` file documenting the number of clusters used and the cluster assignment method (e.g., KMeans with k=8, Euclidean distance)
  - _Requirements: 9.1, 9.3, 9.4_

- [ ] 9. Implement IntentAnalyzer — Whisper transcription and keyword scoring
  - Create `ml/intent_analyzer.py` containing an IntentAnalyzer class
  - At class instantiation, load the Whisper base model and read `data_pipeline/distress_keywords_v1.csv` into a dictionary mapping each keyword to its weight
  - Implement an `analyze` method that: transcribes the audio file using Whisper; checks whether any speech was detected (return intent_score = 0 and no_speech_detected = True if not); scores the transcript by computing the sum of weights for all matched Distress_Keyword_Set terms, then adds a bonus for urgency linguistic patterns detected via spaCy (imperative verbs) and NLTK (time-pressure phrases); normalizes the raw score to the [0, 100] range using min-max scaling; enforces the 10-second timeout using a ThreadPoolExecutor; and returns an IntentResult containing intent_score, transcript, no_speech_detected, anomaly_flag, and processing_time_ms
  - Raise AudioProcessingError for corrupt, empty, or unsupported-format inputs
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [ ]* 9.1 Write property test for intent score range invariant
    - **Property 4: Intent Score Range Invariant**
    - **Validates: Requirements 2.2**
    - Generate random text strings as transcripts; run the intent scoring function directly (bypassing Whisper); assert the returned intent_score satisfies 0 ≤ intent_score ≤ 100

  - [ ]* 9.2 Write property test for keyword density monotonicity
    - **Property 5: Keyword Density Monotonicity**
    - **Validates: Requirements 2.3**
    - For a random base transcript, compute its intent_score; then append one or more Distress_Keyword_Set terms and compute the augmented score; assert the augmented score is greater than or equal to the base score

  - [ ]* 9.3 Write property test for no-speech zero intent score
    - **Property 6: No-Speech Audio Returns Zero Intent Score**
    - **Validates: Requirements 2.4**
    - Generate silent audio (all-zero samples) and white noise audio; run analyze; assert intent_score == 0 and no_speech_detected == True for both

- [ ] 10. Implement IntentAnalyzer — unsupervised anomaly detection and supervised training
  - After scoring, fit a scikit-learn KMeans model on TF-IDF vectors (or spaCy sentence embeddings) derived from the labeled transcript corpus to identify linguistic anomaly clusters; set anomaly_flag = True when the input transcript falls into an anomaly cluster
  - Implement a `train` method that accepts a list of labeled transcripts with coercion intent labels, trains a scikit-learn binary classifier on TF-IDF features, evaluates it on the held-out test split (assert F1 ≥ 0.70), and saves the model to `models/intent_model.joblib` using Joblib
  - _Requirements: 8.3, 8.4, 8.5, 9.2, 9.3_

- [ ] 11. Implement ScoreFuser and RiskClassifier
  - Create `ml/score_fuser.py` containing a ScoreFuser class with a `fuse` method that accepts spectral_score and intent_score, validates both are in [0, 100] (raising ValueError otherwise), and returns `(0.70 × spectral_score) + (0.30 × intent_score)` as a float
  - Create `ml/risk_classifier.py` containing a RiskClassifier class with a `classify` method that accepts final_score and intent_score and returns HIGH_RISK if final_score > 75 AND intent_score > 60, PRANK if final_score > 75 AND intent_score < 40, and SAFE in all other cases. Also define a RiskLabel string enum with values HIGH_RISK, PRANK, and SAFE
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [ ]* 11.1 Write property test for score fusion formula invariant
    - **Property 7: Score Fusion Formula Invariant**
    - **Validates: Requirements 3.1, 3.2, 3.6**
    - Generate random float pairs in [0, 100]; run fuse; assert the absolute difference between the result and (0.70 × spectral + 0.30 × intent) is less than 1e-9, and that the result is in [0, 100]. Use 500 examples for this property

  - [ ]* 11.2 Write property test for risk classification completeness and correctness
    - **Property 8: Risk Classification Completeness and Correctness**
    - **Validates: Requirements 3.3, 3.4, 3.5**
    - Generate random float pairs in [0, 100]; run classify; assert exactly one label is returned for every input pair and no exception is raised

- [ ] 12. Implement model serialization round-trip verification
  - Add a `verify_serialization` utility function in `ml/utils.py` that accepts a model file path and a fixed test dataset, loads the model from disk using Joblib, runs inference on the test set, and asserts that the predictions are identical to those produced by the original in-memory model before serialization
  - _Requirements: 8.5_

  - [ ]* 12.1 Write property test for model serialization round-trip
    - **Property 16: Model Serialization Round-Trip**
    - **Validates: Requirements 8.5**
    - Train a minimal scikit-learn classifier on a small fixed dataset, serialize it to a temp file using Joblib, deserialize it, run inference on the same fixed dataset, and assert the predictions are identical

- [ ] 13. Checkpoint — Core ML
  - Run all ML unit and property tests and confirm they pass. Raise any questions or blockers before proceeding.

---

### Workstream C — Backend API & Persistence

- [ ] 14. Implement SQLAlchemy CallLog model and database setup
  - Create `backend/models.py` defining a CallLog ORM class mapped to the `call_logs` table with columns: id (UUID primary key, auto-generated), timestamp (timezone-aware DateTime, UTC), caller_id (nullable String), audio_file_path (String, not null), transcript (nullable Text), spectral_score (Float, not null), intent_score (Float, not null), final_score (Float, not null), risk_label (Enum: HIGH_RISK/PRANK/SAFE, not null), anomaly_flag (Boolean, default false)
  - Add database-level CHECK constraints on spectral_score, intent_score, and final_score enforcing the [0, 100] range
  - Create `backend/database.py` with a `get_engine` function that reads the DATABASE_URL environment variable (defaulting to PostgreSQL in production, SQLite in-memory for tests) and a `get_session` context manager for database transactions
  - Call `Base.metadata.create_all` at application startup to create the table if it does not exist
  - _Requirements: 5.1, 5.2, 5.3, 5.5_

  - [ ]* 14.1 Write property test for call log persistence round-trip
    - **Property 10: Call Log Persistence Round-Trip**
    - **Validates: Requirements 5.1**
    - Generate random float scores in [0, 100] and random text strings for transcript and caller_id; persist a CallLog record; query by the returned id; assert every field in the retrieved record matches what was written

  - [ ]* 14.2 Write property test for unique call log IDs
    - **Property 11: Call Log IDs Are Unique**
    - **Validates: Requirements 5.2**
    - Insert a batch of N random CallLog records; collect all id values; assert the set of ids has the same length as N (no duplicates)

  - [ ]* 14.3 Write property test for score constraint enforcement at persistence
    - **Property 12: Score Constraints Enforced at Persistence**
    - **Validates: Requirements 5.5**
    - Attempt to insert CallLog records with spectral_score, intent_score, or final_score values outside [0, 100]; assert a database constraint violation is raised and the record is not persisted

- [ ] 15. Implement Pydantic request/response schemas
  - Create `backend/schemas.py` defining three Pydantic models:
    - EvaluateRiskRequest with fields spectral_score (float, ge=0, le=100, required), intent_score (float, ge=0, le=100, required), caller_id (optional string), audio_file_path (optional string)
    - EvaluateRiskResponse with fields final_score (float), risk_label (RiskLabel enum), spectral_score (float), intent_score (float)
    - ErrorResponse with fields error_code (string), description (string), field (optional string)
  - _Requirements: 4.4, 4.5_

- [ ] 16. Implement FastAPI endpoints
  - Create `backend/main.py` with a FastAPI application instance and Uvicorn entry point
  - Implement POST /api/analyze-audio: accept a multipart audio file upload (WAV or MP3, reject files over 25 MB with HTTP 413), call SpectralAnalyzer.analyze, return JSON with spectral_score and anomaly_flag
  - Implement POST /api/analyze-intent: accept a multipart audio file upload (same constraints), call IntentAnalyzer.analyze, return JSON with intent_score, transcript, no_speech_detected, and anomaly_flag
  - Implement POST /api/evaluate-risk: accept an EvaluateRiskRequest JSON body, call ScoreFuser.fuse and RiskClassifier.classify, persist a CallLog record via SQLAlchemy, return an EvaluateRiskResponse JSON body
  - Register a global exception handler that catches AudioProcessingError and returns HTTP 422 with an ErrorResponse JSON body
  - Register a database failure handler that catches any exception from the CallLog write, logs the full exception with the request identifier, and returns HTTP 500 with a generic error message
  - Add a timeout middleware that returns HTTP 504 if any request exceeds 15 seconds
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 5.1, 5.4_

  - [ ]* 16.1 Write property test for API validation rejecting invalid requests
    - **Property 9: API Validation Rejects Invalid Requests**
    - **Validates: Requirements 4.4, 4.5**
    - Generate random float values outside [0, 100] and request bodies with missing required fields; POST to /api/evaluate-risk; assert HTTP 422 is returned with a JSON error body that names the invalid field

- [ ] 17. Write API endpoint smoke tests and DB failure test
  - Write example-based tests for each of the three endpoints using known valid inputs; assert HTTP 200 and that the response JSON contains the expected fields with values in the correct ranges
  - Write a test that mocks the SQLAlchemy session to raise an exception during the CallLog insert; POST to /api/evaluate-risk with valid inputs; assert HTTP 500 is returned and that the exception was logged with the request identifier
  - _Requirements: 4.1, 4.2, 4.3, 5.4_

- [ ] 18. Checkpoint — Backend API & Persistence
  - Run all backend and persistence tests and confirm they pass. Raise any questions or blockers before proceeding.

---

### Workstream D — React UI

- [ ] 19. Scaffold React + Tailwind project and mock data layer
  - Initialize a new React 18 + TypeScript project in `frontend/` using Vite
  - Install and configure Tailwind CSS following the Vite integration guide
  - Create `frontend/src/types.ts` defining a RiskLabel union type (HIGH_RISK | PRANK | SAFE) and an EvaluateRiskResponse interface with fields final_score, risk_label, spectral_score, and intent_score
  - Create `frontend/src/mockData.ts` exporting three hardcoded mock responses — one per risk label — with realistic score values (e.g., HIGH_RISK: final_score 82, spectral_score 91, intent_score 74; PRANK: final_score 80, spectral_score 88, intent_score 22; SAFE: final_score 31, spectral_score 28, intent_score 40)
  - _Requirements: 6.6_

- [ ] 20. Implement UI state components
  - Create `frontend/src/components/CallScreen.tsx` as the root component that holds a state variable for the current screen (loading | SAFE | PRANK | HIGH_RISK) and renders the appropriate sub-component
  - Implement LoadingState: full-screen neutral background, centered spinner animation, no score values displayed, no prior risk label retained from a previous call
  - Implement SafeState: clean incoming-call layout with a green status badge labeled "Safe Call", caller info section, and a score panel showing "Fraud Score: X / 100 | Voice Synthetic: X | Intent Coercion: X"
  - Implement PrankState: amber-toned background, "Digital Avatar Detected" header in bold amber text, a playful robot icon, the message "Synthetic voice detected — harmless content", and the same score panel
  - Implement ScamLikelyState: full-screen red background, "SCAM LIKELY" header in large white bold text, a pulsing red alert icon, the message "Synthetic Voice + Coercion Detected — Do Not Comply", and the score panel with all three values displayed prominently
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ] 21. Wire mock data to UI states and verify all states render correctly
  - Add a dev-mode toggle bar at the top of CallScreen that shows four buttons: Loading, Safe, Prank, Scam — clicking each button sets the component state to the corresponding mock response and renders the matching sub-component
  - Verify that switching to Loading clears any score values and risk label from the previous state
  - Verify that all three score values (final_score, spectral_score, intent_score) are visible as numeric text in the SafeState, PrankState, and ScamLikelyState renders
  - _Requirements: 6.4, 6.5, 6.6_

  - [ ]* 21.1 Write property test for UI score display in all non-loading states
    - **Property 13: UI Displays All Scores in Non-Loading States**
    - **Validates: Requirements 6.5**
    - Using React Testing Library, render CallScreen with random score triples and each of the three risk labels; query the rendered output for the numeric score values; assert all three values appear in the DOM regardless of which risk label is active

- [ ] 22. Checkpoint — React UI
  - Confirm all four UI states render correctly with mock data and all component tests pass. Raise any questions or blockers before proceeding.

---

### Integration & Wiring

- [ ] 23. Connect React UI to live FastAPI endpoints
  - In CallScreen.tsx, replace the mock data calls with real fetch or axios calls to POST /api/analyze-audio, POST /api/analyze-intent, and POST /api/evaluate-risk
  - Set the component state to loading immediately when a request starts; clear it and render the appropriate risk state when the response arrives
  - If the API returns an error (HTTP 422 or 500), display a user-facing error message in a non-destructive banner without crashing the UI or showing a blank screen
  - Remove the dev-mode toggle bar from the production build (keep it behind a VITE_DEV_MODE environment variable)
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 4.6_

- [ ] 24. Wire SpectralAnalyzer and IntentAnalyzer into the API_Server
  - In `backend/main.py`, instantiate SpectralAnalyzer and IntentAnalyzer as module-level singletons at application startup, loading their respective `.joblib` model files from the `models/` directory
  - Ensure the /api/analyze-audio handler calls SpectralAnalyzer.analyze and the /api/analyze-intent handler calls IntentAnalyzer.analyze
  - Verify that the anomaly_flag returned by each analyzer is included in the API response JSON and also written to the anomaly_flag column of the CallLog record
  - _Requirements: 4.1, 4.2, 9.3_

- [ ] 25. Add Docker configuration
  - Create a `Dockerfile` in the project root using the official Python 3.11.9 slim base image, copying `requirements.txt` and installing dependencies, copying the `backend/` and `ml/` directories, and exposing port 8000 with Uvicorn as the entrypoint
  - Create a `docker-compose.yml` with two services: `db` (PostgreSQL 15 official image, with a named volume for data persistence and environment variables for POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB) and `api` (built from the Dockerfile, with DATABASE_URL wired to the db service, depends_on db)
  - Add a `.env.example` file documenting all required environment variables: DATABASE_URL, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB
  - _Requirements: 4, 5_

- [ ] 26. Write end-to-end integration tests
  - Write a full-pipeline integration test that: uploads a real WAV file to POST /api/analyze-audio and records the spectral_score; uploads the same file to POST /api/analyze-intent and records the intent_score; POSTs both scores to POST /api/evaluate-risk and records the returned id; queries the call_logs table by that id; asserts that every field in the database record matches the values returned in the API responses
  - Write timing tests that submit a 60-second WAV file to each endpoint and assert the response arrives within the SLA: 5 seconds for /api/analyze-audio, 10 seconds for /api/analyze-intent, 15 seconds for /api/evaluate-risk
  - _Requirements: 1.5, 2.6, 4.6, 5.1_

- [ ] 27. Final checkpoint — Ensure all tests pass
  - Run the full test suite using `pytest --tb=short` and confirm all unit, property, and integration tests pass with no failures or errors. Raise any remaining questions before marking the project complete.

---

## Notes

- Tasks marked with `*` are optional property-based tests — skip them for a faster MVP, but include them for a more robust submission
- Each task references specific requirements for traceability
- Property tests use Hypothesis with a minimum of 100 randomly generated examples; Property 7 (fusion formula) uses 500
- Each property test file includes a comment in the format: `# Feature: project-safe, Property N: <Title>`
- SQLite in-memory is used for all persistence tests; swap via the DATABASE_URL environment variable
- Workstreams A, B, C, and D can be executed in parallel up to Task 23
