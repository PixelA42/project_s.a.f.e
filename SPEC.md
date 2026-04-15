# Project S.A.F.E. - Specification

## 1) Project Title
**S.A.F.E.** (Synthetic Audio Fraud Engine)

## 2) Problem Statement
AI voice cloning enables distress scams where attackers mimic family members and demand urgent money transfers.  
A single-layer detector that only checks "real vs synthetic voice" causes false alarms for harmless AI memes/pranks.

## 3) Proposed Solution
A dual-layer contextual risk engine:

- **Layer 1 (Spectral/Audio Origin):** Detect whether incoming speech is synthetic via acoustic features (e.g., MFCC-based pipeline).
- **Layer 2 (NLP/Intent Context):** Transcribe and analyze urgency/coercion signals in spoken content.
- **Final Fraud Score:** Weighted fusion (example baseline: 70% spectral + 30% intent).

## 4) Decision Outcomes
- **High Risk (Scam Likely):** Synthetic voice + distress/financial coercion indicators.
- **Low Risk (Prank/Meme):** Synthetic voice + casual/harmless context.

## 5) Team Work Division (Parallel)
| Role | Core Focus | Specific Responsibilities | How You Work in Parallel |
|---|---|---|---|
| UI & System Architect | Call interface and risk intervention UX | Design incoming-call mock UI; integrate Fraud Score display; implement low-risk "Digital Avatar" state; implement high-risk "SCAM LIKELY" state | Needs no model initially. You build UI flows using **tech/resources you find while researching** responses (fake scores) so screens and transitions are ready early. |
| Core ML Architect | Spectral detection + NLP intent modeling | Build synthetic-origin classifier from audio features; build transcription + intent scoring; implement final score fusion logic | Needs no frontend. You work in **tech/resources you find while researching**, test on local audio, and output stable 0-100 scores. |
| Data Pipeline & Backend API | Data readiness, serving, and integration | Prepare labeled and unlabeled data splits; maintain distress keyword resources; create API for audio input and score output; add validation/error handling | Needs no UI. You test **tech/resources you find while researching** endpoints with **tech/resources you find while researching** and provide predictable API responses. |

## 6) Learning Coverage
- **Supervised:** Binary classification (scam-intent or synthetic-origin labels).
- **Unsupervised:** Clustering/anomaly checks on acoustic/text-derived features.
- **Semi-supervised:** Pseudo-labeling or confidence-based expansion using unlabeled data.

## 7) Milestone Plan
### Monday (50% target)
- Dataset structure finalized (or partially finalized with documented gaps).
- At least one baseline model running.
- UI prototype switching between low-risk/high-risk using mocked response.
- API endpoint available with either mocked or partial model output.

### Tuesday (integration target)
- End-to-end pipeline: audio input -> score computation -> UI alert state.
- Initial observations/metrics documented.
- Final demo script and submission-ready notes.

## 8) Deliverables
- Running demo (local).
- Baseline metrics and early observations.
- Architecture diagram + EPC flow.
- Project report with next-step roadmap.
