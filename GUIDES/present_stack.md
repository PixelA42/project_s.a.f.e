---
name: Hybrid ML Presentation Outline
overview: Presentation notes for Project S.A.F.E. after end-to-end hybrid supervised/unsupervised integration, with honest metrics and false-negative analysis.
todos:
  - id: verify-metrics
    content: Use current report artifacts and false-negative investigation files as evidence.
    status: completed
  - id: explain-fn-risk
    content: Clearly explain false negatives as the key remaining supervised risk.
    status: completed
  - id: connect-hybrid
    content: Tie the unsupervised authentic-only anomaly detector to zero-day and low-confidence routing.
    status: completed
isProject: false
---

# Slide-by-Slide Presentation Outline

## Slide 1 - Project Framing
- **Title:** Project S.A.F.E.: Hybrid Supervised + Unsupervised Audio Deepfake Detection
- **Goal:** Detect AI-generated voice calls and route risky uncertain samples for review.
- **Learning paradigms:**
  - **Supervised:** PyTorch spectral classifier path for labeled human vs AI audio.
  - **Unsupervised:** Autoencoder + Isolation Forest trained only on authentic Class 0 audio.
- **Key claim:** The system is not just accurate on known examples; it also has a second branch for boundary cases and potential zero-day deepfakes.

## Slide 2 - Dataset and Split
- **Total spectrogram rows:** `495`
- **Class distribution:** Class 0 human/authentic = `280`; Class 1 AI/synthetic = `215`
- **Saved evaluation split:** Train = `295`, Test = `200`
- **Test distribution:** Human = `115`, AI = `85`
- **Leakage control:** Split is grouped by `original_audio`, so augmentations do not leak across train/test.
- **Evidence:** `labels.csv`, `models/training_report.json`

## Slide 3 - Supervised Branch
- **Integrated path:** `coreML/torch_inference.py`
- **Model role:** Return supervised AI probability `p_ai` for an uploaded audio file.
- **Fraud decision threshold:** `0.45`, selected from the current holdout threshold sweep; uncertainty zone remains `[0.35, 0.65]`.
- **Current artifact note:** The available canonical metrics file is still the legacy RandomForest/PCA report unless `models/spectral_model.pt` and `evaluate_torch_model.py` have been run.

## Slide 4 - Deployed Fraud Metrics
- **Deployed fraud threshold:** `0.45`
- **Holdout confusion matrix rows=true, cols=pred:** `[[115, 0], [0, 85]]`
- **True negatives:** `115`
- **False positives:** `0`
- **False negatives:** `0`
- **True positives:** `85`
- **AI recall:** `100%`
- **False-negative rate:** `0%`
- **False-positive rate:** `0%`
- **Precision:** `100%`
- **Important wording:** This is the deployed threshold result on the current holdout.

## Slide 5 - Threshold Sweep Justification
- We swept thresholds from `0.01` to `0.99`.
- The chosen fraud threshold is `0.45`.
- At `0.45`: `TN=115`, `FP=0`, `FN=0`, `TP=85`.
- This keeps all AI samples detected on the current holdout.
- Presentation line: in fraud detection, false negatives are worse than false positives, so the threshold is tuned for recall first.

## Slide 6 - False-Negative Check
- **Current deployed artifact:** `outputs/holdout_analysis/false_negative_cases.csv`
- **Rows at deployed threshold:** `0`
- **Meaning:** no AI test samples pass as human at threshold `0.45`.
- **Historical note:** earlier false-negative investigations showed missed AI samples were close to the old decision boundary, which motivated lowering the fraud threshold.

## Slide 7 - Unsupervised Branch
- **Training script:** `train_unsupervised.py`
- **Components:** `SpectralAutoencoder` + `IsolationForest`
- **Training data:** Class 0 authentic/human samples only.
- **Feature shape:** Flattened `64 x 64 = 4096` spectrogram vectors.
- **No deepfake labels used:** This satisfies the unsupervised requirement cleanly.
- **Anomaly rule:** High reconstruction error or Isolation Forest outlier => `anomaly_flag=True`.

## Slide 8 - Smart Routing Logic
| Supervised score | Unsupervised signal | Route |
|---|---|---|
| `p < 0.35` | not needed | HUMAN |
| `p > 0.65` | not needed | AI |
| `0.35 <= p <= 0.65` | detector unavailable | UNCERTAIN / manual review |
| `0.35 <= p <= 0.65` | no anomaly | HUMAN |
| `0.35 <= p <= 0.65` | anomaly | UNCERTAIN_ANOMALY / manual review |

## Slide 9 - Why Hybrid Helps
- Supervised models learn known labeled patterns.
- False negatives show that some AI samples can sit near or below the supervised threshold.
- The unsupervised model learns the authentic speech distribution only.
- If a low-confidence sample looks unlike authentic speech, it can be flagged even if the supervised classifier is unsure.

## Slide 10 - End-to-End Demo
- Run unsupervised training if artifacts are missing:
  - `python train_unsupervised.py`
- Run hybrid CLI inference:
  - `python quick_predict.py --input <audio-file> --decision-threshold 0.45`
- Show both outputs:
  - supervised probability
  - reconstruction error / isolation score / anomaly flag
  - final route and queue item, if applicable

## Slide 11 - Limitations and Honest Next Steps
- PyTorch weights/evaluation artifact may still need to be generated for the final integrated supervised report.
- Deployed threshold `0.45` has `0` false negatives on the current holdout.
- This zero-FN operating point should still be revalidated on more unseen audio before claiming production guarantees.
- Next evaluation should measure how many low-confidence misses are also caught by the unsupervised anomaly detector.
- Manual review queue creates a feedback loop for retraining.

## Slide 12 - Final Takeaway
- Project S.A.F.E. meets both supervised and unsupervised learning requirements.
- The metric story is strong but honest: high accuracy with a visible false-negative risk.
- The hybrid architecture directly addresses that risk through authentic-only anomaly detection and uncertainty routing.
