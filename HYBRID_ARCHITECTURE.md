# Hybrid Supervised / Unsupervised Architecture

## Overview

Project S.A.F.E. implements a **hybrid inference pipeline** that combines supervised and unsupervised learning to satisfy project requirements while providing robust detection of both known and zero-day deepfakes.

### The Problem

A purely supervised model can only detect deepfake patterns it has seen during training. When faced with:
- Novel synthesis techniques (zero-day deepfakes)
- Ambiguous samples near the decision boundary
- Distribution shift in real-world deployment

...the supervised model becomes uncertain and may misclassify.

### The Solution

We augment the supervised PyTorch spectral classifier with an **unsupervised anomaly detector** trained exclusively on authentic (Class 0) audio. This detector learns what real human speech looks like without ever seeing deepfake labels. At inference time:

1. **Supervised model** provides a probability score [0, 1]
2. **Unsupervised model** flags samples that deviate from the authentic baseline
3. **Smart routing logic** uses the unsupervised signal to resolve uncertain cases

---

## Architecture Diagram

```text
AUDIO INPUT
   |
   +-----------------------------+
   |                             |
   v                             v
SUPERVISED PATH              UNSUPERVISED PATH
PyTorch SpectralCNN          Autoencoder + Isolation Forest
Trained on Class 0/1         Trained on Class 0 authentic only
Output: p_ai                 Output: reconstruction_error, iso_score, anomaly_flag
   |                             |
   +-------------+---------------+
                 v
          SMART ROUTING

  p < 0.35                         -> HUMAN
  p > 0.65                         -> AI
  0.35 <= p <= 0.65 + no detector  -> UNCERTAIN / review
  0.35 <= p <= 0.65 + anomaly      -> UNCERTAIN_ANOMALY / review
  0.35 <= p <= 0.65 + no anomaly
      and p < fraud threshold      -> HUMAN
  0.35 <= p <= 0.65 + no anomaly
      and p >= fraud threshold     -> AI (fail closed)
```

---

## Components

### 1. Supervised Model (SpectralCNN)

**File:** `coreML/torch_inference.py`  
**Weights:** `models/spectral_model.pt`

A compact convolutional neural network trained on labeled spectrograms:
- **Input:** 1-channel mel-spectrogram (224×224 pixels)
- **Architecture:** 3 Conv2D layers + AdaptiveAvgPool + Linear classifier
- **Training:** Binary cross-entropy on human (0) vs AI (1) labels
- **Output:** Probability that the input is AI-generated [0, 1]

**Strengths:**
- High accuracy on known deepfake techniques
- Fast inference (~100ms per sample)
- Learns discriminative features directly from labels

**Limitations:**
- Cannot detect novel synthesis methods not in training data
- Uncertain on ambiguous samples near decision boundary

---

### 2. Unsupervised Anomaly Detector

**File:** `train_unsupervised.py`  
**Artifacts:**
- `models/unsupervised_autoencoder.pt`
- `models/unsupervised_isolation_forest.joblib`
- `models/unsupervised_scaler.joblib`

**Key constraint:** Trained **exclusively on Class 0 (authentic/human) samples**. No deepfake labels are used.

#### 2a. SpectralAutoencoder (PyTorch)

A fully-connected autoencoder that learns to compress and reconstruct authentic spectrogram features:

- **Input:** Flattened 64×64 grayscale spectrogram (4096 dims)
- **Encoder:** Linear → BatchNorm → ReLU → Dropout → Linear → ReLU → Linear
- **Bottleneck:** 32-dimensional latent space (configurable)
- **Decoder:** Mirrors encoder in reverse
- **Loss:** Mean Squared Error (MSE) on reconstruction

**Anomaly signal:** High reconstruction error indicates the input deviates from the learned real-speech manifold.

**Threshold calibration:** After training, reconstruction errors are computed on the authentic training set. The anomaly threshold is set at:

```
threshold = mean(errors) + 2.0 × std(errors)
```

Samples exceeding this threshold are flagged as anomalous.

#### 2b. IsolationForest (scikit-learn)

An ensemble of isolation trees that partition the authentic feature space:

- **Input:** Same 4096-dim flattened spectrogram features
- **Training:** Fits 200 trees on authentic samples with contamination=0.05
- **Anomaly signal:** Negative `score_samples()` output or `predict() == -1`

**Strengths:**
- Robust to outliers in the training set
- Fast inference
- Complementary to autoencoder (different anomaly detection mechanism)

---

### 3. Smart Uncertainty Routing

**Files:** `quick_predict.py`, `api.py`

The routing logic uses the supervised probability to define an **uncertainty zone** [0.35, 0.65]. Samples in this zone are ambiguous — the supervised model is not confident.

**Decision table:**

| Supervised probability | Unsupervised anomaly | Routing decision | Explanation |
|---|---|---|---|
| **< 0.35** | any | **HUMAN** | Supervised model confident → trust it |
| **> 0.65** | any | **AI** | Supervised model confident → trust it |
| **0.35–0.65** | unsupervised not ready | **UNCERTAIN** | Queue for manual review |
| **0.35-0.65 and p < fraud threshold** | anomaly_flag = False | **HUMAN** | No anomaly detected and supervised score remains below the fraud threshold |
| **0.35-0.65 and p >= fraud threshold** | anomaly_flag = False | **AI** | Fraud-safe fail-closed routing prevents low-confidence AI detections from passing as human |
| **0.35-0.65** | anomaly_flag = True | **UNCERTAIN_ANOMALY** | Anomaly detected -> flag for review |

**Why this works:**
- When the supervised model is confident, we trust it (it has seen many labeled examples).
- When the supervised model is uncertain, we defer to the unsupervised detector, which can catch zero-day deepfakes that the supervised model has never seen.
- Uncertain samples are queued for human review when the unsupervised detector is unavailable or flags an anomaly. If the supervised model is uncertain but already above the fraud threshold, the system fails closed as AI even when the unsupervised detector sees no anomaly.

---

## Training the Hybrid System

### Prerequisites

1. Run the spectrogram generation pipeline:
   ```powershell
   python audio_pipeline.py
   ```
   This creates `spectrograms/` and `labels.csv`.

2. Train the supervised model:
   ```powershell
   python train_model.py
   ```
   This creates `models/spectral_model.pt` (or use the PyTorch training script if available).

### Train the Unsupervised Detector

```powershell
python train_unsupervised.py
```

**What it does:**
1. Loads `labels.csv` and filters for `label == 0` (authentic samples only)
2. Loads spectrogram PNGs and flattens them to feature vectors
3. Fits a StandardScaler on the authentic features
4. Trains the SpectralAutoencoder for 60 epochs (configurable)
5. Trains the IsolationForest on the same features
6. Calibrates the anomaly threshold from reconstruction errors
7. Saves all artifacts to `models/`

**Optional flags:**
```powershell
python train_unsupervised.py --epochs 100 --latent-dim 64 --std-multiplier 2.5
```

**Output:**
```
[1/5] Loading authentic spectrogram features...
  Authentic samples found : 250
  Feature matrix shape    : (250, 4096)
  Loaded in 2.3s

[2/5] Fitting StandardScaler on authentic features...
  Input dimension: 4096

[3/5] Training Autoencoder (60 epochs, latent_dim=32)...
    Epoch    1/60  loss=0.123456
    Epoch   10/60  loss=0.045678
    ...
  Training complete in 45.2s
  Final reconstruction loss: 0.012345

[4/5] Training Isolation Forest...
  Training complete in 3.1s

[5/5] Calibrating anomaly thresholds and saving artifacts...
  Anomaly threshold (AE): 0.034567
  [OK] Autoencoder saved  -> models/unsupervised_autoencoder.pt
  [OK] Isolation Forest   -> models/unsupervised_isolation_forest.joblib
  [OK] Scaler             -> models/unsupervised_scaler.joblib
  [OK] Training report    -> models/unsupervised_training_report.json
```

---

## Running Hybrid Inference

### CLI (quick_predict.py)

```powershell
python quick_predict.py --input path/to/audio.wav
```

**Output:**
```
============================================================
  HYBRID INFERENCE RESULT
============================================================
  Input                  : path/to/audio.wav

  [SUPERVISED]
  Prediction             : 1 (AI)
  Positive probability   : 0.5234
  Decision threshold     : 0.4500
  Uncertain zone         : True

  [UNSUPERVISED]
  Model ready            : True
  Reconstruction error   : 0.045678
  Isolation score        : -0.123456
  AE anomaly             : True
  IF anomaly             : True
  Anomaly flag           : True

  [ROUTING DECISION]
  Decision               : UNCERTAIN_ANOMALY
  Reason                 : Supervised uncertain (p=0.523); unsupervised
                           anomaly detected (recon_err=0.0457,
                           iso_score=-0.1235) — flagged for manual review
  Queue item ID          : 12345678-abcd-1234-5678-1234567890ab
============================================================
```

### HTTP API (api.py)

Start the server:
```powershell
python api.py
```

Send a request:
```bash
curl -X POST http://127.0.0.1:5000/detect-audio \
  -F "file=@path/to/audio.wav"
```

**Response:**
```json
{
  "prediction": "AI",
  "routing_decision": "UNCERTAIN_ANOMALY",
  "routing_reason": "Supervised uncertain (p=0.523); unsupervised anomaly detected (recon_err=0.0457, iso_score=-0.1235) — flagged for manual review",
  "supervised_probability": 0.5234,
  "supervised_label": "AI",
  "is_uncertain": true,
  "reconstruction_error": 0.045678,
  "isolation_score": -0.123456,
  "anomaly_flag": true,
  "ae_anomaly": true,
  "if_anomaly": true,
  "unsupervised_ready": true,
  "filename": "audio.wav",
  "timestamp": "2026-04-29T12:34:56.789Z",
  "queue_item_id": "12345678-abcd-1234-5678-1234567890ab"
}
```

---

## Uncertainty Queue

Samples in the uncertainty zone [0.35, 0.65] are added to the `UncertaintyQueue` when the unsupervised detector is unavailable or flags an anomaly. Threshold-positive uncertain samples fail closed as AI instead of being passed as human.

**Queue directory structure:**
```
outputs/uncertainty_queue/
├── pending/       # Items awaiting review
├── reviewed/      # Items that have been reviewed
└── escalated/     # Items escalated to security team
```

**Queue item fields:**
- `item_id`: UUID for tracking
- `audio_file_path`: Path to the audio file
- `predicted_probability`: Supervised model output [0, 1]
- `predicted_label`: 0 (human) or 1 (AI)
- `confidence_score`: Max probability
- `spectral_score`, `intent_score`: Component scores (if available)
- `reconstruction_error`, `isolation_score`: Unsupervised scores
- `status`: QUEUED → REVIEWED → RESOLVED/ESCALATED/DISMISSED
- `tags`: ["uncertainty_zone", "anomaly_detected", "auto_routed", ...]

**Export queue for review:**
```python
from coreML.uncertainty_queue import UncertaintyQueue

queue = UncertaintyQueue()
queue.export_queue_summary("outputs/uncertainty_review.csv")
```

---

## Why This Satisfies Project Requirements

### Supervised Learning Requirement

✅ The PyTorch SpectralCNN is trained on labeled human (Class 0) and AI (Class 1) spectrograms using binary cross-entropy loss. It directly optimizes for the classification boundary between real and synthetic speech.

### Unsupervised Learning Requirement

✅ The SpectralAutoencoder and IsolationForest are trained with **no labels whatsoever** — only the authentic (Class 0) feature distribution. They satisfy the unsupervised learning requirement by learning the structure of real speech without any supervision signal.

### Practical Value

Beyond satisfying requirements, the hybrid approach provides genuine value:

1. **Zero-day deepfake detection:** The unsupervised model can flag novel synthesis techniques that the supervised model has never seen.

2. **Reduced false positives:** When the supervised model is uncertain, the unsupervised model can clear authentic samples that happen to be near the decision boundary.

3. **Human-in-the-loop:** Uncertain anomalous samples, and uncertain samples that cannot use the unsupervised detector yet, are queued for manual review. Uncertain samples that look authentic to the unsupervised detector are passed as HUMAN.

4. **Graceful degradation:** If the unsupervised model is not trained yet, the system falls back to supervised-only inference and queues uncertain samples for review.

---

## Evaluation Metrics

### Supervised Model

- **Accuracy:** Proportion of correct predictions on labeled test set
- **F1 Score:** Harmonic mean of precision and recall
- **ROC-AUC:** Area under the receiver operating characteristic curve
- **PR-AUC:** Area under the precision-recall curve

### Unsupervised Model

- **Reconstruction error distribution:** Mean and std on authentic test set
- **Anomaly detection rate:** Proportion of deepfakes flagged as anomalous
- **False positive rate:** Proportion of authentic samples flagged as anomalous
- **Complementarity:** Proportion of uncertain supervised samples correctly resolved by unsupervised signal

### Hybrid System

- **Uncertainty zone coverage:** Proportion of test samples in [0.35, 0.65]
- **Routing accuracy:** Proportion of uncertain samples correctly routed
- **Queue efficiency:** Proportion of queued samples that are genuinely ambiguous
- **Zero-day detection:** Ability to flag novel deepfake techniques not in training data

---

## Configuration

All thresholds and hyperparameters are configurable:

**Uncertainty zone boundaries** (in `quick_predict.py` and `api.py`):
```python
UNCERTAINTY_LOWER: float = 0.35
UNCERTAINTY_UPPER: float = 0.65
```

**Autoencoder hyperparameters** (CLI flags for `train_unsupervised.py`):
- `--epochs`: Training epochs (default: 60)
- `--batch-size`: Mini-batch size (default: 64)
- `--latent-dim`: Bottleneck dimension (default: 32)
- `--std-multiplier`: Anomaly threshold multiplier (default: 2.0)

**Isolation Forest hyperparameters** (in `train_unsupervised.py`):
- `n_estimators`: Number of trees (default: 200)
- `contamination`: Expected proportion of outliers in training set (default: 0.05)

---

## Troubleshooting

### "Unsupervised model not available"

**Cause:** The unsupervised artifacts have not been trained yet.

**Solution:** Run `python train_unsupervised.py` to train the autoencoder and isolation forest.

### "No authentic samples found"

**Cause:** `labels.csv` does not contain any rows with `label == 0`.

**Solution:** Ensure your `Audios/human/` directory contains audio files and run `python audio_pipeline.py` to regenerate `labels.csv`.

### High false positive rate

**Cause:** The anomaly threshold is too tight (too many authentic samples flagged).

**Solution:** Increase `--std-multiplier` when training:
```powershell
python train_unsupervised.py --std-multiplier 3.0
```

### High false negative rate

**Cause:** The anomaly threshold is too loose (deepfakes not flagged).

**Solution:** Decrease `--std-multiplier`:
```powershell
python train_unsupervised.py --std-multiplier 1.5
```

---

## Future Enhancements

1. **Multi-modal unsupervised learning:** Train separate autoencoders on MFCC features, spectrograms, and raw waveforms; ensemble their anomaly scores.

2. **Active learning:** Prioritize uncertain samples for human review based on model disagreement and anomaly severity.

3. **Continual learning:** Periodically retrain the supervised model on reviewed queue items to adapt to new deepfake techniques.

4. **Explainability:** Visualize which spectrogram regions contribute most to high reconstruction error.

5. **Calibrated probabilities:** Apply Platt scaling or isotonic regression to the supervised model output for better-calibrated uncertainty estimates.

---

## References

- **Autoencoder anomaly detection:** Sakurada & Yairi (2014), "Anomaly Detection Using Autoencoders with Nonlinear Dimensionality Reduction"
- **Isolation Forest:** Liu et al. (2008), "Isolation Forest"
- **Uncertainty quantification:** Gal & Ghahramani (2016), "Dropout as a Bayesian Approximation"
- **Zero-day deepfake detection:** Frank et al. (2020), "Leveraging Frequency Analysis for Deep Fake Image Recognition"

---

## Contact

For questions or issues with the hybrid architecture, please refer to:
- `project_specs/design.md` — Full system design document
- `project_specs/requirements.md` — Acceptance criteria
- `MODEL_TRAINING_AND_EVALUATION_GUIDE.md` — Training pipeline walkthrough
