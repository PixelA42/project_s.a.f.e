# Model Training and Evaluation Guide (Project S.A.F.E.)

This guide explains what is done, what is complete, and what to study next for model training and evaluation.

## 1. Current Status

Completed:
- Spectrogram dataset generation pipeline is in place.
- Labels file is generated (`labels.csv`).
- End-to-end training script is implemented (`train_model.py`).
- Three model artifacts are produced in `models/`:
  - `spectrogram_supervised_model.joblib`
  - `spectrogram_semisupervised_model.joblib`
  - `spectrogram_kmeans_model.joblib`
- Full metrics report is generated (`models/training_report.json`).
- Core ML tests in `tests/ml/` are passing.

## 2. Training Pipeline Flow

The training script does the following:
1. Read and validate `labels.csv`.
2. Split data by `original_audio` to reduce leakage across train/test.
3. Load spectrogram PNG files and convert to grayscale features.
4. Train supervised classifier (RandomForest).
5. Train unsupervised model (KMeans).
6. Train semi-supervised model with pseudo-labeling.
7. Save models and metrics report.

Run command:

```powershell
h:/Projects_AI/project_safe_v1/envSafeVone/Scripts/python.exe train_model.py
```

## 3. Metrics Generated

From `models/training_report.json`:

### Supervised
- Accuracy
- Macro F1
- Weighted F1
- Macro Precision
- Macro Recall
- Confusion Matrix
- Classification Report (per class + avg)
- Log Loss
- ROC AUC
- PR AUC

### Semi-supervised
- All supervised metrics above
- `pseudo_labels_added`
- `pseudo_threshold`
- `unlabeled_ratio`

### Unsupervised
- `n_clusters`
- `inertia`
- `silhouette_score`

## 4. Current Results Snapshot

- Supervised:
  - accuracy: 1.0000
  - macro_f1: 1.0000
  - roc_auc: 1.0000
- Semi-supervised:
  - accuracy: 0.9950
  - macro_f1: 0.9948
  - roc_auc: ~1.0000
  - pseudo_labels_added: 67
- Unsupervised:
  - n_clusters: 8
  - silhouette_score: 0.1022

## 5. How to Interpret These Results

- Very high supervised/semi-supervised metrics are good, but may indicate easy split conditions.
- The silhouette score (~0.10) suggests weakly separated clusters in unsupervised space.
- Confusion matrix is the most practical check for class-specific mistakes.

## 6. What Is Still Needed (Recommended Next)

Before final submission/demo:
1. Add strict metric gates to fail training if quality drops (for example, `macro_f1 < 0.75`).
2. Add repeatability checks by running training with multiple random seeds.
3. Add a hold-out folder test with truly unseen original audio sessions.
4. Add an inference script to load `spectrogram_supervised_model.joblib` and predict one file.
5. Connect model inference into `api.py` (currently mock response style).

## 7. Quick Validation Commands

Run ML tests:

```powershell
h:/Projects_AI/project_safe_v1/envSafeVone/Scripts/python.exe -m pytest tests/ml -v --tb=short
```

Open metrics report quickly:

```powershell
Get-Content models/training_report.json
```

## 8. Study Order (If You Are Preparing for Viva / Demo)

Read in this order:
1. `project_specs/SPEC.md` (project framing)
2. `project_specs/requirements.md` (acceptance criteria)
3. `project_specs/design.md` (architecture and components)
4. `train_model.py` (implemented training logic)
5. `models/training_report.json` (actual output evidence)
6. `tests/ml/*.py` (proof of behavioral checks)

## 9. Known Limitations Right Now

- `api.py` still uses mock-style output path and is not fully wired to these spectrogram model artifacts.
- The current classifier uses image flattening baseline features; stronger feature engineering could improve robustness.
- No calibration curve or threshold tuning report yet.

## 10. One-Line Project Summary

The model training and evaluation pipeline is implemented and producing complete metrics + artifacts; the main next step is production inference wiring into the API and final robustness checks.
