"""PyTorch spectral model evaluation with canonical report output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

from config import SETTINGS
from coreML.torch_inference import load_torch_spectral_model
from data_pipeline.deep_learning_loaders import SpectrogramDataset


def _legacy_metric_paths() -> list[Path]:
    paths = SETTINGS.paths
    return [
        paths.feature_pipeline_path,
        paths.supervised_model_path,
        paths.semisupervised_model_path,
        paths.unsupervised_model_path,
    ]


def remove_legacy_sklearn_artifacts() -> list[str]:
    removed: list[str] = []
    for path in _legacy_metric_paths():
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed


@torch.inference_mode()
def evaluate_spectral_torch_model(
    audio_paths: list[str],
    labels: list[int],
    report_path: str | Path | None = None,
    weights_path: str | Path | None = None,
    batch_size: int = 16,
) -> dict[str, object]:
    if not audio_paths:
        raise ValueError("audio_paths cannot be empty")
    if len(audio_paths) != len(labels):
        raise ValueError("audio_paths and labels length mismatch")

    report_file = Path(report_path or SETTINGS.paths.supervised_torch_eval_report_path)
    model = load_torch_spectral_model(weights_path=weights_path, device="cpu")
    dataset = SpectrogramDataset(audio_paths=audio_paths, labels=labels, return_path=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    y_true: list[int] = []
    y_prob: list[float] = []
    y_pred: list[int] = []

    for batch in loader:
        specs, batch_labels, _ = batch
        logits = model(specs.float())
        probs = torch.sigmoid(logits).flatten().cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        y_true.extend([int(v) for v in batch_labels.tolist()])
        y_prob.extend([float(v) for v in probs.tolist()])
        y_pred.extend([int(v) for v in preds.tolist()])

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    y_prob_np = np.asarray(y_prob, dtype=np.float32)

    report = {
        "framework": "pytorch",
        "model_path": str(weights_path or SETTINGS.paths.supervised_torch_weights_path),
        "metrics": {
            "supervised": {
                "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true_np, y_pred_np)),
                "macro_f1": float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0)),
                "roc_auc": float(roc_auc_score(y_true_np, y_prob_np)),
                "pr_auc": float(average_precision_score(y_true_np, y_prob_np)),
                "sample_count": int(len(y_true_np)),
            }
        },
    }

    removed_artifacts = remove_legacy_sklearn_artifacts()
    report["deprecated_artifacts_removed"] = removed_artifacts
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

