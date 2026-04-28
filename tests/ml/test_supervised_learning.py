from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import supervised_learning


def test_run_supervised_classification_writes_metrics_and_confusion_outputs(tmp_path, monkeypatch):
    dataset_path = tmp_path / "classification.csv"
    row_count = 30
    dataframe = pd.DataFrame(
        {
            "target": ["synthetic" if index % 2 else "real" for index in range(row_count)],
            "duration_seconds": np.linspace(0.5, 2.5, row_count),
            "sample_rate": [16000 if index % 2 == 0 else 22050 for index in range(row_count)],
            "source_family": ["real_samples" if index % 2 == 0 else "OpenAI" for index in range(row_count)],
        }
    )
    dataframe.to_csv(dataset_path, index=False)

    patched_paths = type(
        "Paths",
        (),
        {"experiment_log_path": tmp_path / "outputs" / "experiment_log.csv"},
    )()
    monkeypatch.setattr(supervised_learning, "PATHS", patched_paths)

    result = supervised_learning.run_supervised(
        dataset_path=dataset_path,
        output_dir=tmp_path / "outputs",
        cv_folds=3,
        model_names=["logistic_regression", "random_forest"],
    )

    assert result.task_type == "classification"
    assert result.target_column == "target"
    assert result.metrics_path.is_file()
    assert result.fold_metrics_path.is_file()
    assert result.artifact_index_path.is_file()
    assert set(result.metrics["model"]) == {"logistic_regression", "random_forest"}

    random_forest_dir = result.output_dir / "random_forest"
    assert (random_forest_dir / "confusion_matrix.csv").is_file()
    assert (random_forest_dir / "confusion_matrix.png").is_file()
    assert (random_forest_dir / "feature_importances.csv").is_file()
    assert (random_forest_dir / "feature_importances.png").is_file()
    assert (random_forest_dir / "oof_predictions.csv").is_file()

    experiment_log = patched_paths.experiment_log_path
    assert experiment_log.is_file()
    log_frame = pd.read_csv(experiment_log)
    assert len(log_frame) == 1
    assert set(log_frame.columns) == {
        "timestamp",
        "dataset_name",
        "model_name",
        "key_metrics",
        "hyperparameters",
    }
    assert log_frame.iloc[0]["dataset_name"] == dataset_path.name
    assert log_frame.iloc[0]["model_name"] in {"logistic_regression", "random_forest"}
    assert isinstance(json.loads(log_frame.iloc[0]["key_metrics"]), dict)
    assert isinstance(json.loads(log_frame.iloc[0]["hyperparameters"]), dict)


def test_run_supervised_regression_detects_numeric_target_and_writes_diagnostics(tmp_path):
    dataset_path = tmp_path / "regression.csv"
    row_count = 36
    durations = np.linspace(0.2, 3.2, row_count)
    sample_rates = np.where(np.arange(row_count) % 2 == 0, 16000, 22050)
    dataframe = pd.DataFrame(
        {
            "target": 0.75 * durations + (sample_rates / 100000.0) + np.linspace(0.0, 0.2, row_count),
            "duration_seconds": durations,
            "sample_rate": sample_rates,
            "voice_type": ["a" if index % 3 == 0 else "b" if index % 3 == 1 else "c" for index in range(row_count)],
        }
    )
    dataframe.to_csv(dataset_path, index=False)

    result = supervised_learning.run_supervised(
        dataset_path=dataset_path,
        output_dir=tmp_path / "outputs_regression",
        cv_folds=4,
        model_names=["linear_regression", "random_forest_regressor"],
    )

    assert result.task_type == "regression"
    assert set(result.metrics["model"]) == {"linear_regression", "random_forest_regressor"}
    assert "r2" in result.metrics.columns
    assert "rmse" in result.fold_metrics.columns

    linear_dir = result.output_dir / "linear_regression"
    assert (linear_dir / "regression_diagnostics.png").is_file()
    assert (linear_dir / "residuals.csv").is_file()
    assert (linear_dir / "feature_importances.csv").is_file()
    assert (linear_dir / "feature_importances.png").is_file()
