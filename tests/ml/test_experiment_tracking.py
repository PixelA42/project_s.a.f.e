from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiment_tracking import append_experiment_log_row


def test_append_experiment_log_row_creates_file_with_headers(tmp_path):
    log_path = tmp_path / "outputs" / "experiment_log.csv"

    append_experiment_log_row(
        log_path,
        {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "dataset_name": "dataset.csv",
            "model_name": "random_forest",
            "key_metrics": '{"accuracy":0.9}',
            "hyperparameters": '{"cv_folds":5}',
        },
    )

    assert log_path.is_file()
    frame = pd.read_csv(log_path)
    assert list(frame.columns) == [
        "timestamp",
        "dataset_name",
        "model_name",
        "key_metrics",
        "hyperparameters",
    ]
    assert len(frame) == 1


def test_append_experiment_log_row_appends_without_rewriting_headers(tmp_path):
    log_path = tmp_path / "outputs" / "experiment_log.csv"

    append_experiment_log_row(
        log_path,
        {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "dataset_name": "dataset_a.csv",
            "model_name": "model_a",
            "key_metrics": "{}",
            "hyperparameters": "{}",
        },
    )
    append_experiment_log_row(
        log_path,
        {
            "timestamp": "2026-01-01T00:00:01+00:00",
            "dataset_name": "dataset_b.csv",
            "model_name": "model_b",
            "key_metrics": "{}",
            "hyperparameters": "{}",
        },
    )

    frame = pd.read_csv(log_path)
    assert len(frame) == 2
    assert frame["dataset_name"].tolist() == ["dataset_a.csv", "dataset_b.csv"]
