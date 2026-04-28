from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

import unsupervised_learning


def test_evaluate_k_candidates_combines_silhouette_and_elbow():
    features, _ = make_blobs(
        n_samples=120,
        centers=3,
        n_features=5,
        cluster_std=0.65,
        random_state=42,
    )

    candidate_scores, selection = unsupervised_learning._evaluate_k_candidates(
        np.asarray(features, dtype=np.float32),
        k_min=2,
        k_max=6,
        random_state=42,
    )

    assert set(candidate_scores.columns) == {"k", "inertia", "silhouette_score", "elbow_distance"}
    assert 2 <= int(selection["selected_k"]) <= 6
    assert int(selection["silhouette_best_k"]) in candidate_scores["k"].tolist()
    assert int(selection["elbow_best_k"]) in candidate_scores["k"].tolist()


def test_evaluate_k_candidates_single_sample_fallback():
    candidate_scores, selection = unsupervised_learning._evaluate_k_candidates(
        np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        k_min=2,
        k_max=8,
        random_state=42,
    )

    assert len(candidate_scores) == 1
    assert int(selection["selected_k"]) == 1
    assert selection["decision_rule"] == "fallback_single_sample"


def test_run_unsupervised_writes_artifacts_and_labels(tmp_path, monkeypatch):
    row_count = 72
    dataframe = pd.DataFrame(
        {
            "target": ["synthetic" if idx % 2 else "real" for idx in range(row_count)],
            "feature_a": np.linspace(0.0, 12.0, row_count),
            "feature_b": np.sin(np.linspace(0.0, 4.0, row_count)),
            "feature_c": np.cos(np.linspace(0.0, 6.0, row_count)),
            "source_family": ["openai" if idx % 2 else "real_samples" for idx in range(row_count)],
        }
    )
    dataset_path = tmp_path / "unsupervised_dataset.csv"
    dataframe.to_csv(dataset_path, index=False)

    patched_paths = type(
        "Paths",
        (),
        {
            "experiment_log_path": tmp_path / "outputs" / "experiment_log.csv",
            "unsupervised_outputs_dir": tmp_path / "outputs" / "unsupervised",
        },
    )()
    monkeypatch.setattr(unsupervised_learning, "PATHS", patched_paths)

    result = unsupervised_learning.run_unsupervised(
        dataset_path=dataset_path,
        output_dir=tmp_path / "outputs" / "unsupervised_run",
        random_state=42,
        k_min=2,
        k_max=6,
    )

    assert result.candidate_scores_path.is_file()
    assert result.labels_path.is_file()
    assert result.merged_dataset_path.is_file()
    assert result.metrics_path.is_file()
    assert result.kmeans_model_path.is_file()
    assert result.agglomerative_model_path.is_file()
    assert result.k_diagnostics_plot_path.is_file()
    assert result.cluster_plot_path.is_file()
    assert result.projection_path.is_file()
    assert result.artifact_index_path.is_file()

    labels_frame = pd.read_csv(result.labels_path)
    assert len(labels_frame) == row_count
    assert {"row_index", "target", "kmeans_cluster", "agglomerative_cluster"}.issubset(labels_frame.columns)

    projection_frame = pd.read_csv(result.projection_path)
    assert {"pc1", "pc2", "kmeans_cluster", "agglomerative_cluster"}.issubset(projection_frame.columns)

    summary = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert "selection" in summary
    assert "decision_rule" in summary["selection"]
    assert int(summary["selected_k"]) >= 1

    experiment_log = pd.read_csv(patched_paths.experiment_log_path)
    assert len(experiment_log) == 1
    assert experiment_log.iloc[0]["dataset_name"] == dataset_path.name
    assert experiment_log.iloc[0]["model_name"] == "kmeans+agglomerative"
