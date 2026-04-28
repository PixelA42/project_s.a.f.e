"""Unsupervised clustering orchestration for Project S.A.F.E."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from config import SETTINGS
from data_loader import load_data
from experiment_tracking import append_experiment_log_row, to_compact_json, utc_timestamp


GENERAL_CONFIG = SETTINGS.general
PATHS = SETTINGS.paths
TRAINING_CONFIG = SETTINGS.training


@dataclass(frozen=True)
class UnsupervisedRunResult:
    selected_k: int
    output_dir: Path
    kmeans_model_path: Path
    agglomerative_model_path: Path
    metrics_path: Path
    candidate_scores_path: Path
    labels_path: Path
    merged_dataset_path: Path
    artifact_index_path: Path
    k_diagnostics_plot_path: Path
    cluster_plot_path: Path
    projection_path: Path
    candidate_scores: pd.DataFrame
    labels: pd.DataFrame
    artifact_index: pd.DataFrame


def run_unsupervised(
    dataset_path: str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
    random_state: int = GENERAL_CONFIG.random_seed,
    k_min: int = 2,
    k_max: int = TRAINING_CONFIG.kmeans_max_clusters,
) -> UnsupervisedRunResult:
    loaded = load_data(dataset_path=dataset_path, random_state=random_state)
    output_path = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else PATHS.unsupervised_outputs_dir.resolve()
    )
    output_path.mkdir(parents=True, exist_ok=True)

    features = np.asarray(loaded.X_full, dtype=np.float32)
    candidate_scores, selection = _evaluate_k_candidates(
        features,
        k_min=k_min,
        k_max=k_max,
        random_state=random_state,
    )
    selected_k = int(selection["selected_k"])

    kmeans_model, kmeans_labels = _fit_kmeans(
        features,
        selected_k=selected_k,
        random_state=random_state,
    )
    agglomerative_model, agglomerative_labels = _fit_agglomerative(features, selected_k=selected_k)

    projection_2d = _project_for_visualization(features, random_state=random_state)

    candidate_scores_path = output_path / "candidate_k_scores.csv"
    labels_path = output_path / "cluster_labels.csv"
    merged_dataset_path = output_path / "dataset_with_clusters.csv"
    metrics_path = output_path / "run_summary.json"
    artifact_index_path = output_path / "artifact_index.csv"
    kmeans_model_path = output_path / "kmeans_model.joblib"
    agglomerative_model_path = output_path / "agglomerative_model.joblib"
    k_diagnostics_plot_path = output_path / "k_selection_diagnostics.png"
    cluster_plot_path = output_path / "pca_cluster_plot.png"
    projection_path = output_path / "pca_projection_2d.csv"

    candidate_scores.to_csv(candidate_scores_path, index=False)

    labels_frame = pd.DataFrame(
        {
            "row_index": np.arange(features.shape[0], dtype=int),
            "target": loaded.y_full,
            "kmeans_cluster": kmeans_labels,
            "agglomerative_cluster": agglomerative_labels,
        }
    )
    labels_frame.to_csv(labels_path, index=False)

    merged_frame = loaded.X_frame.copy()
    merged_frame.insert(0, "target", loaded.y_full)
    merged_frame.insert(0, "row_index", np.arange(len(merged_frame), dtype=int))
    merged_frame["kmeans_cluster"] = kmeans_labels
    merged_frame["agglomerative_cluster"] = agglomerative_labels
    merged_frame.to_csv(merged_dataset_path, index=False)

    projection_frame = pd.DataFrame(
        {
            "row_index": np.arange(features.shape[0], dtype=int),
            "pc1": projection_2d[:, 0],
            "pc2": projection_2d[:, 1],
            "kmeans_cluster": kmeans_labels,
            "agglomerative_cluster": agglomerative_labels,
        }
    )
    projection_frame.to_csv(projection_path, index=False)

    _plot_k_selection(candidate_scores, selected_k=selected_k, output_path=k_diagnostics_plot_path)
    _plot_cluster_projection(
        projection_2d,
        kmeans_labels=kmeans_labels,
        agglomerative_labels=agglomerative_labels,
        output_path=cluster_plot_path,
    )

    joblib.dump(kmeans_model, kmeans_model_path)
    joblib.dump(agglomerative_model, agglomerative_model_path)

    metrics_payload = {
        "dataset_path": str(loaded.dataset_path),
        "sample_count": int(features.shape[0]),
        "feature_count": int(features.shape[1]) if features.ndim == 2 else 0,
        "selected_k": selected_k,
        "selection": selection,
        "kmeans": {
            "inertia": float(kmeans_model.inertia_) if hasattr(kmeans_model, "inertia_") else None,
            "cluster_sizes": _cluster_sizes(kmeans_labels),
        },
        "agglomerative": {
            "cluster_sizes": _cluster_sizes(agglomerative_labels),
            "linkage": "ward" if selected_k > 1 else "not_fitted",
        },
        "artifacts": {
            "candidate_scores_csv": str(candidate_scores_path),
            "labels_csv": str(labels_path),
            "merged_dataset_csv": str(merged_dataset_path),
            "projection_csv": str(projection_path),
            "k_diagnostics_plot": str(k_diagnostics_plot_path),
            "cluster_plot": str(cluster_plot_path),
            "kmeans_model": str(kmeans_model_path),
            "agglomerative_model": str(agglomerative_model_path),
        },
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    artifact_rows = [
        {"artifact_type": "candidate_scores_csv", "path": str(candidate_scores_path)},
        {"artifact_type": "labels_csv", "path": str(labels_path)},
        {"artifact_type": "merged_dataset_csv", "path": str(merged_dataset_path)},
        {"artifact_type": "projection_csv", "path": str(projection_path)},
        {"artifact_type": "k_diagnostics_plot", "path": str(k_diagnostics_plot_path)},
        {"artifact_type": "cluster_plot", "path": str(cluster_plot_path)},
        {"artifact_type": "kmeans_model", "path": str(kmeans_model_path)},
        {"artifact_type": "agglomerative_model", "path": str(agglomerative_model_path)},
        {"artifact_type": "metrics_json", "path": str(metrics_path)},
    ]
    artifact_index = pd.DataFrame(artifact_rows).sort_values(by=["artifact_type", "path"]).reset_index(drop=True)
    artifact_index.to_csv(artifact_index_path, index=False)

    try:
        append_experiment_log_row(
            PATHS.experiment_log_path,
            {
                "timestamp": utc_timestamp(),
                "dataset_name": loaded.dataset_path.name,
                "model_name": "kmeans+agglomerative",
                "key_metrics": to_compact_json(
                    {
                        "selected_k": selected_k,
                        "silhouette_best_k": selection.get("silhouette_best_k"),
                        "elbow_best_k": selection.get("elbow_best_k"),
                        "silhouette_at_selected_k": selection.get("silhouette_at_selected_k"),
                        "inertia_at_selected_k": selection.get("inertia_at_selected_k"),
                    }
                ),
                "hyperparameters": to_compact_json(
                    {
                        "random_state": int(random_state),
                        "k_min": int(k_min),
                        "k_max": int(k_max),
                        "kmeans_n_init": str(TRAINING_CONFIG.kmeans_n_init),
                    }
                ),
            },
        )
    except Exception as exc:
        print(f"[warn] Failed to append experiment log: {exc}")

    return UnsupervisedRunResult(
        selected_k=selected_k,
        output_dir=output_path,
        kmeans_model_path=kmeans_model_path,
        agglomerative_model_path=agglomerative_model_path,
        metrics_path=metrics_path,
        candidate_scores_path=candidate_scores_path,
        labels_path=labels_path,
        merged_dataset_path=merged_dataset_path,
        artifact_index_path=artifact_index_path,
        k_diagnostics_plot_path=k_diagnostics_plot_path,
        cluster_plot_path=cluster_plot_path,
        projection_path=projection_path,
        candidate_scores=candidate_scores,
        labels=labels_frame,
        artifact_index=artifact_index,
    )


def _evaluate_k_candidates(
    features: np.ndarray,
    *,
    k_min: int,
    k_max: int,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sample_count = int(features.shape[0])
    if sample_count <= 1:
        frame = pd.DataFrame(
            [{"k": 1, "inertia": 0.0, "silhouette_score": np.nan, "elbow_distance": 0.0}]
        )
        selection = {
            "selected_k": 1,
            "silhouette_best_k": 1,
            "elbow_best_k": 1,
            "decision_rule": "fallback_single_sample",
            "silhouette_at_selected_k": None,
            "inertia_at_selected_k": 0.0,
        }
        return frame, selection

    lower_k = max(2, int(k_min))
    upper_k = min(int(k_max), sample_count - 1)
    if upper_k < lower_k:
        lower_k = 1
        upper_k = 1

    if lower_k == 1 and upper_k == 1:
        model = KMeans(n_clusters=1, random_state=random_state, n_init=TRAINING_CONFIG.kmeans_n_init)
        model.fit(features)
        frame = pd.DataFrame(
            [{"k": 1, "inertia": float(model.inertia_), "silhouette_score": np.nan, "elbow_distance": 0.0}]
        )
        selection = {
            "selected_k": 1,
            "silhouette_best_k": 1,
            "elbow_best_k": 1,
            "decision_rule": "fallback_insufficient_samples_for_multi_cluster",
            "silhouette_at_selected_k": None,
            "inertia_at_selected_k": float(model.inertia_),
        }
        return frame, selection

    rows: list[dict[str, Any]] = []
    for k_value in range(lower_k, upper_k + 1):
        kmeans = KMeans(
            n_clusters=int(k_value),
            random_state=random_state,
            n_init=TRAINING_CONFIG.kmeans_n_init,
        )
        labels = kmeans.fit_predict(features)
        silhouette = np.nan
        if k_value > 1 and sample_count > k_value:
            try:
                silhouette = float(silhouette_score(features, labels))
            except Exception:
                silhouette = np.nan
        rows.append(
            {
                "k": int(k_value),
                "inertia": float(kmeans.inertia_),
                "silhouette_score": silhouette,
            }
        )

    frame = pd.DataFrame(rows).sort_values(by="k").reset_index(drop=True)
    frame["elbow_distance"] = _elbow_distances(frame["k"].to_numpy(), frame["inertia"].to_numpy())

    silhouette_series = frame["silhouette_score"].replace([np.inf, -np.inf], np.nan)
    valid_silhouette = frame.loc[silhouette_series.notna(), ["k", "silhouette_score"]]
    if valid_silhouette.empty:
        silhouette_best_k = int(frame.iloc[0]["k"])
        silhouette_best_score = None
    else:
        best_row = valid_silhouette.loc[valid_silhouette["silhouette_score"].idxmax()]
        silhouette_best_k = int(best_row["k"])
        silhouette_best_score = float(best_row["silhouette_score"])

    elbow_best_k = int(frame.loc[frame["elbow_distance"].idxmax()]["k"])

    selected_k = silhouette_best_k
    decision_rule = "silhouette_primary"
    if silhouette_best_score is not None:
        near_optimal = set(
            frame.loc[
                frame["silhouette_score"].fillna(-np.inf) >= silhouette_best_score - 0.02,
                "k",
            ].astype(int)
        )
        if elbow_best_k in near_optimal:
            selected_k = elbow_best_k
            decision_rule = "silhouette_primary_elbow_tiebreak"

    selected_row = frame.loc[frame["k"] == selected_k].iloc[0]
    selection = {
        "selected_k": int(selected_k),
        "silhouette_best_k": int(silhouette_best_k),
        "elbow_best_k": int(elbow_best_k),
        "decision_rule": decision_rule,
        "silhouette_at_selected_k": (
            None if pd.isna(selected_row["silhouette_score"]) else float(selected_row["silhouette_score"])
        ),
        "inertia_at_selected_k": float(selected_row["inertia"]),
    }
    return frame, selection


def _elbow_distances(k_values: np.ndarray, inertia_values: np.ndarray) -> np.ndarray:
    if len(k_values) <= 2:
        return np.zeros(len(k_values), dtype=float)

    x = np.asarray(k_values, dtype=float)
    y = np.asarray(inertia_values, dtype=float)

    x_norm = (x - x.min()) / max(1e-12, (x.max() - x.min()))
    y_norm = (y - y.min()) / max(1e-12, (y.max() - y.min()))

    x1, y1 = float(x_norm[0]), float(y_norm[0])
    x2, y2 = float(x_norm[-1]), float(y_norm[-1])
    denominator = np.hypot(y2 - y1, x2 - x1)
    if denominator <= 1e-12:
        return np.zeros(len(k_values), dtype=float)

    distances = np.abs((y2 - y1) * x_norm - (x2 - x1) * y_norm + x2 * y1 - y2 * x1) / denominator
    return np.asarray(distances, dtype=float)


def _fit_kmeans(features: np.ndarray, *, selected_k: int, random_state: int) -> tuple[KMeans, np.ndarray]:
    model = KMeans(
        n_clusters=int(selected_k),
        random_state=random_state,
        n_init=TRAINING_CONFIG.kmeans_n_init,
    )
    labels = model.fit_predict(features)
    return model, np.asarray(labels, dtype=int)


def _fit_agglomerative(
    features: np.ndarray,
    *,
    selected_k: int,
) -> tuple[AgglomerativeClustering | dict[str, Any], np.ndarray]:
    if selected_k <= 1:
        labels = np.zeros(features.shape[0], dtype=int)
        placeholder = {
            "model_type": "agglomerative",
            "fitted": False,
            "reason": "selected_k<=1",
        }
        return placeholder, labels

    model = AgglomerativeClustering(n_clusters=int(selected_k), linkage="ward")
    labels = model.fit_predict(features)
    return model, np.asarray(labels, dtype=int)


def _project_for_visualization(features: np.ndarray, *, random_state: int) -> np.ndarray:
    sample_count, feature_count = int(features.shape[0]), int(features.shape[1])
    component_count = max(1, min(2, sample_count, feature_count))
    pca = PCA(n_components=component_count, random_state=random_state)
    projection = pca.fit_transform(features)
    if component_count == 1:
        projection = np.column_stack([projection[:, 0], np.zeros(sample_count, dtype=float)])
    return np.asarray(projection, dtype=float)


def _plot_k_selection(candidate_scores: pd.DataFrame, *, selected_k: int, output_path: Path) -> None:
    figure, axis_left = plt.subplots(figsize=(8, 5))
    axis_right = axis_left.twinx()

    axis_left.plot(
        candidate_scores["k"],
        candidate_scores["inertia"],
        marker="o",
        color="#2f6690",
        label="Inertia (Elbow)",
    )
    axis_right.plot(
        candidate_scores["k"],
        candidate_scores["silhouette_score"],
        marker="s",
        color="#cf5c36",
        label="Silhouette",
    )
    axis_left.axvline(selected_k, linestyle="--", color="#2a9d8f", label=f"Selected K={selected_k}")

    axis_left.set_xlabel("K")
    axis_left.set_ylabel("Inertia")
    axis_right.set_ylabel("Silhouette Score")
    axis_left.set_title("K Selection: Elbow and Silhouette")

    left_handles, left_labels = axis_left.get_legend_handles_labels()
    right_handles, right_labels = axis_right.get_legend_handles_labels()
    axis_left.legend(left_handles + right_handles, left_labels + right_labels, loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_cluster_projection(
    projection_2d: np.ndarray,
    *,
    kmeans_labels: np.ndarray,
    agglomerative_labels: np.ndarray,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    scatter_1 = axes[0].scatter(
        projection_2d[:, 0],
        projection_2d[:, 1],
        c=kmeans_labels,
        cmap="tab10",
        alpha=0.85,
        s=20,
    )
    axes[0].set_title("PCA-2D: KMeans Clusters")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    figure.colorbar(scatter_1, ax=axes[0], fraction=0.046, pad=0.04)

    scatter_2 = axes[1].scatter(
        projection_2d[:, 0],
        projection_2d[:, 1],
        c=agglomerative_labels,
        cmap="tab10",
        alpha=0.85,
        s=20,
    )
    axes[1].set_title("PCA-2D: Agglomerative Clusters")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    figure.colorbar(scatter_2, ax=axes[1], fraction=0.046, pad=0.04)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _cluster_sizes(labels: np.ndarray) -> dict[str, int]:
    series = pd.Series(np.asarray(labels, dtype=int))
    return {str(int(label)): int(count) for label, count in series.value_counts().sort_index().items()}


__all__ = ["UnsupervisedRunResult", "run_unsupervised", "_evaluate_k_candidates"]