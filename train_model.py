"""Train Project S.A.F.E. image-based baseline models from spectrogram labels.

This script supports:
- Supervised training (RandomForest)
- Unsupervised clustering (KMeans)
- Semi-supervised training via pseudo-labeling
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
	accuracy_score,
	average_precision_score,
	balanced_accuracy_score,
	classification_report,
	confusion_matrix,
	f1_score,
	log_loss,
	precision_score,
	recall_score,
	roc_auc_score,
	silhouette_score,
)
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import SETTINGS


GENERAL_CONFIG = SETTINGS.general
TRAINING_CONFIG = SETTINGS.training
PATHS = SETTINGS.paths
AUDIO_CONFIG = SETTINGS.audio


def _to_python_scalar(value: Any) -> Any:
	if isinstance(value, np.generic):
		return value.item()
	return value


def _validate_fraction(
	value: float,
	name: str,
	*,
	min_inclusive: float = 0.0,
	max_inclusive: float = 1.0,
	max_exclusive: bool = False,
) -> None:
	if value < min_inclusive:
		raise ValueError(f"{name} must be >= {min_inclusive}. Received: {value}")
	if max_exclusive:
		if value >= max_inclusive:
			raise ValueError(f"{name} must be < {max_inclusive}. Received: {value}")
	elif value > max_inclusive:
		raise ValueError(f"{name} must be <= {max_inclusive}. Received: {value}")


def _can_stratify(labels: np.ndarray) -> bool:
	if labels.size < 2:
		return False
	label_counts = pd.Series(labels).value_counts()
	return label_counts.size > 1 and int(label_counts.min()) >= 2


def _ordered_labels(*label_sets: np.ndarray) -> list[Any]:
	ordered: list[Any] = []
	for label_array in label_sets:
		for label in np.asarray(label_array).tolist():
			label = _to_python_scalar(label)
			if label not in ordered:
				ordered.append(label)
	return ordered


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Train S.A.F.E. baseline models from spectrogram labels."
	)
	parser.add_argument(
		"--labels",
		default=str(PATHS.labels_csv_path),
		help="Path to labels CSV file",
	)
	parser.add_argument(
		"--models-dir",
		default=str(PATHS.models_dir),
		help="Directory to write model artifacts",
	)
	parser.add_argument(
		"--report-path",
		default=None,
		help="Path to write training report JSON",
	)
	parser.add_argument(
		"--supervised-model-path",
		default=None,
		help="Path to write the supervised model artifact",
	)
	parser.add_argument(
		"--semisupervised-model-path",
		default=None,
		help="Path to write the semi-supervised model artifact",
	)
	parser.add_argument(
		"--unsupervised-model-path",
		default=None,
		help="Path to write the unsupervised model artifact",
	)
	parser.add_argument(
		"--feature-pipeline-path",
		default=None,
		help="Path to write the scaler/PCA feature pipeline artifact",
	)
	parser.add_argument(
		"--image-size",
		type=int,
		default=AUDIO_CONFIG.feature_image_size,
		help="Square resize target used for feature extraction",
	)
	parser.add_argument(
		"--test-size",
		type=float,
		default=TRAINING_CONFIG.test_split_ratio,
		help="Held-out split size",
	)
	parser.add_argument(
		"--pseudo-threshold",
		type=float,
		default=TRAINING_CONFIG.pseudo_threshold,
		help="Confidence threshold used for pseudo-label acceptance",
	)
	parser.add_argument(
		"--semi-unlabeled-ratio",
		type=float,
		default=TRAINING_CONFIG.semi_unlabeled_ratio,
		help="Fraction of training split treated as unlabeled in semi-supervised stage",
	)
	parser.add_argument("--random-state", type=int, default=GENERAL_CONFIG.random_seed)
	parser.add_argument(
		"--pca-components",
		type=int,
		default=TRAINING_CONFIG.pca_components,
		help="Target PCA dimensionality for training features",
	)
	parser.add_argument(
		"--embedding-method",
		choices=["tsne", "umap", "none"],
		default=TRAINING_CONFIG.embedding_method,
		help="2D manifold projection method used for plotting",
	)
	parser.add_argument(
		"--embedding-path",
		default=None,
		help="Path to save per-sample 2D embedding values",
	)
	parser.add_argument(
		"--embedding-plot-path",
		default=None,
		help="Path to save 2D embedding scatter plot",
	)
	return parser.parse_args()


def _resolve_model_output_path(raw_path: str | None, models_dir: Path, default_path: Path) -> Path:
	if raw_path:
		return Path(raw_path)
	return models_dir / default_path.name


def _validate_dataframe(df: pd.DataFrame) -> None:
	required_columns = {"file_path", "label", "original_audio"}
	missing = required_columns.difference(df.columns)
	if missing:
		raise ValueError(f"Labels CSV is missing required columns: {sorted(missing)}")


def _resize_grayscale_image(img: np.ndarray, size: int) -> np.ndarray:
	if img.ndim == 3:
		img = img.mean(axis=2)
	if img.ndim != 2:
		raise ValueError("Unexpected image shape after grayscale conversion")

	height, width = img.shape
	row_idx = np.linspace(0, max(0, height - 1), size).astype(int)
	col_idx = np.linspace(0, max(0, width - 1), size).astype(int)
	return img[np.ix_(row_idx, col_idx)]


def _extract_image_features(file_paths: np.ndarray, image_size: int) -> np.ndarray:
	features: list[np.ndarray] = []
	for raw_path in file_paths:
		path = Path(str(raw_path))
		if not path.is_file():
			raise FileNotFoundError(f"Spectrogram image not found: {path}")
		image = mpimg.imread(path)
		image = _resize_grayscale_image(np.asarray(image, dtype=np.float32), image_size)
		features.append(image.flatten())
	return np.vstack(features)


def _fit_pca_projection(
	x_train: np.ndarray,
	x_test: np.ndarray,
	pca_components: int,
	random_state: int,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, PCA, dict[str, Any]]:
	if pca_components < 2:
		raise ValueError("--pca-components must be >= 2")

	scaler = StandardScaler()
	x_train_scaled = scaler.fit_transform(x_train)
	x_test_scaled = scaler.transform(x_test)

	max_components = min(x_train_scaled.shape[0], x_train_scaled.shape[1])
	if max_components < 2:
		raise ValueError(
			"Not enough training samples/features for PCA. Need at least 2 effective components."
		)
	effective_components = max(2, min(pca_components, max_components))

	pca = PCA(n_components=effective_components, random_state=random_state)
	x_train_reduced = pca.fit_transform(x_train_scaled)
	x_test_reduced = pca.transform(x_test_scaled)

	metrics = {
		"input_dimensions": int(x_train.shape[1]),
		"pca_components_requested": int(pca_components),
		"pca_components_used": int(effective_components),
		"explained_variance_ratio": float(np.sum(pca.explained_variance_ratio_)),
	}
	return x_train_reduced, x_test_reduced, scaler, pca, metrics


def _compute_2d_embedding(
	x_reduced: np.ndarray,
	method: str,
	random_state: int,
) -> tuple[np.ndarray, dict[str, Any]]:
	if method == "none":
		return np.empty((x_reduced.shape[0], 0), dtype=np.float32), {"method": "none"}

	if method == "umap":
		try:
			import umap
		except Exception as exc:
			raise ImportError(
				"UMAP selected but package not installed. Install with: pip install umap-learn"
			) from exc
		reducer = umap.UMAP(n_components=2, random_state=random_state)
		embedding = reducer.fit_transform(x_reduced)
		return embedding.astype(np.float32, copy=False), {"method": "umap"}

	n_samples = x_reduced.shape[0]
	if n_samples < 4:
		raise ValueError(
			"Need at least 4 samples to compute t-SNE embedding. Use --embedding-method none or umap."
		)
	perplexity = min(
		TRAINING_CONFIG.tsne_perplexity_cap,
		max(3, n_samples // 4),
		n_samples - 1,
	)
	reducer = TSNE(
		n_components=2,
		random_state=random_state,
		perplexity=perplexity,
		init="pca",
		learning_rate="auto",
	)
	embedding = reducer.fit_transform(x_reduced)
	return embedding.astype(np.float32, copy=False), {
		"method": "tsne",
		"perplexity": float(perplexity),
	}


def _save_embedding_outputs(
	embedding: np.ndarray,
	labels: np.ndarray,
	split: np.ndarray,
	method: str,
	embedding_path: Path,
	embedding_plot_path: Path,
) -> None:
	if embedding.shape[0] == 0 or embedding.shape[1] != 2:
		return

	embedding_path.parent.mkdir(parents=True, exist_ok=True)
	embedding_df = pd.DataFrame(
		{
			"x": embedding[:, 0],
			"y": embedding[:, 1],
			"label": labels,
			"split": split,
		}
	)
	embedding_df.to_csv(embedding_path, index=False)

	fig, ax = plt.subplots(figsize=TRAINING_CONFIG.embedding_plot_size)
	for label in sorted(pd.unique(labels)):
		label_mask = labels == label
		ax.scatter(
			embedding[label_mask, 0],
			embedding[label_mask, 1],
			label=str(label),
			alpha=TRAINING_CONFIG.embedding_plot_alpha,
			s=TRAINING_CONFIG.embedding_plot_marker_size,
		)
	ax.set_title(f"2D Feature Projection ({method.upper()} on PCA features)")
	ax.set_xlabel("Component 1")
	ax.set_ylabel("Component 2")
	ax.legend(loc="best")
	ax.grid(alpha=TRAINING_CONFIG.embedding_grid_alpha)
	fig.tight_layout()
	embedding_plot_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(embedding_plot_path, dpi=TRAINING_CONFIG.embedding_plot_dpi)
	plt.close(fig)


def _split_without_leakage(
	df: pd.DataFrame,
	test_size: float,
	random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
	_validate_fraction(test_size, "--test-size", min_inclusive=0.05, max_inclusive=0.95)

	grouped = (
		df.groupby("original_audio", as_index=False)
		.agg(label=("label", "first"), sample_count=("file_path", "count"))
		.copy()
	)
	label_conflicts = df.groupby("original_audio")["label"].nunique()
	conflicting_groups = label_conflicts[label_conflicts > 1]
	if not conflicting_groups.empty:
		raise ValueError(
			"Each original_audio must map to exactly one label before splitting. "
			f"Conflicting groups: {conflicting_groups.index.tolist()[:5]}"
		)

	if len(grouped) < 2:
		raise ValueError("Need at least 2 unique original_audio groups to create a train/test split.")

	group_labels = grouped["label"].to_numpy()
	unique_files = grouped["original_audio"].to_numpy()
	stratify_labels = group_labels if _can_stratify(group_labels) else None
	split_strategy = "group_random"
	split_note = None
	try:
		train_files, test_files = train_test_split(
			unique_files,
			test_size=test_size,
			random_state=random_state,
			stratify=stratify_labels,
		)
		if stratify_labels is not None:
			split_strategy = "group_stratified"
	except ValueError as exc:
		if stratify_labels is None:
			raise
		train_files, test_files = train_test_split(
			unique_files,
			test_size=test_size,
			random_state=random_state,
			stratify=None,
		)
		split_strategy = "group_random_fallback"
		split_note = str(exc)

	train_df = df[df["original_audio"].isin(train_files)].copy()
	test_df = df[df["original_audio"].isin(test_files)].copy()

	if train_df.empty or test_df.empty:
		raise ValueError("Train/test split produced an empty partition. Adjust --test-size or dataset size.")

	train_group_counts = grouped[grouped["original_audio"].isin(train_files)]["label"].value_counts()
	test_group_counts = grouped[grouped["original_audio"].isin(test_files)]["label"].value_counts()
	split_metrics = {
		"split_level": "original_audio",
		"strategy": split_strategy,
		"group_count_total": int(len(grouped)),
		"group_count_train": int(len(train_files)),
		"group_count_test": int(len(test_files)),
		"group_label_distribution_train": {
			str(_to_python_scalar(label)): int(count)
			for label, count in train_group_counts.items()
		},
		"group_label_distribution_test": {
			str(_to_python_scalar(label)): int(count)
			for label, count in test_group_counts.items()
		},
	}
	if split_note is not None:
		split_metrics["note"] = split_note

	return train_df, test_df, split_metrics


def _train_supervised(
	x_train: np.ndarray,
	y_train: np.ndarray,
	x_test: np.ndarray,
	y_test: np.ndarray,
	random_state: int,
	n_estimators: int = TRAINING_CONFIG.random_forest_n_estimators,
	n_jobs: int = TRAINING_CONFIG.random_forest_n_jobs,
) -> tuple[RandomForestClassifier, dict[str, Any]]:
	model = RandomForestClassifier(
		n_estimators=n_estimators,
		random_state=random_state,
		n_jobs=n_jobs,
	)
	model.fit(x_train, y_train)
	metrics = _compute_classification_metrics(model, x_test, y_test)
	return model, metrics


def _compute_classification_metrics(
	model: RandomForestClassifier,
	x_test: np.ndarray,
	y_test: np.ndarray,
) -> dict[str, Any]:
	preds = model.predict(x_test)
	probs = model.predict_proba(x_test) if hasattr(model, "predict_proba") else None
	classes = np.asarray(getattr(model, "classes_", np.unique(y_test)))
	label_order = _ordered_labels(classes, y_test, preds)
	metric_notes: list[str] = []
	class_support = {str(label): int(np.sum(y_test == label)) for label in label_order}
	prediction_distribution = {str(label): int(np.sum(preds == label)) for label in label_order}

	metrics: dict[str, Any] = {
		"class_labels": label_order,
		"support_by_class": class_support,
		"prediction_distribution": prediction_distribution,
		"accuracy": float(accuracy_score(y_test, preds)),
		"balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
		"macro_f1": float(f1_score(y_test, preds, average="macro", zero_division=0)),
		"weighted_f1": float(f1_score(y_test, preds, average="weighted", zero_division=0)),
		"macro_precision": float(
			precision_score(y_test, preds, average="macro", zero_division=0)
		),
		"macro_recall": float(
			recall_score(y_test, preds, average="macro", zero_division=0)
		),
		"confusion_matrix": confusion_matrix(y_test, preds, labels=label_order).tolist(),
		"classification_report": classification_report(
			y_test,
			preds,
			labels=label_order,
			output_dict=True,
			zero_division=0,
		),
		"log_loss": None,
		"roc_auc": None,
		"pr_auc": None,
	}

	if probs is not None:
		test_classes = _ordered_labels(y_test)
		if set(test_classes).issubset(set(label_order)) and probs.shape[1] == len(classes):
			try:
				metrics["log_loss"] = float(log_loss(y_test, probs, labels=classes))
			except Exception as exc:
				metric_notes.append(f"log_loss unavailable: {exc}")
		else:
			metric_notes.append(
				"log_loss skipped because test labels were not fully represented in the probability output."
			)

		if probs.shape[1] == 2 and np.unique(y_test).size == 2 and classes.size == 2:
			positive_class = 1 if 1 in classes else _to_python_scalar(classes[-1])
			metrics["binary_positive_class"] = positive_class
			positive_index = int(np.where(classes == positive_class)[0][0])
			positive_probs = probs[:, positive_index]
			try:
				metrics["roc_auc"] = float(roc_auc_score(y_test, positive_probs))
			except Exception as exc:
				metric_notes.append(f"roc_auc unavailable: {exc}")
			try:
				metrics["pr_auc"] = float(average_precision_score(y_test, positive_probs))
			except Exception as exc:
				metric_notes.append(f"pr_auc unavailable: {exc}")
		else:
			metric_notes.append(
				"ROC AUC and PR AUC skipped because binary probability outputs for both classes were not available."
			)
	else:
		metric_notes.append("Probability-based metrics skipped because predict_proba is unavailable.")

	metrics["metric_notes"] = metric_notes

	return metrics


def _train_unsupervised(
	x_train: np.ndarray,
	random_state: int,
	max_clusters: int = TRAINING_CONFIG.kmeans_max_clusters,
	n_init: str = TRAINING_CONFIG.kmeans_n_init,
) -> tuple[KMeans, dict[str, Any]]:
	if x_train.shape[0] == 0:
		raise ValueError("Need at least 1 training sample for unsupervised clustering.")

	n_clusters = max(1, min(max_clusters, x_train.shape[0]))
	kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
	kmeans.fit(x_train)
	cluster_ids = kmeans.labels_
	silhouette = None
	if n_clusters > 1 and x_train.shape[0] > n_clusters:
		try:
			silhouette = float(silhouette_score(x_train, cluster_ids))
		except Exception:
			silhouette = None

	metrics = {
		"n_clusters": int(n_clusters),
		"inertia": float(kmeans.inertia_),
		"silhouette_score": silhouette,
	}
	return kmeans, metrics


def _train_semisupervised(
	base_model: RandomForestClassifier,
	x_train: np.ndarray,
	y_train: np.ndarray,
	x_test: np.ndarray,
	y_test: np.ndarray,
	unlabeled_ratio: float,
	threshold: float,
	random_state: int,
) -> tuple[RandomForestClassifier, dict[str, Any]]:
	_validate_fraction(unlabeled_ratio, "--semi-unlabeled-ratio", max_inclusive=1.0, max_exclusive=True)
	_validate_fraction(threshold, "--pseudo-threshold")

	indices = np.arange(x_train.shape[0])
	if unlabeled_ratio == 0 or x_train.shape[0] < 2:
		labeled_idx = indices
		unlabeled_idx = np.array([], dtype=int)
	else:
		stratify_labels = y_train if _can_stratify(y_train) else None
		try:
			labeled_idx, unlabeled_idx = train_test_split(
				indices,
				test_size=unlabeled_ratio,
				random_state=random_state,
				stratify=stratify_labels,
			)
		except ValueError:
			labeled_idx, unlabeled_idx = train_test_split(
				indices,
				test_size=unlabeled_ratio,
				random_state=random_state,
				stratify=None,
			)

	x_seed = x_train[labeled_idx]
	y_seed = y_train[labeled_idx]
	x_unlabeled = x_train[unlabeled_idx]

	semi_model = clone(base_model)
	semi_model.fit(x_seed, y_seed)

	pseudo_added = 0
	if x_unlabeled.size > 0 and hasattr(semi_model, "predict_proba"):
		probs = semi_model.predict_proba(x_unlabeled)
		confidence = probs.max(axis=1)
		accepted_mask = confidence >= threshold
		if np.any(accepted_mask):
			pseudo_labels = semi_model.predict(x_unlabeled[accepted_mask])
			x_aug = np.vstack([x_seed, x_unlabeled[accepted_mask]])
			y_aug = np.concatenate([y_seed, pseudo_labels])
			pseudo_added = int(np.sum(accepted_mask))
			semi_model.fit(x_aug, y_aug)

	metrics = _compute_classification_metrics(semi_model, x_test, y_test)
	metrics.update({
		"pseudo_labels_added": pseudo_added,
		"pseudo_threshold": float(threshold),
		"unlabeled_ratio": float(unlabeled_ratio),
		"seed_samples": int(x_seed.shape[0]),
		"unlabeled_pool_samples": int(x_unlabeled.shape[0]),
		"seed_label_distribution": {
			str(_to_python_scalar(label)): int(count)
			for label, count in pd.Series(y_seed).value_counts().items()
		},
	})
	return semi_model, metrics


def main() -> None:
	args = parse_args()
	labels_path = Path(args.labels)
	models_dir = Path(args.models_dir)
	report_path = _resolve_model_output_path(args.report_path, models_dir, PATHS.training_report_path)
	embedding_path = _resolve_model_output_path(
		args.embedding_path,
		models_dir,
		PATHS.embedding_csv_path,
	)
	embedding_plot_path = _resolve_model_output_path(
		args.embedding_plot_path,
		models_dir,
		PATHS.embedding_plot_path,
	)
	supervised_path = _resolve_model_output_path(
		args.supervised_model_path,
		models_dir,
		PATHS.supervised_model_path,
	)
	semisupervised_path = _resolve_model_output_path(
		args.semisupervised_model_path,
		models_dir,
		PATHS.semisupervised_model_path,
	)
	unsupervised_path = _resolve_model_output_path(
		args.unsupervised_model_path,
		models_dir,
		PATHS.unsupervised_model_path,
	)
	feature_pipeline_path = _resolve_model_output_path(
		args.feature_pipeline_path,
		models_dir,
		PATHS.feature_pipeline_path,
	)

	if not labels_path.is_file():
		raise FileNotFoundError(f"labels CSV not found: {labels_path}")

	df = pd.read_csv(labels_path)
	_validate_dataframe(df)

	train_df, test_df, split_metrics = _split_without_leakage(df, args.test_size, args.random_state)

	print("=" * 50)
	print("DATASET SPLIT COMPLETE")
	print("=" * 50)
	print(f"Total Images    : {len(df)}")
	print(f"Training Images : {len(train_df)}")
	print(f"Testing Images  : {len(test_df)}")

	x_train_paths = train_df["file_path"].values
	y_train = train_df["label"].values
	x_test_paths = test_df["file_path"].values
	y_test = test_df["label"].values

	x_train = _extract_image_features(x_train_paths, args.image_size)
	x_test = _extract_image_features(x_test_paths, args.image_size)
	x_train_reduced, x_test_reduced, scaler, pca, reduction_metrics = _fit_pca_projection(
		x_train,
		x_test,
		args.pca_components,
		args.random_state,
	)

	all_reduced = np.vstack([x_train_reduced, x_test_reduced])
	all_labels = np.concatenate([y_train, y_test])
	all_splits = np.concatenate(
		[
			np.full(shape=len(y_train), fill_value="train", dtype=object),
			np.full(shape=len(y_test), fill_value="test", dtype=object),
		]
	)
	embedding_2d, embedding_metrics = _compute_2d_embedding(
		all_reduced,
		args.embedding_method,
		args.random_state,
	)
	_save_embedding_outputs(
		embedding_2d,
		all_labels,
		all_splits,
		args.embedding_method,
		embedding_path,
		embedding_plot_path,
	)

	supervised_model, supervised_metrics = _train_supervised(
		x_train_reduced,
		y_train,
		x_test_reduced,
		y_test,
		args.random_state,
		TRAINING_CONFIG.random_forest_n_estimators,
		TRAINING_CONFIG.random_forest_n_jobs,
	)
	kmeans_model, unsupervised_metrics = _train_unsupervised(
		x_train_reduced,
		args.random_state,
		TRAINING_CONFIG.kmeans_max_clusters,
		TRAINING_CONFIG.kmeans_n_init,
	)
	semi_model, semisupervised_metrics = _train_semisupervised(
		supervised_model,
		x_train_reduced,
		y_train,
		x_test_reduced,
		y_test,
		args.semi_unlabeled_ratio,
		args.pseudo_threshold,
		args.random_state,
	)

	models_dir.mkdir(parents=True, exist_ok=True)
	for output_path in (
		supervised_path,
		semisupervised_path,
		unsupervised_path,
		feature_pipeline_path,
	):
		output_path.parent.mkdir(parents=True, exist_ok=True)

	joblib.dump(supervised_model, supervised_path)
	joblib.dump(semi_model, semisupervised_path)
	joblib.dump(kmeans_model, unsupervised_path)
	joblib.dump({"scaler": scaler, "pca": pca}, feature_pipeline_path)

	report_path.parent.mkdir(parents=True, exist_ok=True)
	report = {
		"data": {
			"labels_path": str(labels_path),
			"total_samples": int(len(df)),
			"train_samples": int(len(train_df)),
			"test_samples": int(len(test_df)),
			"image_size": int(args.image_size),
			"feature_vector_dims_raw": int(x_train.shape[1]),
			"feature_vector_dims_reduced": int(x_train_reduced.shape[1]),
			"label_distribution_total": {
				str(_to_python_scalar(label)): int(count)
				for label, count in df["label"].value_counts().items()
			},
			"label_distribution_train": {
				str(_to_python_scalar(label)): int(count)
				for label, count in train_df["label"].value_counts().items()
			},
			"label_distribution_test": {
				str(_to_python_scalar(label)): int(count)
				for label, count in test_df["label"].value_counts().items()
			},
			"split": split_metrics,
		},
		"models": {
			"supervised": str(supervised_path),
			"semisupervised": str(semisupervised_path),
			"unsupervised": str(unsupervised_path),
			"feature_pipeline": str(feature_pipeline_path),
		},
		"dimensionality_reduction": {
			"pca": reduction_metrics,
			"embedding": {
				**embedding_metrics,
				"csv_path": str(embedding_path) if args.embedding_method != "none" else None,
				"plot_path": str(embedding_plot_path) if args.embedding_method != "none" else None,
			},
		},
		"metrics": {
			"supervised": supervised_metrics,
			"semisupervised": semisupervised_metrics,
			"unsupervised": unsupervised_metrics,
		},
	}
	with open(report_path, "w", encoding="utf-8") as f:
		json.dump(report, f, indent=2)

	print("\nTraining complete.")
	print("\nModel metrics summary:")
	print(
		"- Supervised   | Acc={:.4f}, MacroF1={:.4f}, ROC_AUC={}".format(
			report["metrics"]["supervised"].get("accuracy", 0.0),
			report["metrics"]["supervised"].get("macro_f1", 0.0),
			report["metrics"]["supervised"].get("roc_auc"),
		)
	)
	print(
		"- Semi-superv. | Acc={:.4f}, MacroF1={:.4f}, ROC_AUC={}, PseudoAdded={}".format(
			report["metrics"]["semisupervised"].get("accuracy", 0.0),
			report["metrics"]["semisupervised"].get("macro_f1", 0.0),
			report["metrics"]["semisupervised"].get("roc_auc"),
			report["metrics"]["semisupervised"].get("pseudo_labels_added", 0),
		)
	)
	print(
		"- Unsupervised | Inertia={:.4f}, Silhouette={}".format(
			report["metrics"]["unsupervised"].get("inertia", 0.0),
			report["metrics"]["unsupervised"].get("silhouette_score"),
		)
	)
	print(f"- Supervised model     : {supervised_path}")
	print(f"- Semi-supervised model: {semisupervised_path}")
	print(f"- Unsupervised model   : {unsupervised_path}")
	print(f"- Feature pipeline     : {feature_pipeline_path}")
	if args.embedding_method != "none":
		print(f"- 2D embedding CSV     : {embedding_path}")
		print(f"- 2D embedding plot    : {embedding_plot_path}")
	print(f"- Training report      : {report_path}")


if __name__ == "__main__":
	main()

