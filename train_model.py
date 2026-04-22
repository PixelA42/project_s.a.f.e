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
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
	accuracy_score,
	average_precision_score,
	classification_report,
	confusion_matrix,
	f1_score,
	log_loss,
	precision_score,
	recall_score,
	roc_auc_score,
	silhouette_score,
)
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train S.A.F.E. baseline models from labels.csv")
	parser.add_argument("--labels", default="labels.csv", help="Path to labels CSV file")
	parser.add_argument("--models-dir", default="models", help="Directory to write model artifacts")
	parser.add_argument(
		"--report-path",
		default="models/training_report.json",
		help="Path to write training report JSON",
	)
	parser.add_argument(
		"--image-size",
		type=int,
		default=64,
		help="Square resize target used for feature extraction",
	)
	parser.add_argument("--test-size", type=float, default=0.4, help="Held-out split size")
	parser.add_argument(
		"--pseudo-threshold",
		type=float,
		default=0.85,
		help="Confidence threshold used for pseudo-label acceptance",
	)
	parser.add_argument(
		"--semi-unlabeled-ratio",
		type=float,
		default=0.3,
		help="Fraction of training split treated as unlabeled in semi-supervised stage",
	)
	parser.add_argument("--random-state", type=int, default=42)
	return parser.parse_args()


def _validate_dataframe(df: pd.DataFrame) -> None:
	required_columns = {"file_path", "label", "original_audio"}
	missing = required_columns.difference(df.columns)
	if missing:
		raise ValueError(f"labels.csv is missing required columns: {sorted(missing)}")


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


def _split_without_leakage(
	df: pd.DataFrame,
	test_size: float,
	random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	unique_files = df["original_audio"].unique()
	train_files, test_files = train_test_split(
		unique_files,
		test_size=test_size,
		random_state=random_state,
	)
	train_df = df[df["original_audio"].isin(train_files)].copy()
	test_df = df[df["original_audio"].isin(test_files)].copy()
	return train_df, test_df


def _train_supervised(
	x_train: np.ndarray,
	y_train: np.ndarray,
	x_test: np.ndarray,
	y_test: np.ndarray,
	random_state: int,
) -> tuple[RandomForestClassifier, dict[str, Any]]:
	model = RandomForestClassifier(
		n_estimators=350,
		random_state=random_state,
		n_jobs=-1,
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

	metrics: dict[str, Any] = {
		"accuracy": float(accuracy_score(y_test, preds)),
		"macro_f1": float(f1_score(y_test, preds, average="macro", zero_division=0)),
		"weighted_f1": float(f1_score(y_test, preds, average="weighted", zero_division=0)),
		"macro_precision": float(
			precision_score(y_test, preds, average="macro", zero_division=0)
		),
		"macro_recall": float(
			recall_score(y_test, preds, average="macro", zero_division=0)
		),
		"confusion_matrix": confusion_matrix(y_test, preds).tolist(),
		"classification_report": classification_report(
			y_test,
			preds,
			output_dict=True,
			zero_division=0,
		),
	}

	if probs is not None:
		try:
			metrics["log_loss"] = float(log_loss(y_test, probs, labels=classes))
		except Exception:
			metrics["log_loss"] = None

		if probs.shape[1] == 2 and np.unique(y_test).size == 2:
			positive_class = 1 if 1 in classes else classes[1]
			positive_index = int(np.where(classes == positive_class)[0][0])
			positive_probs = probs[:, positive_index]
			try:
				metrics["roc_auc"] = float(roc_auc_score(y_test, positive_probs))
			except Exception:
				metrics["roc_auc"] = None
			try:
				metrics["pr_auc"] = float(average_precision_score(y_test, positive_probs))
			except Exception:
				metrics["pr_auc"] = None

	return metrics


def _train_unsupervised(x_train: np.ndarray, random_state: int) -> tuple[KMeans, dict[str, Any]]:
	n_clusters = max(2, min(8, x_train.shape[0]))
	kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
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
	indices = np.arange(x_train.shape[0])
	labeled_idx, unlabeled_idx = train_test_split(
		indices,
		test_size=unlabeled_ratio,
		random_state=random_state,
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
	})
	return semi_model, metrics


def main() -> None:
	args = parse_args()
	labels_path = Path(args.labels)
	models_dir = Path(args.models_dir)
	report_path = Path(args.report_path)

	if not labels_path.is_file():
		raise FileNotFoundError(f"labels CSV not found: {labels_path}")

	df = pd.read_csv(labels_path)
	_validate_dataframe(df)

	train_df, test_df = _split_without_leakage(df, args.test_size, args.random_state)

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

	supervised_model, supervised_metrics = _train_supervised(
		x_train,
		y_train,
		x_test,
		y_test,
		args.random_state,
	)
	kmeans_model, unsupervised_metrics = _train_unsupervised(x_train, args.random_state)
	semi_model, semisupervised_metrics = _train_semisupervised(
		supervised_model,
		x_train,
		y_train,
		x_test,
		y_test,
		args.semi_unlabeled_ratio,
		args.pseudo_threshold,
		args.random_state,
	)

	models_dir.mkdir(parents=True, exist_ok=True)

	supervised_path = models_dir / "spectrogram_supervised_model.joblib"
	semisupervised_path = models_dir / "spectrogram_semisupervised_model.joblib"
	unsupervised_path = models_dir / "spectrogram_kmeans_model.joblib"

	joblib.dump(supervised_model, supervised_path)
	joblib.dump(semi_model, semisupervised_path)
	joblib.dump(kmeans_model, unsupervised_path)

	report_path.parent.mkdir(parents=True, exist_ok=True)
	report = {
		"data": {
			"labels_path": str(labels_path),
			"total_samples": int(len(df)),
			"train_samples": int(len(train_df)),
			"test_samples": int(len(test_df)),
			"image_size": int(args.image_size),
		},
		"models": {
			"supervised": str(supervised_path),
			"semisupervised": str(semisupervised_path),
			"unsupervised": str(unsupervised_path),
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
	print(f"- Training report      : {report_path}")


if __name__ == "__main__":
	main()

