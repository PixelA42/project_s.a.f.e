"""Cross-validated supervised model training for Project S.A.F.E."""

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
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

from config import SETTINGS
from experiment_tracking import append_experiment_log_row, to_compact_json, utc_timestamp
from data_loader import LoadedData, load_data


GENERAL_CONFIG = SETTINGS.general
PATHS = SETTINGS.paths


@dataclass(frozen=True)
class SupervisedRunResult:
    task_type: str
    target_column: str
    best_model_name: str
    best_model_path: Path
    output_dir: Path
    metrics_path: Path
    fold_metrics_path: Path
    artifact_index_path: Path
    metrics: pd.DataFrame
    fold_metrics: pd.DataFrame
    artifact_index: pd.DataFrame


def run_supervised(
    dataset_path: str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
    cv_folds: int = 5,
    random_state: int = GENERAL_CONFIG.random_seed,
    model_names: list[str] | None = None,
) -> SupervisedRunResult:
    loaded = load_data(dataset_path=dataset_path, random_state=random_state)
    output_path = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else PATHS.supervised_outputs_dir.resolve()
    )
    output_path.mkdir(parents=True, exist_ok=True)

    estimators = _build_estimators(loaded.task_type, random_state)
    if model_names is not None:
        requested = set(model_names)
        estimators = {name: estimator for name, estimator in estimators.items() if name in requested}
        missing = sorted(requested.difference(estimators))
        if missing:
            raise ValueError(f"Unknown model names requested: {missing}")
    if not estimators:
        raise ValueError("No supervised models are available to run.")

    cv = _build_cv_splitter(loaded.task_type, loaded.y_labeled, cv_folds, random_state)

    metrics_rows: list[dict[str, Any]] = []
    fold_frames: list[pd.DataFrame] = []
    artifact_rows: list[dict[str, Any]] = []
    fitted_pipelines: dict[str, Pipeline] = {}

    for model_name, estimator in estimators.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(loaded.preprocessor)),
                ("model", estimator),
            ]
        )
        scoring = _build_scoring(loaded.task_type, loaded.y_labeled)
        cv_result = cross_validate(
            pipeline,
            loaded.X_labeled_frame,
            loaded.y_labeled,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=1,
        )
        fold_frame = _fold_metrics_frame(model_name, loaded.task_type, cv_result)
        fold_frames.append(fold_frame)

        oof_predictions = cross_val_predict(
            pipeline,
            loaded.X_labeled_frame,
            loaded.y_labeled,
            cv=cv,
            method="predict",
            n_jobs=1,
        )
        oof_probabilities = None
        if loaded.task_type == "classification":
            oof_probabilities = cross_val_predict(
                pipeline,
                loaded.X_labeled_frame,
                loaded.y_labeled,
                cv=cv,
                method="predict_proba",
                n_jobs=1,
            )

        fitted_pipeline = clone(pipeline)
        fitted_pipeline.fit(loaded.X_labeled_frame, loaded.y_labeled)
        fitted_pipelines[model_name] = fitted_pipeline

        model_output_dir = output_path / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        prediction_path = model_output_dir / "oof_predictions.csv"
        prediction_frame = _prediction_frame(
            loaded.task_type,
            loaded.y_labeled,
            oof_predictions,
            oof_probabilities,
        )
        prediction_frame.to_csv(prediction_path, index=False)
        artifact_rows.append(
            {"model": model_name, "artifact_type": "oof_predictions", "path": str(prediction_path)}
        )

        metrics_row, extra_artifacts = _evaluate_and_save_artifacts(
            model_name=model_name,
            task_type=loaded.task_type,
            target=loaded.y_labeled,
            predictions=oof_predictions,
            probabilities=oof_probabilities,
            fitted_pipeline=fitted_pipeline,
            output_dir=model_output_dir,
        )
        metrics_rows.append(metrics_row)
        artifact_rows.extend(extra_artifacts)

        model_path = model_output_dir / "pipeline.joblib"
        joblib.dump(fitted_pipeline, model_path)
        artifact_rows.append(
            {"model": model_name, "artifact_type": "fitted_pipeline", "path": str(model_path)}
        )

    metrics_frame = pd.DataFrame(metrics_rows).sort_values(
        by="primary_score", ascending=False
    ).reset_index(drop=True)
    fold_metrics_frame = pd.concat(fold_frames, ignore_index=True)
    artifact_index = pd.DataFrame(artifact_rows).sort_values(
        by=["model", "artifact_type", "path"]
    ).reset_index(drop=True)

    metrics_path = output_path / "metrics_summary.csv"
    fold_metrics_path = output_path / "cv_fold_metrics.csv"
    artifact_index_path = output_path / "artifact_index.csv"
    metrics_frame.to_csv(metrics_path, index=False)
    fold_metrics_frame.to_csv(fold_metrics_path, index=False)
    artifact_index.to_csv(artifact_index_path, index=False)

    best_model_name = str(metrics_frame.iloc[0]["model"])
    best_model_path = output_path / best_model_name / "pipeline.joblib"
    summary_path = output_path / "run_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "task_type": loaded.task_type,
                "target_column": loaded.target_column,
                "dataset_path": str(loaded.dataset_path),
                "best_model_name": best_model_name,
                "best_model_path": str(best_model_path),
                "models_evaluated": metrics_frame["model"].tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    try:
        best_metrics = metrics_frame.iloc[0].to_dict()
        key_metrics = {
            "task_type": loaded.task_type,
            "target_column": loaded.target_column,
            "primary_score": float(best_metrics.get("primary_score", 0.0)),
        }
        for metric_name in (
            "accuracy",
            "balanced_accuracy",
            "f1_macro",
            "f1_weighted",
            "precision_macro",
            "recall_macro",
            "roc_auc",
            "pr_auc",
            "r2",
            "rmse",
            "mae",
        ):
            if metric_name in best_metrics and pd.notna(best_metrics[metric_name]):
                key_metrics[metric_name] = float(best_metrics[metric_name])

        append_experiment_log_row(
            PATHS.experiment_log_path,
            {
                "timestamp": utc_timestamp(),
                "dataset_name": loaded.dataset_path.name,
                "model_name": best_model_name,
                "key_metrics": to_compact_json(key_metrics),
                "hyperparameters": to_compact_json(
                    {
                        "cv_folds": int(cv_folds),
                        "random_state": int(random_state),
                        "model_names": sorted(model_names) if model_names is not None else None,
                    }
                ),
            },
        )
    except Exception as exc:
        print(f"[warn] Failed to append experiment log: {exc}")

    return SupervisedRunResult(
        task_type=loaded.task_type,
        target_column=loaded.target_column,
        best_model_name=best_model_name,
        best_model_path=best_model_path,
        output_dir=output_path,
        metrics_path=metrics_path,
        fold_metrics_path=fold_metrics_path,
        artifact_index_path=artifact_index_path,
        metrics=metrics_frame,
        fold_metrics=fold_metrics_frame,
        artifact_index=artifact_index,
    )


def _build_estimators(task_type: str, random_state: int) -> dict[str, Any]:
    if task_type == "classification":
        return {
            "logistic_regression": LogisticRegression(
                max_iter=1000,
                random_state=random_state,
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
            "extra_trees": ExtraTreesClassifier(
                n_estimators=300,
                random_state=random_state,
                n_jobs=-1,
            ),
        }

    return {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(random_state=random_state),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting_regressor": GradientBoostingRegressor(random_state=random_state),
        "extra_trees_regressor": ExtraTreesRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def _build_cv_splitter(task_type: str, target: pd.Series, cv_folds: int, random_state: int) -> Any:
    if cv_folds < 2:
        raise ValueError("cv_folds must be >= 2.")

    sample_count = int(len(target))
    if sample_count < 2:
        raise ValueError("At least 2 labeled samples are required for cross-validation.")

    if task_type == "classification":
        counts = target.value_counts()
        min_class_count = int(counts.min())
        if min_class_count >= 2:
            n_splits = min(cv_folds, sample_count, min_class_count)
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    n_splits = min(cv_folds, sample_count)
    if n_splits < 2:
        raise ValueError("Cross-validation requires at least 2 folds.")
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _build_scoring(task_type: str, target: pd.Series) -> dict[str, str]:
    if task_type == "classification":
        scoring = {
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1_macro": "f1_macro",
            "f1_weighted": "f1_weighted",
            "precision_macro": "precision_macro",
            "recall_macro": "recall_macro",
        }
        if int(target.nunique()) == 2:
            scoring.update(
                {
                    "roc_auc": "roc_auc",
                    "neg_log_loss": "neg_log_loss",
                }
            )
        else:
            scoring.update(
                {
                    "roc_auc_ovr_weighted": "roc_auc_ovr_weighted",
                    "neg_log_loss": "neg_log_loss",
                }
            )
        return scoring

    return {
        "r2": "r2",
        "explained_variance": "explained_variance",
        "neg_mean_absolute_error": "neg_mean_absolute_error",
        "neg_mean_squared_error": "neg_mean_squared_error",
        "neg_median_absolute_error": "neg_median_absolute_error",
    }


def _fold_metrics_frame(model_name: str, task_type: str, cv_result: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(cv_result)
    renamed: dict[str, str] = {}
    for column in frame.columns:
        if column.startswith("test_"):
            renamed[column] = column.removeprefix("test_")
    frame = frame.rename(columns=renamed)

    for column in list(frame.columns):
        if column.startswith("neg_"):
            positive_name = column.removeprefix("neg_")
            frame[positive_name] = -frame[column]
            frame = frame.drop(columns=[column])

    if task_type == "regression" and "mean_squared_error" in frame.columns:
        frame["rmse"] = np.sqrt(frame["mean_squared_error"])

    frame.insert(0, "fold", np.arange(1, len(frame) + 1))
    frame.insert(0, "model", model_name)
    return frame


def _prediction_frame(
    task_type: str,
    target: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
) -> pd.DataFrame:
    frame = pd.DataFrame({"actual": target.to_numpy(), "predicted": predictions})
    if task_type == "classification" and probabilities is not None:
        classes = np.unique(target.to_numpy())
        for index, label in enumerate(classes):
            frame[f"prob_{label}"] = probabilities[:, index]
    if task_type == "regression":
        frame["residual"] = frame["actual"] - frame["predicted"]
    return frame


def _evaluate_and_save_artifacts(
    *,
    model_name: str,
    task_type: str,
    target: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
    fitted_pipeline: Pipeline,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if task_type == "classification":
        return _evaluate_classification(
            model_name=model_name,
            target=target,
            predictions=predictions,
            probabilities=probabilities,
            fitted_pipeline=fitted_pipeline,
            output_dir=output_dir,
        )

    return _evaluate_regression(
        model_name=model_name,
        target=target,
        predictions=predictions,
        fitted_pipeline=fitted_pipeline,
        output_dir=output_dir,
    )


def _evaluate_classification(
    *,
    model_name: str,
    target: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
    fitted_pipeline: Pipeline,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    actual = target.to_numpy()
    classes = np.unique(actual)
    metrics = {
        "model": model_name,
        "task_type": "classification",
        "sample_count": int(len(actual)),
        "class_count": int(len(classes)),
        "accuracy": float(accuracy_score(actual, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predictions)),
        "f1_macro": float(f1_score(actual, predictions, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(actual, predictions, average="weighted", zero_division=0)),
        "precision_macro": float(
            precision_score(actual, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(actual, predictions, average="macro", zero_division=0)),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "log_loss": np.nan,
    }

    if probabilities is not None:
        try:
            metrics["log_loss"] = float(log_loss(actual, probabilities, labels=classes))
        except Exception:
            metrics["log_loss"] = np.nan

        if len(classes) == 2:
            positive_index = 1
            positive_probs = probabilities[:, positive_index]
            positive_label = classes[positive_index]
            binary_actual = (actual == positive_label).astype(int)
            metrics["roc_auc"] = float(roc_auc_score(binary_actual, positive_probs))
            metrics["pr_auc"] = float(average_precision_score(binary_actual, positive_probs))
        else:
            metrics["roc_auc"] = float(
                roc_auc_score(actual, probabilities, multi_class="ovr", average="weighted")
            )

    metrics["primary_score"] = float(metrics["f1_macro"])

    artifacts: list[dict[str, Any]] = []

    report_frame = pd.DataFrame(
        classification_report(actual, predictions, output_dict=True, zero_division=0)
    ).transpose()
    report_path = output_dir / "classification_report.csv"
    report_frame.to_csv(report_path)
    artifacts.append(
        {"model": model_name, "artifact_type": "classification_report", "path": str(report_path)}
    )

    confusion = confusion_matrix(actual, predictions, labels=classes)
    confusion_frame = pd.DataFrame(confusion, index=classes, columns=classes)
    confusion_csv_path = output_dir / "confusion_matrix.csv"
    confusion_png_path = output_dir / "confusion_matrix.png"
    confusion_frame.to_csv(confusion_csv_path)
    _plot_confusion_matrix(confusion_frame, confusion_png_path, model_name)
    artifacts.extend(
        [
            {"model": model_name, "artifact_type": "confusion_matrix_csv", "path": str(confusion_csv_path)},
            {"model": model_name, "artifact_type": "confusion_matrix_png", "path": str(confusion_png_path)},
        ]
    )

    importance_artifacts = _save_feature_importances(model_name, fitted_pipeline, output_dir)
    artifacts.extend(importance_artifacts)
    return metrics, artifacts


def _evaluate_regression(
    *,
    model_name: str,
    target: pd.Series,
    predictions: np.ndarray,
    fitted_pipeline: Pipeline,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    actual = target.to_numpy(dtype=float)
    preds = np.asarray(predictions, dtype=float)
    mse = float(mean_squared_error(actual, preds))

    try:
        mape = float(mean_absolute_percentage_error(actual, preds))
    except Exception:
        mape = np.nan

    metrics = {
        "model": model_name,
        "task_type": "regression",
        "sample_count": int(len(actual)),
        "mae": float(mean_absolute_error(actual, preds)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(actual, preds)),
        "explained_variance": float(explained_variance_score(actual, preds)),
        "median_absolute_error": float(median_absolute_error(actual, preds)),
        "mape": mape,
    }
    metrics["primary_score"] = float(metrics["r2"])

    residual_frame = pd.DataFrame(
        {"actual": actual, "predicted": preds, "residual": actual - preds}
    )
    residual_csv_path = output_dir / "residuals.csv"
    residual_png_path = output_dir / "regression_diagnostics.png"
    residual_frame.to_csv(residual_csv_path, index=False)
    _plot_regression_diagnostics(residual_frame, residual_png_path, model_name)

    artifacts: list[dict[str, Any]] = [
        {"model": model_name, "artifact_type": "residuals_csv", "path": str(residual_csv_path)},
        {"model": model_name, "artifact_type": "regression_diagnostics_png", "path": str(residual_png_path)},
    ]
    artifacts.extend(_save_feature_importances(model_name, fitted_pipeline, output_dir))
    return metrics, artifacts


def _save_feature_importances(
    model_name: str,
    fitted_pipeline: Pipeline,
    output_dir: Path,
) -> list[dict[str, Any]]:
    model = fitted_pipeline.named_steps["model"]
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    feature_names = np.asarray(preprocessor.get_feature_names_out())
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_, dtype=float)
        if coefficients.ndim == 1:
            importances = np.abs(coefficients)
        else:
            importances = np.abs(coefficients).mean(axis=0)

    if importances is None or importances.shape[0] != feature_names.shape[0]:
        return []

    frame = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        by="importance", ascending=False
    )
    csv_path = output_dir / "feature_importances.csv"
    png_path = output_dir / "feature_importances.png"
    frame.to_csv(csv_path, index=False)
    _plot_feature_importances(frame, png_path, model_name)
    return [
        {"model": model_name, "artifact_type": "feature_importances_csv", "path": str(csv_path)},
        {"model": model_name, "artifact_type": "feature_importances_png", "path": str(png_path)},
    ]


def _plot_confusion_matrix(confusion_frame: pd.DataFrame, output_path: Path, model_name: str) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    matrix = confusion_frame.to_numpy()
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks(np.arange(confusion_frame.shape[1]), labels=confusion_frame.columns)
    axis.set_yticks(np.arange(confusion_frame.shape[0]), labels=confusion_frame.index)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(f"{model_name} Confusion Matrix")

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            axis.text(
                col_index,
                row_index,
                str(matrix[row_index, col_index]),
                ha="center",
                va="center",
                color="black",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_feature_importances(frame: pd.DataFrame, output_path: Path, model_name: str) -> None:
    top_frame = frame.head(20).iloc[::-1]
    figure, axis = plt.subplots(figsize=(8, max(4, int(len(top_frame) * 0.35))))
    axis.barh(top_frame["feature"], top_frame["importance"], color="#2b7a78")
    axis.set_xlabel("Importance")
    axis.set_title(f"{model_name} Feature Importances")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_regression_diagnostics(
    residual_frame: pd.DataFrame,
    output_path: Path,
    model_name: str,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].scatter(residual_frame["actual"], residual_frame["predicted"], alpha=0.7, color="#2b7a78")
    min_value = min(residual_frame["actual"].min(), residual_frame["predicted"].min())
    max_value = max(residual_frame["actual"].max(), residual_frame["predicted"].max())
    axes[0].plot([min_value, max_value], [min_value, max_value], linestyle="--", color="#c44536")
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title("Actual vs Predicted")

    axes[1].scatter(
        residual_frame["predicted"],
        residual_frame["residual"],
        alpha=0.7,
        color="#3a86ff",
    )
    axes[1].axhline(0.0, linestyle="--", color="#c44536")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residual Plot")

    figure.suptitle(f"{model_name} Regression Diagnostics")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


__all__ = ["SupervisedRunResult", "run_supervised"]
