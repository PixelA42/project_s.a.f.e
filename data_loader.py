"""Dataset loading and preprocessing utilities for Project S.A.F.E."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import wave

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import SETTINGS


GENERAL_CONFIG = SETTINGS.general
PATHS = SETTINGS.paths
DATA_CONFIG = SETTINGS.data
AUDIO_CONFIG = SETTINGS.audio


class DataLoadingError(ValueError):
    """Raised when the dataset cannot be loaded into a valid ML-ready structure."""


@dataclass(frozen=True)
class LoadedData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    X_full: np.ndarray
    target_column: str
    feature_names: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    dropped_columns: dict[str, str]
    null_strategy_by_column: dict[str, str]
    dataset_path: Path
    source_type: str
    metadata: dict[str, Any]
    preprocessor: ColumnTransformer


def load_data(
    dataset_path: str | Path | None = None,
    *,
    test_size: float = DATA_CONFIG.test_split_ratio,
    random_state: int = GENERAL_CONFIG.random_seed,
) -> LoadedData:
    """Load, validate, preprocess, and split the configured dataset."""

    resolved_path = Path(dataset_path) if dataset_path is not None else PATHS.audio_dataset_dir
    resolved_path = resolved_path.expanduser().resolve()

    if not resolved_path.exists():
        raise DataLoadingError(f"Dataset path does not exist: {resolved_path}")

    raw_df, source_type, source_metadata = _read_dataset(resolved_path)
    _validate_raw_dataframe(raw_df, resolved_path)

    target_column = _detect_target_column(raw_df)
    if target_column is None:
        raise DataLoadingError(
            "No valid target column could be inferred. Expected a low-cardinality label-like "
            "column such as target/label/class, or a labeled audio directory structure."
        )

    feature_frame = raw_df.copy()
    y_all = feature_frame[target_column]

    dropped_columns = _identify_columns_to_drop(feature_frame, target_column, source_type)
    if dropped_columns:
        feature_frame = feature_frame.drop(columns=list(dropped_columns))

    feature_frame = feature_frame.drop(columns=[target_column], errors="ignore")
    if feature_frame.empty:
        raise DataLoadingError(
            "No usable feature columns remain after removing target and identifier columns."
        )

    feature_frame, numeric_columns, categorical_columns = _infer_feature_types(feature_frame)
    feature_frame, null_strategy_by_column, null_drops = _apply_null_handling(
        feature_frame,
        numeric_columns,
        categorical_columns,
    )
    dropped_columns.update(null_drops)

    constant_columns = {
        column: "constant_column"
        for column in feature_frame.columns
        if feature_frame[column].nunique(dropna=False) <= 1
    }
    if constant_columns:
        feature_frame = feature_frame.drop(columns=list(constant_columns))
        dropped_columns.update(constant_columns)

    if feature_frame.empty:
        raise DataLoadingError("All feature columns were dropped during validation/preprocessing.")

    feature_frame, numeric_columns, categorical_columns = _infer_feature_types(feature_frame)
    if not numeric_columns and not categorical_columns:
        raise DataLoadingError("No numeric or categorical feature columns are available to model.")

    labeled_mask = y_all.notna()
    labeled_count = int(labeled_mask.sum())
    unlabeled_count = int((~labeled_mask).sum())
    if labeled_count < 2:
        raise DataLoadingError(
            "No valid target was found for supervised learning. Need at least 2 labeled rows."
        )

    x_labeled = feature_frame.loc[labeled_mask].reset_index(drop=True)
    y_labeled = y_all.loc[labeled_mask].reset_index(drop=True)
    if y_labeled.nunique(dropna=True) < 2:
        raise DataLoadingError(
            f"Target column '{target_column}' must contain at least 2 classes for supervised learning."
        )

    stratify = y_labeled if _can_stratify(y_labeled) else None
    split_strategy = "random"
    try:
        x_train_df, x_test_df, y_train, y_test = train_test_split(
            x_labeled,
            y_labeled,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError as exc:
        if stratify is None:
            raise DataLoadingError(
                f"Unable to create a supervised train/test split from '{target_column}': {exc}"
            ) from exc
        x_train_df, x_test_df, y_train, y_test = train_test_split(
            x_labeled,
            y_labeled,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )
        split_strategy = "random_fallback"
    else:
        if stratify is not None:
            split_strategy = "stratified"

    preprocessor = _build_preprocessor(numeric_columns, categorical_columns)
    x_train = preprocessor.fit_transform(x_train_df)
    x_test = preprocessor.transform(x_test_df)
    x_full = preprocessor.transform(feature_frame)
    feature_names = list(preprocessor.get_feature_names_out())

    metadata = {
        **source_metadata,
        "row_count": int(len(raw_df)),
        "labeled_row_count": labeled_count,
        "unlabeled_row_count": unlabeled_count,
        "target_column": target_column,
        "target_distribution": {
            str(label): int(count) for label, count in y_labeled.value_counts().items()
        },
        "feature_count_before_encoding": int(feature_frame.shape[1]),
        "feature_count_after_encoding": int(len(feature_names)),
        "split_strategy": split_strategy,
        "dropped_columns": dict(dropped_columns),
        "null_strategy_by_column": dict(null_strategy_by_column),
    }

    return LoadedData(
        X_train=np.asarray(x_train, dtype=np.float32),
        X_test=np.asarray(x_test, dtype=np.float32),
        y_train=y_train.to_numpy(),
        y_test=y_test.to_numpy(),
        X_full=np.asarray(x_full, dtype=np.float32),
        target_column=target_column,
        feature_names=feature_names,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        dropped_columns=dropped_columns,
        null_strategy_by_column=null_strategy_by_column,
        dataset_path=resolved_path,
        source_type=source_type,
        metadata=metadata,
        preprocessor=preprocessor,
    )


def _read_dataset(dataset_path: Path) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    if dataset_path.is_file():
        return _read_tabular_file(dataset_path), "tabular_file", {"dataset_file": str(dataset_path)}

    tabular_files = sorted(
        path
        for path in dataset_path.iterdir()
        if path.is_file() and path.suffix.lower() in DATA_CONFIG.supported_tabular_extensions
    )
    if len(tabular_files) == 1:
        return (
            _read_tabular_file(tabular_files[0]),
            "tabular_file",
            {"dataset_file": str(tabular_files[0]), "dataset_directory": str(dataset_path)},
        )
    if len(tabular_files) > 1:
        raise DataLoadingError(
            f"Multiple tabular files were found in {dataset_path}. "
            "Pass an explicit file path to load_data(dataset_path=...)."
        )

    audio_files = sorted(
        path
        for path in dataset_path.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_CONFIG.supported_extensions
    )
    if not audio_files:
        raise DataLoadingError(
            f"No supported dataset files were found under {dataset_path}. "
            f"Expected tabular files {DATA_CONFIG.supported_tabular_extensions} or audio files "
            f"{AUDIO_CONFIG.supported_extensions}."
        )

    dataframe, metadata = _build_audio_metadata_dataframe(dataset_path, audio_files)
    return dataframe, "audio_directory", metadata


def _read_tabular_file(dataset_file: Path) -> pd.DataFrame:
    suffix = dataset_file.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(dataset_file)
        if suffix == ".json":
            return pd.read_json(dataset_file)
        if suffix == ".parquet":
            return pd.read_parquet(dataset_file)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(dataset_file)
    except pd.errors.EmptyDataError as exc:
        raise DataLoadingError(f"Dataset file is empty: {dataset_file}") from exc
    except Exception as exc:
        raise DataLoadingError(f"Failed to read dataset file '{dataset_file}': {exc}") from exc

    raise DataLoadingError(f"Unsupported dataset file extension: {dataset_file.suffix}")


def _build_audio_metadata_dataframe(
    dataset_path: Path,
    audio_files: list[Path],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    unreadable_files = 0

    for audio_file in audio_files:
        stats = audio_file.stat()
        relative_parts = audio_file.relative_to(dataset_path).parts
        target, reason = _infer_audio_target(relative_parts)

        duration_seconds = np.nan
        sample_rate = np.nan
        channels = np.nan
        frame_count = np.nan
        subtype: str | None = None

        try:
            duration_seconds, sample_rate, channels, frame_count, subtype = _read_audio_metadata(
                audio_file
            )
        except Exception:
            unreadable_files += 1

        rows.append(
            {
                "file_path": str(audio_file.resolve()),
                "relative_path": str(audio_file.relative_to(dataset_path)),
                "top_level_dir": relative_parts[0] if len(relative_parts) > 1 else audio_file.parent.name,
                "parent_dir": audio_file.parent.name,
                "extension": audio_file.suffix.lower(),
                "size_bytes": float(stats.st_size),
                "duration_seconds": duration_seconds,
                "sample_rate": sample_rate,
                "channels": channels,
                "frame_count": frame_count,
                "subtype": subtype,
                "target": target,
                "label_inference_reason": reason,
            }
        )

    return pd.DataFrame(rows), {
        "dataset_directory": str(dataset_path),
        "audio_file_count": int(len(audio_files)),
        "unreadable_audio_files": int(unreadable_files),
    }


def _read_audio_metadata(audio_file: Path) -> tuple[float, float, float, float, str | None]:
    if audio_file.suffix.lower() != ".wav":
        raise DataLoadingError(
            f"Audio metadata inspection currently supports WAV files only: {audio_file}"
        )

    with wave.open(str(audio_file), "rb") as wav_file:
        frame_count = float(wav_file.getnframes())
        sample_rate = float(wav_file.getframerate())
        channels = float(wav_file.getnchannels())
        sample_width_bits = int(wav_file.getsampwidth()) * 8

    duration_seconds = frame_count / sample_rate if sample_rate else np.nan
    subtype = f"PCM_{sample_width_bits}" if sample_width_bits > 0 else None
    return duration_seconds, sample_rate, channels, frame_count, subtype


def _infer_audio_target(relative_parts: tuple[str, ...]) -> tuple[str | None, str]:
    normalized_parts = [_normalize_token(part) for part in relative_parts[:-1]]

    if _contains_directory_marker(normalized_parts, DATA_CONFIG.unlabeled_directory_markers):
        return None, "unlabeled_directory"
    if _contains_directory_marker(normalized_parts, DATA_CONFIG.synthetic_directory_markers):
        return "synthetic", "synthetic_directory_marker"
    if _contains_directory_marker(normalized_parts, DATA_CONFIG.real_directory_markers):
        return "real", "real_directory_marker"
    return "real", "default_directory_label"


def _contains_directory_marker(parts: list[str], markers: tuple[str, ...]) -> bool:
    return any(marker in part for part in parts for marker in markers)


def _normalize_token(value: str) -> str:
    return "".join(ch for ch in value.casefold() if ch.isalnum())


def _validate_raw_dataframe(dataframe: pd.DataFrame, dataset_path: Path) -> None:
    if dataframe is None:
        raise DataLoadingError(f"Dataset at {dataset_path} could not be loaded.")
    if dataframe.empty:
        raise DataLoadingError(f"Dataset is empty: {dataset_path}")
    if len(dataframe.columns) == 0:
        raise DataLoadingError(f"Dataset has no columns: {dataset_path}")


def _detect_target_column(dataframe: pd.DataFrame) -> str | None:
    normalized_map = {_normalize_token(column): column for column in dataframe.columns}

    for candidate_name in DATA_CONFIG.target_candidate_names:
        column = normalized_map.get(_normalize_token(candidate_name))
        if column and _is_valid_target_series(dataframe[column], allow_numeric=True):
            return column

    fallback_candidates: list[tuple[int, str]] = []
    for column in dataframe.columns:
        if _is_valid_target_series(dataframe[column], allow_numeric=False):
            unique_count = int(dataframe[column].dropna().nunique())
            fallback_candidates.append((unique_count, column))

    if not fallback_candidates:
        return None

    fallback_candidates.sort(key=lambda item: (item[0], item[1]))
    return fallback_candidates[0][1]


def _is_valid_target_series(series: pd.Series, *, allow_numeric: bool) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False

    unique_count = int(non_null.nunique())
    if unique_count < 2:
        return False
    if unique_count >= len(non_null):
        return False

    if pd.api.types.is_numeric_dtype(non_null):
        if not allow_numeric:
            return False
        return unique_count <= max(20, int(len(non_null) * 0.1))

    return unique_count <= max(50, int(len(non_null) * 0.5))


def _identify_columns_to_drop(
    dataframe: pd.DataFrame,
    target_column: str,
    source_type: str,
) -> dict[str, str]:
    dropped: dict[str, str] = {}
    reserved_metadata = {"label_inference_reason"}
    if source_type == "audio_directory":
        reserved_metadata.update({"file_path", "relative_path", "top_level_dir"})

    for column in dataframe.columns:
        if column == target_column:
            continue
        if column in reserved_metadata:
            dropped[column] = "metadata_only_column"
            continue

        series = dataframe[column]
        non_null = series.dropna()
        if non_null.empty:
            dropped[column] = "all_values_missing"
            continue

        unique_ratio = float(non_null.nunique() / max(1, len(non_null)))
        normalized_name = _normalize_token(column)
        if unique_ratio >= DATA_CONFIG.identifier_unique_ratio_threshold and (
            "id" in normalized_name
            or "path" in normalized_name
            or "file" in normalized_name
            or "name" in normalized_name
        ):
            dropped[column] = "identifier_like_column"

    return dropped


def _infer_feature_types(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    frame = dataframe.copy()
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []

    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_bool_dtype(series):
            frame[column] = series.astype(str)
            categorical_columns.append(column)
            continue
        if pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column)
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            frame[column] = pd.to_datetime(series, errors="coerce").view("int64") / 10**9
            numeric_columns.append(column)
            continue

        converted = pd.to_numeric(series, errors="coerce")
        non_null = series.dropna()
        conversion_ratio = float(converted.notna().sum() / max(1, len(non_null))) if len(non_null) else 0.0
        if conversion_ratio >= 0.90 and converted.notna().sum() > 0:
            frame[column] = converted
            numeric_columns.append(column)
        else:
            frame[column] = series.astype("string")
            categorical_columns.append(column)

    return frame, numeric_columns, categorical_columns


def _apply_null_handling(
    dataframe: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    frame = dataframe.copy()
    strategies: dict[str, str] = {}
    dropped_columns: dict[str, str] = {}

    for column in list(numeric_columns):
        series = frame[column]
        null_ratio = float(series.isna().mean())
        if null_ratio == 0.0:
            strategies[column] = "none"
            continue
        if null_ratio <= DATA_CONFIG.numeric_fill_threshold:
            fill_value = float(series.median()) if series.notna().any() else 0.0
            frame[column] = series.fillna(fill_value)
            strategies[column] = "fill_median"
            continue
        if null_ratio <= DATA_CONFIG.numeric_interpolate_threshold:
            interpolated = series.interpolate(method="linear", limit_direction="both")
            fill_value = float(interpolated.median()) if interpolated.notna().any() else 0.0
            frame[column] = interpolated.fillna(fill_value)
            strategies[column] = "interpolate_then_fill_median"
            continue

        frame = frame.drop(columns=[column])
        dropped_columns[column] = "dropped_high_null_numeric"
        strategies[column] = "drop_column"

    for column in list(categorical_columns):
        if column not in frame.columns:
            continue

        series = frame[column]
        null_ratio = float(series.isna().mean())
        if null_ratio == 0.0:
            strategies[column] = "none"
            continue
        if null_ratio <= DATA_CONFIG.categorical_fill_threshold:
            fill_value = _mode_or_default(series, "missing")
            frame[column] = series.fillna(fill_value)
            strategies[column] = "fill_mode"
            continue
        if null_ratio <= DATA_CONFIG.categorical_missing_bucket_threshold:
            frame[column] = series.fillna("missing")
            strategies[column] = "fill_missing_bucket"
            continue

        frame = frame.drop(columns=[column])
        dropped_columns[column] = "dropped_high_null_categorical"
        strategies[column] = "drop_column"

    return frame, strategies, dropped_columns


def _mode_or_default(series: pd.Series, default: str) -> str:
    mode = series.mode(dropna=True)
    if mode.empty:
        return default
    return str(mode.iloc[0])


def _build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(steps=[("scaler", StandardScaler())]),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        )
                    ]
                ),
                categorical_columns,
            )
        )

    if not transformers:
        raise DataLoadingError("No feature transformers could be created from the dataset.")

    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def _can_stratify(labels: pd.Series) -> bool:
    if labels.empty:
        return False
    counts = labels.value_counts(dropna=True)
    return counts.size > 1 and int(counts.min()) >= 2


__all__ = ["DataLoadingError", "LoadedData", "load_data"]
