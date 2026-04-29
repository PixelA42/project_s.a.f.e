from __future__ import annotations

from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
import librosa

from config import SETTINGS
from train_model import (
    _build_holdout_error_analysis_frame,
    _extract_image_features,
    _split_without_leakage,
)


def main() -> None:
    paths = SETTINGS.paths
    labels = pd.read_csv(paths.labels_csv_path)
    _, test_df, _ = _split_without_leakage(
        labels,
        test_size=SETTINGS.training.test_split_ratio,
        random_state=SETTINGS.general.random_seed,
    )

    feature_pipeline = joblib.load(paths.feature_pipeline_path)
    model = joblib.load(paths.supervised_model_path)

    resolved_test_paths = test_df["file_path"].map(lambda value: str((paths.project_root / Path(str(value))).resolve()))
    x_test = _extract_image_features(resolved_test_paths.to_numpy(), SETTINGS.audio.feature_image_size)
    x_test_scaled = feature_pipeline["scaler"].transform(x_test)
    x_test_reduced = feature_pipeline["pca"].transform(x_test_scaled)
    predictions = model.predict(x_test_reduced)
    probabilities = model.predict_proba(x_test_reduced)
    positive_index = list(model.classes_).index(1)
    positive_probabilities = probabilities[:, positive_index]

    false_negatives = _build_holdout_error_analysis_frame(
        test_df,
        test_df["label"].to_numpy(),
        predictions,
        positive_probabilities,
        image_size=SETTINGS.audio.feature_image_size,
        reduced_feature_dim=x_test_reduced.shape[1],
        decision_threshold=SETTINGS.training.decision_threshold,
    )

    target_dir = paths.project_root / "investigation" / "false_negatives"
    target_dir.mkdir(parents=True, exist_ok=True)
    summary_path = target_dir / "false_negative_summary.csv"

    summary_rows = []
    copied = 0
    for _, row in false_negatives.iterrows():
        source_name = str(row["original_audio"])
        source_matches = list((paths.project_root / "audio_dataset").rglob(source_name))
        if not source_matches:
            source_matches = list(paths.project_root.rglob(source_name))
        if not source_matches:
            raise FileNotFoundError(f"Could not locate source audio for {source_name}")
        source_path = source_matches[0]
        destination_name = f"{int(row['false_negative_rank']):02d}__{Path(str(row['file_name'])).stem}{source_path.suffix}"
        destination_path = target_dir / destination_name
        shutil.copy2(source_path, destination_path)
        copied += 1

        file_size = destination_path.stat().st_size
        waveform, sample_rate = librosa.load(str(destination_path), sr=None, mono=True)
        duration_seconds = float(len(waveform) / sample_rate) if sample_rate else None
        leading_window = waveform[: max(1, int(0.25 * sample_rate))]
        trailing_window = waveform[-max(1, int(0.25 * sample_rate)) :]
        zero_tolerance = 1e-5
        leading_zero_ratio = float(np.mean(np.isclose(leading_window, 0.0, atol=zero_tolerance)))
        trailing_zero_ratio = float(np.mean(np.isclose(trailing_window, 0.0, atol=zero_tolerance)))
        overall_zero_ratio = float(np.mean(np.isclose(waveform, 0.0, atol=zero_tolerance)))
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=sample_rate,
            n_mels=SETTINGS.audio.n_mels,
            fmax=SETTINGS.audio.mel_fmax,
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        summary_rows.append(
            {
                'file_name': row['file_name'],
                'false_negative_rank': int(row['false_negative_rank']),
                'source_audio_name': source_name,
                'source_path': str(source_path),
                'copied_path': str(destination_path),
                'file_size_bytes': int(file_size),
                'duration_seconds': duration_seconds,
                'sample_rate': int(sample_rate),
                'num_samples': int(len(waveform)),
                'positive_probability': float(row['positive_probability']),
                'margin_to_threshold': float(row['margin_to_threshold']),
                'raw_feature_shape': row['raw_feature_shape'],
                'flattened_feature_dim': int(row['flattened_feature_dim']),
                'reduced_feature_dim': int(row['reduced_feature_dim']),
                'mel_feature_shape': f"{mel_db.shape[0]}x{mel_db.shape[1]}",
                'leading_zero_ratio': leading_zero_ratio,
                'trailing_zero_ratio': trailing_zero_ratio,
                'overall_zero_ratio': overall_zero_ratio,
                'obvious_anomaly': bool(
                    duration_seconds < 2.0
                    or leading_zero_ratio > 0.25
                    or trailing_zero_ratio > 0.25
                    or overall_zero_ratio > 0.10
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"Summary CSV written to {summary_path}")
    print(f"Copied {copied} files to {target_dir}")
    print(f"False negatives in analysis: {len(false_negatives)}")


if __name__ == '__main__':
    main()
