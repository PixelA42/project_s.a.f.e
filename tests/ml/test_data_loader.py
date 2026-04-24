from __future__ import annotations

from pathlib import Path
import wave

import numpy as np
import pandas as pd
import pytest

import data_loader


def _write_wav(path: Path, *, sample_rate: int = 16000, duration_seconds: float = 0.25) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_count = max(1, int(sample_rate * duration_seconds))
    time_axis = np.linspace(0, duration_seconds, sample_count, endpoint=False)
    waveform = (0.2 * np.sin(2 * np.pi * 220.0 * time_axis) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(waveform.tobytes())


def test_load_data_reads_audio_directory_from_config(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "Audios"
    _write_wav(dataset_dir / "real_samples" / "real_1.wav", sample_rate=16000, duration_seconds=0.25)
    _write_wav(dataset_dir / "real_samples" / "real_2.wav", sample_rate=22050, duration_seconds=0.30)
    _write_wav(dataset_dir / "OpenAI" / "synthetic_1.wav", sample_rate=16000, duration_seconds=0.20)
    _write_wav(dataset_dir / "xTTS" / "synthetic_2.wav", sample_rate=22050, duration_seconds=0.35)
    _write_wav(dataset_dir / "unlabeled" / "held_out.wav", sample_rate=16000, duration_seconds=0.40)

    patched_paths = type("Paths", (), {"audio_dataset_dir": dataset_dir})()
    monkeypatch.setattr(data_loader, "PATHS", patched_paths)

    loaded = data_loader.load_data()

    assert loaded.source_type == "audio_directory"
    assert loaded.target_column == "target"
    assert loaded.X_train.shape[0] + loaded.X_test.shape[0] == 4
    assert loaded.X_full.shape[0] == 5
    assert set(loaded.y_train).union(set(loaded.y_test)) == {"real", "synthetic"}
    assert loaded.feature_names
    assert loaded.metadata["unlabeled_row_count"] == 1


def test_load_data_applies_null_strategies_for_tabular_dataset(tmp_path):
    dataset_file = tmp_path / "dataset.csv"
    dataframe = pd.DataFrame(
        {
            "target": ["real", "real", "synthetic", "synthetic", "real", "synthetic"],
            "duration_seconds": [1.0, np.nan, 1.4, 1.6, np.nan, 2.1],
            "sample_rate": [16000, 16000, 22050, 22050, 16000, 22050],
            "codec": ["pcm", None, "pcm", "pcm", "pcm", "pcm"],
            "mostly_missing": [None, None, None, "x", None, None],
        }
    )
    dataframe.to_csv(dataset_file, index=False)

    loaded = data_loader.load_data(dataset_path=dataset_file, test_size=0.33, random_state=7)

    assert loaded.source_type == "tabular_file"
    assert loaded.target_column == "target"
    assert loaded.null_strategy_by_column["duration_seconds"] == "interpolate_then_fill_median"
    assert loaded.null_strategy_by_column["codec"] == "fill_mode"
    assert loaded.null_strategy_by_column["mostly_missing"] == "drop_column"
    assert loaded.dropped_columns["mostly_missing"] == "dropped_high_null_categorical"
    assert loaded.X_full.shape[0] == len(dataframe)
    assert not np.isnan(loaded.X_train).any()
    assert not np.isnan(loaded.X_test).any()


def test_load_data_rejects_empty_dataset_directory(tmp_path):
    empty_dir = tmp_path / "Audios"
    empty_dir.mkdir()

    with pytest.raises(data_loader.DataLoadingError, match="No supported dataset files"):
        data_loader.load_data(dataset_path=empty_dir)


def test_load_data_rejects_missing_target_column(tmp_path):
    dataset_file = tmp_path / "dataset.csv"
    dataframe = pd.DataFrame(
        {
            "file_id": [f"row_{index}" for index in range(6)],
            "duration_seconds": np.linspace(0.2, 1.2, 6),
            "sample_rate": [16000, 16000, 22050, 22050, 44100, 44100],
        }
    )
    dataframe.to_csv(dataset_file, index=False)

    with pytest.raises(data_loader.DataLoadingError, match="No valid target column could be inferred"):
        data_loader.load_data(dataset_path=dataset_file)
