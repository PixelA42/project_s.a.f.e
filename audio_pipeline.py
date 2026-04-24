"""Audio to spectrogram dataset generation for the ML pipeline."""

from __future__ import annotations

import warnings
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import SETTINGS


warnings.filterwarnings("ignore")

AUDIO_CONFIG = SETTINGS.audio
GENERAL_CONFIG = SETTINGS.general
PATHS = SETTINGS.paths


def load_audio(filepath: Path) -> tuple[np.ndarray | None, int | None]:
    try:
        audio_signal, sample_rate = librosa.load(
            str(filepath),
            sr=AUDIO_CONFIG.sample_rate,
            duration=AUDIO_CONFIG.clip_duration_seconds,
            mono=True,
        )
        if len(audio_signal) < sample_rate:
            print(f"  [SKIP] Too short: {filepath}")
            return None, None
        return audio_signal, sample_rate
    except Exception as exc:
        print(f"  [ERROR] Could not load {filepath}: {exc}")
        return None, None


def audio_to_mel_spectrogram(audio_signal: np.ndarray, sample_rate: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio_signal,
        sr=sample_rate,
        n_mels=AUDIO_CONFIG.n_mels,
        fmax=AUDIO_CONFIG.mel_fmax,
    )
    return librosa.power_to_db(mel, ref=np.max)


def save_spectrogram_image(mel_db: np.ndarray, sample_rate: int, output_path: Path) -> None:
    fig = plt.figure(
        figsize=(AUDIO_CONFIG.figure_size_inches, AUDIO_CONFIG.figure_size_inches),
        dpi=AUDIO_CONFIG.spectrogram_dpi,
    )
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.axis("off")
    librosa.display.specshow(mel_db, sr=sample_rate, cmap=AUDIO_CONFIG.colormap, ax=ax)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def augment_audio(
    audio_signal: np.ndarray,
    sample_rate: int,
    augmentation_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if augmentation_type == "noise":
        noise = rng.normal(
            loc=0.0,
            scale=AUDIO_CONFIG.augmentation_noise_scale,
            size=len(audio_signal),
        )
        return np.clip(audio_signal + noise, -1.0, 1.0)
    if augmentation_type == "pitch_up":
        return librosa.effects.pitch_shift(
            audio_signal,
            sr=sample_rate,
            n_steps=AUDIO_CONFIG.augmentation_pitch_up_steps,
        )
    if augmentation_type == "pitch_down":
        return librosa.effects.pitch_shift(
            audio_signal,
            sr=sample_rate,
            n_steps=AUDIO_CONFIG.augmentation_pitch_down_steps,
        )
    if augmentation_type == "stretch":
        stretched = librosa.effects.time_stretch(
            audio_signal,
            rate=AUDIO_CONFIG.augmentation_time_stretch_rate,
        )
        if len(stretched) > len(audio_signal):
            return stretched[: len(audio_signal)]
        return np.pad(stretched, (0, len(audio_signal) - len(stretched)))
    if augmentation_type == "volume_up":
        return np.clip(audio_signal * AUDIO_CONFIG.augmentation_volume_up_gain, -1.0, 1.0)
    if augmentation_type == "volume_down":
        return audio_signal * AUDIO_CONFIG.augmentation_volume_down_gain
    return audio_signal


def process_dataset() -> list[dict[str, str | int]]:
    records: list[dict[str, str | int]] = []
    rng = np.random.default_rng(GENERAL_CONFIG.random_seed)
    augmentation_types = AUDIO_CONFIG.augmentation_types[: AUDIO_CONFIG.augmentation_copies]

    for category in AUDIO_CONFIG.categories:
        label = AUDIO_CONFIG.label_map[category]
        audio_folder = PATHS.audio_dataset_dir / category
        spectrogram_folder = PATHS.spectrogram_dir / category
        augmented_folder = PATHS.spectrogram_dir / "augmented" / category

        spectrogram_folder.mkdir(parents=True, exist_ok=True)
        if AUDIO_CONFIG.augmentation_enabled:
            augmented_folder.mkdir(parents=True, exist_ok=True)

        if not audio_folder.exists():
            print(f"\n[WARNING] Folder not found: {audio_folder}")
            print(f"  Please create it and add your {category} audio files.")
            continue

        audio_files = sorted(
            file_path
            for file_path in audio_folder.iterdir()
            if file_path.is_file() and file_path.suffix.lower() in AUDIO_CONFIG.supported_extensions
        )

        if not audio_files:
            print(f"\n[WARNING] No audio files found in {audio_folder}")
            continue

        print(f"\n{'=' * 50}")
        print(f"  Processing {category.upper()} audio ({len(audio_files)} files)")
        print(f"{'=' * 50}")

        for index, filepath in enumerate(audio_files, start=1):
            print(f"  [{index}/{len(audio_files)}] {filepath.name}")

            audio_signal, sample_rate = load_audio(filepath)
            if audio_signal is None:
                continue

            mel_db = audio_to_mel_spectrogram(audio_signal, sample_rate)
            output_path = spectrogram_folder / f"{filepath.stem}.png"
            save_spectrogram_image(mel_db, sample_rate, output_path)

            records.append(
                {
                    "file_name": output_path.name,
                    "file_path": str(output_path),
                    "label": label,
                    "category": category,
                    "source": "original",
                    "aug_type": "none",
                    "original_audio": filepath.name,
                }
            )
            print("    [OK] Spectrogram saved")

            if AUDIO_CONFIG.augmentation_enabled:
                for augmentation_type in augmentation_types:
                    augmented_signal = augment_audio(
                        audio_signal,
                        sample_rate,
                        augmentation_type,
                        rng,
                    )
                    augmented_mel = audio_to_mel_spectrogram(augmented_signal, sample_rate)
                    augmented_path = augmented_folder / f"{filepath.stem}_{augmentation_type}.png"
                    save_spectrogram_image(augmented_mel, sample_rate, augmented_path)

                    records.append(
                        {
                            "file_name": augmented_path.name,
                            "file_path": str(augmented_path),
                            "label": label,
                            "category": category,
                            "source": "augmented",
                            "aug_type": augmentation_type,
                            "original_audio": filepath.name,
                        }
                    )
                print(f"    [OK] {len(augmentation_types)} augmented versions saved")

    return records


def save_labels(records: list[dict[str, str | int]]) -> pd.DataFrame:
    dataframe = pd.DataFrame(records)
    dataframe.to_csv(PATHS.labels_csv_path, index=False)
    print(f"\n[OK] Labels saved to {PATHS.labels_csv_path} ({len(dataframe)} total records)")
    return dataframe


def print_summary(dataframe: pd.DataFrame) -> None:
    lines = [
        "=" * 50,
        "  DATASET SUMMARY",
        "=" * 50,
        f"  Total samples       : {len(dataframe)}",
        f"  Human (label=0)     : {len(dataframe[dataframe['label'] == 0])}",
        f"  AI    (label=1)     : {len(dataframe[dataframe['label'] == 1])}",
        f"  Original samples    : {len(dataframe[dataframe['source'] == 'original'])}",
        f"  Augmented samples   : {len(dataframe[dataframe['source'] == 'augmented'])}",
        "",
        "  Augmentation breakdown:",
    ]

    for augmentation_type in dataframe["aug_type"].unique():
        count = len(dataframe[dataframe["aug_type"] == augmentation_type])
        lines.append(f"    {augmentation_type:<15}: {count}")

    lines.extend(
        [
            "",
            "  Output folders:",
            f"    Spectrograms  -> {PATHS.spectrogram_dir}",
            f"    Labels CSV    -> {PATHS.labels_csv_path}",
            "=" * 50,
        ]
    )

    summary = "\n".join(lines)
    print(summary)
    PATHS.dataset_stats_path.write_text(summary, encoding="utf-8")
    print(f"\n[OK] Stats saved to {PATHS.dataset_stats_path}")


if __name__ == "__main__":
    print(f"\n{'=' * 50}")
    print("  AUDIO TO SPECTROGRAM PIPELINE")
    print("  Human vs AI Audio Detection Dataset")
    print("=" * 50)

    if not PATHS.audio_dataset_dir.exists():
        print(f"\n[ERROR] Audio directory '{PATHS.audio_dataset_dir}' not found.")
        print("Please create this structure:")
        category_lines = "\n".join(f"|- {category}/" for category in AUDIO_CONFIG.categories)
        print(f"\n{PATHS.audio_dataset_dir}\n{category_lines}\n")
        raise SystemExit(1)

    dataset_records = process_dataset()

    if not dataset_records:
        print("\n[ERROR] No files were processed. Check your audio folder.")
        raise SystemExit(1)

    labels_dataframe = save_labels(dataset_records)
    print_summary(labels_dataframe)
    print("\nPipeline complete. Dataset artifacts are ready for training.\n")
