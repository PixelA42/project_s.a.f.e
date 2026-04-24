"""Run one-off spectrogram inference against the trained baseline model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import librosa
import librosa.display
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from config import SETTINGS


AUDIO_CONFIG = SETTINGS.audio
PATHS = SETTINGS.paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict HUMAN/AI from one audio file")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to audio file (.wav/.mp3/.flac/.m4a)",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep generated temp spectrogram image",
    )
    return parser.parse_args()


def resize_grayscale_image(image: np.ndarray, size: int) -> np.ndarray:
    if image.ndim == 3:
        image = image.mean(axis=2)
    height, width = image.shape
    row_idx = np.linspace(0, max(0, height - 1), size).astype(int)
    col_idx = np.linspace(0, max(0, width - 1), size).astype(int)
    return image[np.ix_(row_idx, col_idx)]


def main() -> None:
    args = parse_args()
    input_audio = Path(args.input).expanduser()

    if not input_audio.is_absolute():
        input_audio = (PATHS.project_root / input_audio).resolve()

    if not input_audio.is_file():
        raise FileNotFoundError(
            f"Input audio not found: {input_audio}\n"
            "Pass a real file path, for example:\n"
            "python quick_predict.py --input path/to/audio.wav"
        )

    audio_signal, sample_rate = librosa.load(
        str(input_audio),
        sr=AUDIO_CONFIG.sample_rate,
        duration=AUDIO_CONFIG.clip_duration_seconds,
        mono=True,
    )
    mel = librosa.feature.melspectrogram(
        y=audio_signal,
        sr=sample_rate,
        n_mels=AUDIO_CONFIG.n_mels,
        fmax=AUDIO_CONFIG.mel_fmax,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    figure = plt.figure(
        figsize=(AUDIO_CONFIG.figure_size_inches, AUDIO_CONFIG.figure_size_inches),
        dpi=AUDIO_CONFIG.spectrogram_dpi,
    )
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axis.axis("off")
    librosa.display.specshow(mel_db, sr=sample_rate, cmap=AUDIO_CONFIG.colormap, ax=axis)
    plt.savefig(PATHS.temp_spectrogram_path, bbox_inches="tight", pad_inches=0)
    plt.close(figure)

    image = mpimg.imread(PATHS.temp_spectrogram_path).astype(np.float32)
    image = resize_grayscale_image(image, AUDIO_CONFIG.feature_image_size)
    features = image.flatten().reshape(1, -1)

    feature_pipeline = joblib.load(PATHS.feature_pipeline_path)
    model = joblib.load(PATHS.supervised_model_path)

    features_scaled = feature_pipeline["scaler"].transform(features)
    features_reduced = feature_pipeline["pca"].transform(features_scaled)

    prediction = int(model.predict(features_reduced)[0])
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features_reduced)[0]
        classes = list(model.classes_)
        probability_map = {int(label): float(prob) for label, prob in zip(classes, probabilities)}
    else:
        probability_map = {}

    label_text = "AI" if prediction == 1 else "HUMAN"

    print("Input:", input_audio)
    print("Prediction:", prediction, f"({label_text})")
    print("Probabilities:", probability_map)
    print("Temp spectrogram:", PATHS.temp_spectrogram_path)

    if not args.keep_temp and PATHS.temp_spectrogram_path.exists():
        PATHS.temp_spectrogram_path.unlink()


if __name__ == "__main__":
    main()
