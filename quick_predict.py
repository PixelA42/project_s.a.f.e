# save as quick_predict.py (in project root), then run with env python
import argparse
from pathlib import Path
import numpy as np
import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PROJECT_ROOT = Path(__file__).resolve().parent
TMP_SPEC = PROJECT_ROOT / "tmp_uploaded_spec.png"

# Match training preprocessing from audio_pipeline.py and train_model.py
SAMPLE_RATE = 22050
DURATION = 5
N_MELS = 128
IMAGE_SIZE = 64  # train_model default image-size for features


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

def resize_grayscale_image(img: np.ndarray, size: int) -> np.ndarray:
    if img.ndim == 3:
        img = img.mean(axis=2)
    h, w = img.shape
    row_idx = np.linspace(0, max(0, h - 1), size).astype(int)
    col_idx = np.linspace(0, max(0, w - 1), size).astype(int)
    return img[np.ix_(row_idx, col_idx)]


def main() -> None:
    args = parse_args()
    input_audio = Path(args.input).expanduser()

    if not input_audio.is_absolute():
        input_audio = (PROJECT_ROOT / input_audio).resolve()

    if not input_audio.is_file():
        raise FileNotFoundError(
            f"Input audio not found: {input_audio}\n"
            "Pass a real file path, for example:\n"
            "python quick_predict.py --input audio_dataset/human/hr10.m4a.mp3"
        )

    # 1) audio -> mel spectrogram image
    y, sr = librosa.load(str(input_audio), sr=SAMPLE_RATE, duration=DURATION, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    dpi = 224 / 2.24
    fig = plt.figure(figsize=(2.24, 2.24), dpi=dpi)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.axis("off")
    librosa.display.specshow(mel_db, sr=sr, cmap="magma", ax=ax)
    plt.savefig(TMP_SPEC, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # 2) extract image features exactly like training
    img = mpimg.imread(TMP_SPEC).astype(np.float32)
    img = resize_grayscale_image(img, IMAGE_SIZE)
    x = img.flatten().reshape(1, -1)

    # 3) load pipeline + model and predict
    feature_pipe = joblib.load(PROJECT_ROOT / "models" / "spectrogram_feature_pipeline.joblib")
    model = joblib.load(PROJECT_ROOT / "models" / "spectrogram_supervised_model.joblib")

    x_scaled = feature_pipe["scaler"].transform(x)
    x_reduced = feature_pipe["pca"].transform(x_scaled)

    pred = int(model.predict(x_reduced)[0])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x_reduced)[0]
        classes = list(model.classes_)
        prob_map = {int(c): float(p) for c, p in zip(classes, probs)}
    else:
        prob_map = {}

    label_text = "AI" if pred == 1 else "HUMAN"

    print("Input:", input_audio)
    print("Prediction:", pred, f"({label_text})")
    print("Probabilities:", prob_map)
    print("Temp spectrogram:", TMP_SPEC)

    if not args.keep_temp and TMP_SPEC.exists():
        TMP_SPEC.unlink()


if __name__ == "__main__":
    main()