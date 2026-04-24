import os
import numpy as np
import librosa
from PIL import Image
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# ===== PATHS =====
TRAIN_DIR  = "Training_data"
OUTPUT_DIR = "SPECTROGRAMS"
LABELS_CSV = "labels.csv"

# ===== CONFIG =====
SAMPLE_RATE = 16000
DURATION    = 4.0
N_MELS      = 128
N_FFT       = 1024
HOP_LENGTH  = 512
F_MIN       = 50.0
F_MAX       = 8000.0
TOP_DB      = 80.0
IMG_SIZE    = 224

LABEL_MAP  = {'human': 0, 'ai': 1}
AUDIO_EXTS = ('.wav', '.mp3', '.flac', '.ogg')


# ===== AUDIO LOAD =====
def load_audio(path):
    n = int(SAMPLE_RATE * DURATION)
    try:
        y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"  [skip] {path}: {e}")
        return None

    if len(y) < SAMPLE_RATE:
        return None

    if len(y) >= n:
        start = (len(y) - n) // 2
        y = y[start:start + n]
    else:
        pad = n - len(y)
        y = np.pad(y, (pad // 2, pad - pad // 2))

    return y.astype(np.float32)


# ===== SPECTROGRAM =====
def make_spectrogram(y, out_path):
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=TOP_DB)
    log_mel -= log_mel.min()
    log_mel /= (log_mel.max() + 1e-9)
    img = Image.fromarray((log_mel * 255).astype(np.uint8))
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


# ===== PROCESS LABELED =====
def process_labeled(records):
    for category, label in LABEL_MAP.items():
        in_dir  = os.path.join(TRAIN_DIR, category)
        out_dir = os.path.join(OUTPUT_DIR, category)
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(in_dir):
            print(f"[warn] missing folder: {in_dir}")
            continue

        files = [f for f in os.listdir(in_dir)
                 if f.lower().endswith(AUDIO_EXTS)]
        print(f"\n{category}: {len(files)} files")

        for fname in files:
            src      = os.path.join(in_dir, fname)
            out_path = os.path.join(out_dir, os.path.splitext(fname)[0] + ".png")

            if os.path.exists(out_path):
                records.append({"file_path": out_path, "label": label})
                continue

            y = load_audio(src)
            if y is None:
                continue

            make_spectrogram(y, out_path)
            records.append({"file_path": out_path, "label": label})


# ===== MAIN =====
def main():
    records = []
    process_labeled(records)

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["file_path"]).reset_index(drop=True)
    df.to_csv(LABELS_CSV, index=False)

    print(f"\nDone. Total: {len(df)} spectrograms")
    if len(df) > 0:
        counts = df['label'].value_counts().to_dict()
        print(f"  human={counts.get(0,0)}  ai={counts.get(1,0)}")
    else:
        print("  No records found — check your folder paths.")


if __name__ == "__main__":
    main()