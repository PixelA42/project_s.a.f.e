import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf
import warnings

warnings.filterwarnings('ignore')

AUDIO_DIR = "Testing_data"
OUTPUT_DIR = "SPECTROGRAMS"
LABELS_CSV = "labels.csv"
STATS_FILE = "dataset_stats.txt"

DURATION = 5
SAMPLE_RATE = 22050
N_MELS = 128
IMAGE_SIZE = 224

DO_AUGMENTATION = True
AUGMENT_COPIES = 4

COLORMAP = 'magma'

LABEL_MAP = {'human': 0, 'ai': 1}

AUG_TYPES = ['noise', 'pitch_up', 'pitch_down', 'stretch', 'volume_up', 'volume_down']

def load_audio(filepath):
    try:
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION, mono=True)
        if len(y) < sr:
            print(f"  [SKIP] Too short: {filepath}")
            return None, None
        return y, sr
    except Exception as e:
        print(f"  [ERROR] Could not load {filepath}: {e}")
        return None, None

def audio_to_mel_spectrogram(y, sr):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def save_spectrogram_image(mel_db, sr, output_path):
    dpi = IMAGE_SIZE / 2.24
    fig = plt.figure(figsize=(2.24, 2.24), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    librosa.display.specshow(mel_db, sr=sr, cmap=COLORMAP, ax=ax)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def augment_audio(y, sr, aug_type):
    if aug_type == 'noise':
        noise = np.random.randn(len(y)) * 0.005
        return np.clip(y + noise, -1.0, 1.0)
    elif aug_type == 'pitch_up':
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    elif aug_type == 'pitch_down':
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)
    elif aug_type == 'stretch':
        y_stretch = librosa.effects.time_stretch(y, rate=0.85)
        if len(y_stretch) > len(y):
            return y_stretch[:len(y)]
        else:
            return np.pad(y_stretch, (0, len(y) - len(y_stretch)))
    elif aug_type == 'volume_up':
        return np.clip(y * 1.5, -1.0, 1.0)
    elif aug_type == 'volume_down':
        return y * 0.5
    else:
        return y

def process_dataset():
    records = []

    for category in ['human', 'ai']:
        label = LABEL_MAP[category]
        audio_folder = os.path.join(AUDIO_DIR, category)
        spec_folder = os.path.join(OUTPUT_DIR, category)
        aug_folder = os.path.join(OUTPUT_DIR, 'augmented', category)

        os.makedirs(spec_folder, exist_ok=True)
        if DO_AUGMENTATION:
            os.makedirs(aug_folder, exist_ok=True)

        if not os.path.exists(audio_folder):
            print(f"\n[WARNING] Folder not found: {audio_folder}")
            print(f"  Please create it and add your {category} audio files.")
            continue

        audio_files = [
            f for f in os.listdir(audio_folder)
            if f.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a'))
        ]

        if not audio_files:
            print(f"\n[WARNING] No audio files found in {audio_folder}")
            continue

        print(f"\n{'='*50}")
        print(f"  Processing {category.upper()} audio ({len(audio_files)} files)")
        print(f"{'='*50}")

        for i, filename in enumerate(audio_files, 1):
            filepath = os.path.join(audio_folder, filename)
            base_name = os.path.splitext(filename)[0]

            print(f"  [{i}/{len(audio_files)}] {filename}")

            y, sr = load_audio(filepath)
            if y is None:
                continue

            out_filename = f"{base_name}.png"
            out_path = os.path.join(spec_folder, out_filename)
            mel_db = audio_to_mel_spectrogram(y, sr)
            save_spectrogram_image(mel_db, sr, out_path)

            records.append({
                'file_name': out_filename,
                'file_path': out_path,
                'label': label,
                'category': category,
                'source': 'original',
                'aug_type': 'none',
                'original_audio': filename
            })
            print(f"    ✓ Spectrogram saved")

            if DO_AUGMENTATION:
                aug_types_to_use = AUG_TYPES[:AUGMENT_COPIES]
                for aug_type in aug_types_to_use:
                    y_aug = augment_audio(y, sr, aug_type)
                    mel_aug = audio_to_mel_spectrogram(y_aug, sr)

                    aug_filename = f"{base_name}_{aug_type}.png"
                    aug_path = os.path.join(aug_folder, aug_filename)
                    save_spectrogram_image(mel_aug, sr, aug_path)

                    records.append({
                        'file_name': aug_filename,
                        'file_path': aug_path,
                        'label': label,
                        'category': category,
                        'source': 'augmented',
                        'aug_type': aug_type,
                        'original_audio': filename
                    })
                print(f"    ✓ {len(aug_types_to_use)} augmented versions saved")

    return records

def save_labels(records):
    df = pd.DataFrame(records)
    df.to_csv(LABELS_CSV, index=False)
    print(f"\n✅ Labels saved to {LABELS_CSV} ({len(df)} total records)")
    return df

def print_summary(df):
    lines = []
    lines.append("=" * 50)
    lines.append("  DATASET SUMMARY")
    lines.append("=" * 50)
    lines.append(f"  Total samples       : {len(df)}")
    lines.append(f"  Human (label=0)     : {len(df[df['label']==0])}")
    lines.append(f"  AI    (label=1)     : {len(df[df['label']==1])}")
    lines.append(f"  Original samples    : {len(df[df['source']=='original'])}")
    lines.append(f"  Augmented samples   : {len(df[df['source']=='augmented'])}")
    lines.append("")
    lines.append("  Augmentation breakdown:")
    for aug_type in df['aug_type'].unique():
        count = len(df[df['aug_type'] == aug_type])
        lines.append(f"    {aug_type:<15}: {count}")
    lines.append("")
    lines.append("  Output folders:")
    lines.append(f"    Spectrograms  → {OUTPUT_DIR}/")
    lines.append(f"    Labels CSV    → {LABELS_CSV}")
    lines.append("=" * 50)

    summary = "\n".join(lines)
    print(summary)

    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"\n✅ Stats saved to {STATS_FILE}")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  AUDIO → SPECTROGRAM PIPELINE")
    print("  Human vs AI Audio Detection Dataset")
    print("="*50)

    if not os.path.exists(AUDIO_DIR):
        print(f"\n[ERROR] Audio directory '{AUDIO_DIR}' not found!")
        print("Please create this structure:")
        print(f"\n{AUDIO_DIR}/\n├── human/\n└── ai/\n")
        exit(1)

    records = process_dataset()

    if not records:
        print("\n[ERROR] No files were processed. Check your audio folder.")
        exit(1)

    df = save_labels(records)
    print_summary(df)
    print("\n🎉 Pipeline complete! Your dataset is ready for ML training.\n")