"""Unsupervised anomaly detector for Project S.A.F.E.

Trains an Autoencoder (PyTorch) and an Isolation Forest (scikit-learn)
exclusively on Class 0 (authentic/human) audio features to learn the
baseline distribution of real human speech. No deepfake labels are used.

At inference time, reconstruction error (Autoencoder) and anomaly score
(Isolation Forest) together form the unsupervised anomaly signal that
complements the supervised PyTorch classifier in the hybrid pipeline.

Usage
-----
Train and save models:
    python train_unsupervised.py

Optional flags:
    --labels-csv   Path to labels.csv  (default: config value)
    --models-dir   Directory to save artifacts  (default: models/)
    --epochs       Autoencoder training epochs  (default: 60)
    --batch-size   Mini-batch size  (default: 64)
    --latent-dim   Autoencoder bottleneck size  (default: 32)
    --device       "cpu" or "cuda"  (default: cpu)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    raise SystemExit(
        "PyTorch is not installed in the active Python interpreter.\n"
        "Use the project Python 3.11.9 environment instead:\n"
        r"  .\envSafeVone\Scripts\python.exe train_unsupervised.py"
        "\nOr install project dependencies with:\n"
        r"  .\envSafeVone\Scripts\python.exe -m pip install -r requirements.txt"
    ) from exc
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from config import SETTINGS


# ---------------------------------------------------------------------------
# Config shortcuts
# ---------------------------------------------------------------------------
PATHS = SETTINGS.paths
GENERAL_CONFIG = SETTINGS.general
AUDIO_CONFIG = SETTINGS.audio

# Paths for saved artifacts
AUTOENCODER_WEIGHTS_PATH = PATHS.models_dir / "unsupervised_autoencoder.pt"
ISOLATION_FOREST_PATH = PATHS.models_dir / "unsupervised_isolation_forest.joblib"
SCALER_PATH = PATHS.models_dir / "unsupervised_scaler.joblib"

# Label value for authentic (human) class
AUTHENTIC_LABEL: int = 0


def unsupervised_artifact_paths(models_dir: Path) -> dict[str, Path]:
    """Return all artifact paths for a selected model directory."""
    return {
        "autoencoder": models_dir / "unsupervised_autoencoder.pt",
        "isolation_forest": models_dir / "unsupervised_isolation_forest.joblib",
        "scaler": models_dir / "unsupervised_scaler.joblib",
        "report": models_dir / "unsupervised_training_report.json",
    }


# ---------------------------------------------------------------------------
# Autoencoder architecture
# ---------------------------------------------------------------------------

class SpectralAutoencoder(nn.Module):
    """Fully-connected autoencoder for flattened spectrogram features.

    Trained only on authentic (Class 0) samples.  At inference time,
    high reconstruction error signals that the input deviates from the
    learned real-speech manifold, which can indicate a novel deepfake.
    """

    def __init__(self, input_dim: int, latent_dim: int = 32) -> None:
        super().__init__()
        hidden = max(latent_dim * 4, 128)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-sample mean-squared reconstruction error."""
        with torch.no_grad():
            reconstructed = self.forward(x)
            return torch.mean((x - reconstructed) ** 2, dim=1)


# ---------------------------------------------------------------------------
# Feature loading helpers
# ---------------------------------------------------------------------------

def _load_spectrogram_features(
    file_path: str,
    image_size: int,
) -> np.ndarray | None:
    """Load a spectrogram PNG and return a flattened float32 feature vector."""
    try:
        img = mpimg.imread(file_path)
        # Convert to grayscale if RGB/RGBA
        if img.ndim == 3:
            img = img.mean(axis=2)
        if img.ndim != 2:
            raise ValueError(f"Expected a 2D or 3D spectrogram image, got shape {img.shape}")

        from PIL import Image  # lazy import, only needed here

        img = np.asarray(img, dtype=np.float32)
        if img.size == 0:
            raise ValueError("empty image")
        if img.max() <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0.0, 255.0).astype(np.uint8)

        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((image_size, image_size), Image.BILINEAR)
        arr = np.array(pil_img, dtype=np.float32) / 255.0
        return arr.flatten()
    except Exception as exc:
        print(f"  [WARN] Could not load {file_path}: {exc}")
        return None


def load_authentic_features(
    labels_csv_path: Path,
    image_size: int = AUDIO_CONFIG.feature_image_size,
) -> np.ndarray:
    """Load spectrogram features for Class 0 (authentic) samples only.

    This is the core unsupervised constraint: the model never sees deepfake
    labels during training. It only learns what real speech looks like.
    """
    labels_csv_path = labels_csv_path.expanduser().resolve()
    df = pd.read_csv(labels_csv_path)
    required_columns = {"label", "file_path"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"labels CSV missing required columns: {sorted(missing)}")

    # Keep only authentic (human) samples
    authentic_df = df[df["label"] == AUTHENTIC_LABEL].copy()
    print(f"  Authentic samples found : {len(authentic_df)}")

    features: list[np.ndarray] = []
    for _, row in authentic_df.iterrows():
        spectrogram_path = Path(str(row["file_path"])).expanduser()
        if not spectrogram_path.is_absolute():
            csv_relative = labels_csv_path.parent / spectrogram_path
            project_relative = PATHS.project_root / spectrogram_path
            spectrogram_path = csv_relative if csv_relative.exists() else project_relative

        vec = _load_spectrogram_features(str(spectrogram_path), image_size)
        if vec is not None:
            features.append(vec)

    if not features:
        raise RuntimeError(
            "No authentic spectrogram features could be loaded. "
            "Run audio_pipeline.py first to generate spectrograms."
        )

    X = np.stack(features, axis=0)
    print(f"  Feature matrix shape    : {X.shape}")
    return X


# ---------------------------------------------------------------------------
# Training routines
# ---------------------------------------------------------------------------

def train_autoencoder(
    X_scaled: np.ndarray,
    input_dim: int,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
    random_state: int,
) -> tuple[SpectralAutoencoder, list[float]]:
    """Train the autoencoder on authentic features and return (model, loss_history)."""
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    model = SpectralAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    loss_history: list[float] = []
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        scheduler.step()
        avg_loss = epoch_loss / len(X_tensor)
        loss_history.append(avg_loss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>4}/{epochs}  loss={avg_loss:.6f}")

    model.eval()
    return model, loss_history


def train_isolation_forest(
    X_scaled: np.ndarray,
    random_state: int,
    n_estimators: int = 200,
    contamination: float = 0.05,
) -> IsolationForest:
    """Fit an Isolation Forest on authentic features.

    contamination=0.05 means we expect ~5% of the authentic training set
    to be borderline/noisy, a conservative estimate that keeps the
    decision boundary tight around the real-speech distribution.
    """
    iso_forest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    iso_forest.fit(X_scaled)
    return iso_forest


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------

def calibrate_thresholds(
    autoencoder: SpectralAutoencoder,
    X_scaled: np.ndarray,
    device: torch.device,
    std_multiplier: float = 2.0,
) -> dict[str, float]:
    """Compute anomaly thresholds from the authentic training distribution.

    Threshold = mean + std_multiplier * std of reconstruction errors on
    authentic samples.  Samples exceeding this threshold at inference time
    are flagged as anomalous.
    """
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    errors = autoencoder.reconstruction_error(X_tensor).cpu().numpy()

    mean_err = float(np.mean(errors))
    std_err = float(np.std(errors))
    threshold = mean_err + std_multiplier * std_err

    return {
        "mean_reconstruction_error": mean_err,
        "std_reconstruction_error": std_err,
        "anomaly_threshold": threshold,
        "std_multiplier": std_multiplier,
    }


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_artifacts(
    autoencoder: SpectralAutoencoder,
    iso_forest: IsolationForest,
    scaler: StandardScaler,
    thresholds: dict[str, float],
    input_dim: int,
    latent_dim: int,
    loss_history: list[float],
    models_dir: Path,
    report_path: Path,
) -> None:
    """Serialize all unsupervised artifacts to disk."""
    models_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = unsupervised_artifact_paths(models_dir)

    # Save autoencoder weights
    torch.save(
        {
            "model_state_dict": autoencoder.state_dict(),
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "thresholds": thresholds,
        },
        str(artifact_paths["autoencoder"]),
    )
    print(f"  [OK] Autoencoder saved  -> {artifact_paths['autoencoder']}")

    # Save Isolation Forest + scaler
    joblib.dump(iso_forest, str(artifact_paths["isolation_forest"]))
    joblib.dump(scaler, str(artifact_paths["scaler"]))
    print(f"  [OK] Isolation Forest   -> {artifact_paths['isolation_forest']}")
    print(f"  [OK] Scaler             -> {artifact_paths['scaler']}")

    # Write training report
    report: dict[str, Any] = {
        "model_type": "hybrid_unsupervised",
        "components": ["SpectralAutoencoder", "IsolationForest"],
        "training_class": "authentic_only (label=0)",
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "autoencoder": {
            "final_loss": loss_history[-1] if loss_history else None,
            "epochs_trained": len(loss_history),
            **thresholds,
        },
        "isolation_forest": {
            "n_estimators": iso_forest.n_estimators,
            "contamination": iso_forest.contamination,
        },
        "artifacts": {
            "autoencoder_weights": str(artifact_paths["autoencoder"]),
            "isolation_forest": str(artifact_paths["isolation_forest"]),
            "scaler": str(artifact_paths["scaler"]),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"  [OK] Training report    -> {report_path}")


# ---------------------------------------------------------------------------
# Public inference API (used by hybrid pipeline)
# ---------------------------------------------------------------------------

class UnsupervisedAnomalyDetector:
    """Loads trained unsupervised artifacts and scores new audio samples.

    This class is the runtime interface consumed by the hybrid inference
    pipeline in quick_predict.py and api.py.  It is intentionally kept
    separate from the training code so it can be imported without
    triggering any training side-effects.
    """

    def __init__(
        self,
        autoencoder_path: str | Path = AUTOENCODER_WEIGHTS_PATH,
        iso_forest_path: str | Path = ISOLATION_FOREST_PATH,
        scaler_path: str | Path = SCALER_PATH,
        device: str = "cpu",
        image_size: int = AUDIO_CONFIG.feature_image_size,
    ) -> None:
        self.device = torch.device(device)
        self.image_size = image_size
        self._autoencoder: SpectralAutoencoder | None = None
        self._iso_forest: IsolationForest | None = None
        self._scaler: StandardScaler | None = None
        self._thresholds: dict[str, float] = {}

        self._load_artifacts(
            Path(autoencoder_path),
            Path(iso_forest_path),
            Path(scaler_path),
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_artifacts(
        self,
        ae_path: Path,
        iso_path: Path,
        scaler_path: Path,
    ) -> None:
        """Load all three artifacts; silently skip if not yet trained."""
        if ae_path.exists():
            checkpoint = torch.load(str(ae_path), map_location=self.device)
            input_dim: int = checkpoint["input_dim"]
            latent_dim: int = checkpoint["latent_dim"]
            self._thresholds = checkpoint.get("thresholds", {})
            self._autoencoder = SpectralAutoencoder(input_dim, latent_dim).to(self.device)
            self._autoencoder.load_state_dict(checkpoint["model_state_dict"])
            self._autoencoder.eval()

        if iso_path.exists():
            self._iso_forest = joblib.load(str(iso_path))

        if scaler_path.exists():
            self._scaler = joblib.load(str(scaler_path))

    @property
    def is_ready(self) -> bool:
        """True when all three artifacts are loaded and ready for inference."""
        return (
            self._autoencoder is not None
            and self._iso_forest is not None
            and self._scaler is not None
        )

    # ------------------------------------------------------------------
    # Feature extraction (mirrors training pipeline)
    # ------------------------------------------------------------------

    def _extract_features(self, spectrogram_path: str) -> np.ndarray | None:
        """Extract and scale features from a spectrogram PNG."""
        vec = _load_spectrogram_features(spectrogram_path, self.image_size)
        if vec is None:
            return None
        if self._scaler is not None:
            vec = self._scaler.transform(vec.reshape(1, -1))[0]
        return vec

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, spectrogram_path: str) -> dict[str, Any]:
        """Compute unsupervised anomaly scores for a spectrogram image.

        Returns
        -------
        dict with keys:
            reconstruction_error : autoencoder MSE (lower = more authentic)
            isolation_score      : IF anomaly score (more negative = more anomalous)
            anomaly_flag         : True if either model flags the sample
            ae_anomaly           : autoencoder-specific flag
            if_anomaly           : isolation forest-specific flag
            unsupervised_ready   : False when artifacts are not loaded
        """
        if not self.is_ready:
            return {
                "reconstruction_error": 0.0,
                "isolation_score": 0.0,
                "anomaly_flag": False,
                "ae_anomaly": False,
                "if_anomaly": False,
                "unsupervised_ready": False,
            }

        features = self._extract_features(spectrogram_path)
        if features is None:
            return {
                "reconstruction_error": 0.0,
                "isolation_score": 0.0,
                "anomaly_flag": False,
                "ae_anomaly": False,
                "if_anomaly": False,
                "unsupervised_ready": True,
            }

        # --- Autoencoder reconstruction error ---
        x_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        recon_error = float(self._autoencoder.reconstruction_error(x_tensor).item())  # type: ignore[union-attr]
        ae_threshold = self._thresholds.get("anomaly_threshold", float("inf"))
        ae_anomaly = recon_error > ae_threshold

        # --- Isolation Forest score ---
        # score_samples returns the anomaly score; more negative = more anomalous
        iso_score = float(self._iso_forest.score_samples(features.reshape(1, -1))[0])  # type: ignore[union-attr]
        # IsolationForest.predict returns -1 for anomalies, +1 for inliers
        if_anomaly = int(self._iso_forest.predict(features.reshape(1, -1))[0]) == -1  # type: ignore[union-attr]

        anomaly_flag = ae_anomaly or if_anomaly

        return {
            "reconstruction_error": recon_error,
            "isolation_score": iso_score,
            "anomaly_flag": anomaly_flag,
            "ae_anomaly": ae_anomaly,
            "if_anomaly": if_anomaly,
            "unsupervised_ready": True,
        }

    def reconstruction_error_to_probability(self, recon_error: float) -> float:
        """Map reconstruction error to a [0, 1] anomaly probability.

        Uses a sigmoid centred on the calibrated threshold so that:
          - error == threshold -> probability near 0.5
          - error >> threshold -> probability approaches 1.0 (very anomalous)
          - error << threshold -> probability approaches 0.0 (very authentic)
        """
        threshold = self._thresholds.get("anomaly_threshold", 1.0)
        std = self._thresholds.get("std_reconstruction_error", 1.0) or 1.0
        z = (recon_error - threshold) / std
        return float(1.0 / (1.0 + np.exp(-z)))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train unsupervised anomaly detector on authentic (Class 0) audio features."
    )
    parser.add_argument(
        "--labels-csv",
        default=str(PATHS.labels_csv_path),
        help="Path to labels.csv (default: config value)",
    )
    parser.add_argument(
        "--models-dir",
        default=str(PATHS.models_dir),
        help="Directory to save model artifacts",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Autoencoder training epochs (default: 60)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size (default: 64)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=32,
        help="Autoencoder bottleneck dimension (default: 32)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Compute device (default: cpu)",
    )
    parser.add_argument(
        "--std-multiplier",
        type=float,
        default=2.0,
        help="Std multiplier for anomaly threshold calibration (default: 2.0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; falling back to CPU.")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    models_dir = Path(args.models_dir).expanduser().resolve()
    labels_csv = Path(args.labels_csv).expanduser().resolve()

    print("\n" + "=" * 60)
    print("  UNSUPERVISED ANOMALY DETECTOR - TRAINING")
    print("  Training exclusively on authentic (Class 0) samples")
    print("=" * 60)

    if not labels_csv.exists():
        print(f"\n[ERROR] labels.csv not found at {labels_csv}")
        print("Run audio_pipeline.py first to generate spectrograms and labels.")
        raise SystemExit(1)

    # ------------------------------------------------------------------ #
    # 1. Load authentic features (Class 0 only; no deepfake labels used)   #
    # ------------------------------------------------------------------ #
    print("\n[1/5] Loading authentic spectrogram features...")
    t0 = perf_counter()
    X_authentic = load_authentic_features(labels_csv)
    print(f"  Loaded in {perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------ #
    # 2. Scale features                                                    #
    # ------------------------------------------------------------------ #
    print("\n[2/5] Fitting StandardScaler on authentic features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_authentic).astype(np.float32)
    input_dim = X_scaled.shape[1]
    print(f"  Input dimension: {input_dim}")

    # ------------------------------------------------------------------ #
    # 3. Train Autoencoder                                                 #
    # ------------------------------------------------------------------ #
    print(f"\n[3/5] Training Autoencoder ({args.epochs} epochs, latent_dim={args.latent_dim})...")
    t0 = perf_counter()
    autoencoder, loss_history = train_autoencoder(
        X_scaled=X_scaled,
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        random_state=GENERAL_CONFIG.random_seed,
    )
    print(f"  Training complete in {perf_counter() - t0:.1f}s")
    print(f"  Final reconstruction loss: {loss_history[-1]:.6f}")

    # ------------------------------------------------------------------ #
    # 4. Train Isolation Forest                                            #
    # ------------------------------------------------------------------ #
    print("\n[4/5] Training Isolation Forest...")
    t0 = perf_counter()
    iso_forest = train_isolation_forest(
        X_scaled=X_scaled,
        random_state=GENERAL_CONFIG.random_seed,
    )
    print(f"  Training complete in {perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------ #
    # 5. Calibrate thresholds and save                                     #
    # ------------------------------------------------------------------ #
    print("\n[5/5] Calibrating anomaly thresholds and saving artifacts...")
    thresholds = calibrate_thresholds(
        autoencoder=autoencoder,
        X_scaled=X_scaled,
        device=device,
        std_multiplier=args.std_multiplier,
    )
    print(f"  Anomaly threshold (AE): {thresholds['anomaly_threshold']:.6f}")

    save_artifacts(
        autoencoder=autoencoder,
        iso_forest=iso_forest,
        scaler=scaler,
        thresholds=thresholds,
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        loss_history=loss_history,
        models_dir=models_dir,
        report_path=unsupervised_artifact_paths(models_dir)["report"],
    )

    print("\n" + "=" * 60)
    print("  UNSUPERVISED TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Autoencoder final loss : {loss_history[-1]:.6f}")
    print(f"  Anomaly threshold      : {thresholds['anomaly_threshold']:.6f}")
    print(f"  Artifacts saved to     : {models_dir}")
    print()


if __name__ == "__main__":
    main()
