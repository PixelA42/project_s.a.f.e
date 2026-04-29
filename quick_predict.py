"""Hybrid inference pipeline for Project S.A.F.E.

Combines a supervised PyTorch spectral classifier with an unsupervised
anomaly detector (Autoencoder + Isolation Forest) to produce a two-pronged
prediction for every audio file.

Decision logic
--------------
1. Run the supervised model  →  positive_probability  (0–1)
2. Run the unsupervised model →  anomaly scores (reconstruction error,
                                  isolation score, anomaly_flag)
3. Smart uncertainty routing:
   - If supervised probability is OUTSIDE [0.35, 0.65]  →  trust the
     supervised label directly.
   - If supervised probability is INSIDE  [0.35, 0.65]  →  the model is
     uncertain; use the unsupervised anomaly score to decide:
       * anomaly_flag=True  →  flag for manual review (ESCALATE)
       * anomaly_flag=False and p < threshold → lean toward HUMAN
       * anomaly_flag=False and p >= threshold → fail closed as AI

Usage
-----
    python quick_predict.py --input path/to/audio.wav
    python quick_predict.py --input path/to/audio.wav --keep-temp
    python quick_predict.py --input path/to/audio.wav --decision-threshold 0.45
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import torch

from config import SETTINGS
from coreML.torch_inference import infer_audio_probability, load_torch_spectral_model
from coreML.uncertainty_queue import UncertaintyQueue
from train_unsupervised import UnsupervisedAnomalyDetector


PATHS = SETTINGS.paths

# Uncertainty zone boundaries (supervised model)
UNCERTAINTY_LOWER: float = 0.35
UNCERTAINTY_UPPER: float = 0.65


# ---------------------------------------------------------------------------
# Spectrogram helper (reuses audio_pipeline logic without side-effects)
# ---------------------------------------------------------------------------

def _audio_to_temp_spectrogram(audio_path: Path) -> Path:
    """Convert an audio file to a temporary spectrogram PNG for the unsupervised model.

    Returns the path to the generated PNG.  The caller is responsible for
    deleting it when done.
    """
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import numpy as np
    from config import SETTINGS as _S

    _AUDIO = _S.audio
    signal, sr = librosa.load(str(audio_path), sr=_AUDIO.sample_rate, mono=True,
                               duration=_AUDIO.clip_duration_seconds)
    mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=_AUDIO.n_mels,
                                          fmax=_AUDIO.mel_fmax)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    fig = plt.figure(figsize=(_AUDIO.figure_size_inches, _AUDIO.figure_size_inches),
                     dpi=_AUDIO.spectrogram_dpi)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.axis("off")
    librosa.display.specshow(mel_db, sr=sr, cmap=_AUDIO.colormap, ax=ax)
    plt.savefig(tmp.name, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Hybrid inference
# ---------------------------------------------------------------------------

def run_hybrid_inference(
    input_audio: Path,
    supervised_model: torch.nn.Module,
    unsupervised_detector: UnsupervisedAnomalyDetector,
    decision_threshold: float,
    uncertainty_queue: UncertaintyQueue,
    keep_temp: bool = False,
) -> dict:
    """Run the full hybrid supervised + unsupervised inference pipeline.

    Parameters
    ----------
    input_audio:
        Path to the audio file to analyse.
    supervised_model:
        Loaded PyTorch SpectralCNN model.
    unsupervised_detector:
        Loaded UnsupervisedAnomalyDetector instance.
    decision_threshold:
        Probability cut-off for the supervised model (default 0.45).
    uncertainty_queue:
        UncertaintyQueue instance for routing uncertain samples.
    keep_temp:
        If True, the temporary spectrogram PNG is not deleted.

    Returns
    -------
    dict with all scores, flags, and routing decision.
    """
    # ------------------------------------------------------------------ #
    # Step 1 — Supervised classification                                   #
    # ------------------------------------------------------------------ #
    positive_probability = infer_audio_probability(
        supervised_model, str(input_audio), device="cpu"
    )
    supervised_prediction = 1 if positive_probability >= decision_threshold else 0
    supervised_label = "AI" if supervised_prediction == 1 else "HUMAN"
    is_uncertain = UNCERTAINTY_LOWER <= positive_probability <= UNCERTAINTY_UPPER

    # ------------------------------------------------------------------ #
    # Step 2 — Unsupervised anomaly scoring                                #
    # ------------------------------------------------------------------ #
    temp_spec_path: Path | None = None
    unsupervised_scores: dict = {
        "reconstruction_error": 0.0,
        "isolation_score": 0.0,
        "anomaly_flag": False,
        "ae_anomaly": False,
        "if_anomaly": False,
        "unsupervised_ready": False,
    }

    if unsupervised_detector.is_ready:
        try:
            temp_spec_path = _audio_to_temp_spectrogram(input_audio)
            unsupervised_scores = unsupervised_detector.score(str(temp_spec_path))
        except Exception as exc:
            print(f"  [WARN] Unsupervised scoring failed: {exc}")
        finally:
            if temp_spec_path and not keep_temp:
                try:
                    temp_spec_path.unlink(missing_ok=True)
                except Exception:
                    pass

    # ------------------------------------------------------------------ #
    # Step 3 — Smart uncertainty routing                                   #
    # ------------------------------------------------------------------ #
    routing_decision: str
    routing_reason: str
    queue_item = None

    if not is_uncertain:
        # Supervised model is confident — use its label directly
        routing_decision = supervised_label
        routing_reason = (
            f"Supervised model confident (p={positive_probability:.3f}, "
            f"threshold={decision_threshold})"
        )
    else:
        # Supervised model is uncertain — defer to unsupervised anomaly signal
        anomaly_flag = unsupervised_scores.get("anomaly_flag", False)
        unsupervised_ready = unsupervised_scores.get("unsupervised_ready", False)

        if not unsupervised_ready:
            # Unsupervised model not trained yet — queue for manual review
            routing_decision = "UNCERTAIN"
            routing_reason = (
                f"Supervised uncertain (p={positive_probability:.3f}) and "
                "unsupervised model not available — queued for manual review"
            )
        elif anomaly_flag:
            # Unsupervised model detects anomaly → likely AI, flag for review
            routing_decision = "UNCERTAIN_ANOMALY"
            routing_reason = (
                f"Supervised uncertain (p={positive_probability:.3f}); "
                f"unsupervised anomaly detected "
                f"(recon_err={unsupervised_scores['reconstruction_error']:.4f}, "
                f"iso_score={unsupervised_scores['isolation_score']:.4f}) "
                "— flagged for manual review"
            )
        elif supervised_prediction == 1:
            # Fraud-safe policy: do not override a threshold-positive AI call to HUMAN.
            routing_decision = "AI"
            routing_reason = (
                f"Supervised uncertain but above fraud threshold "
                f"(p={positive_probability:.3f}, threshold={decision_threshold}); "
                "unsupervised model sees no anomaly, but the system fails closed as AI"
            )
        else:
            # Unsupervised model sees no anomaly → lean toward authentic
            routing_decision = "HUMAN"
            routing_reason = (
                f"Supervised uncertain (p={positive_probability:.3f}); "
                "unsupervised model sees no anomaly — classified as HUMAN"
            )

        if (not unsupervised_ready) or anomaly_flag:
            queue_item = uncertainty_queue.add_to_queue(
                audio_file_path=str(input_audio),
                predicted_probability=positive_probability,
                predicted_label=supervised_prediction,
                confidence_score=max(positive_probability, 1.0 - positive_probability),
                model_name="hybrid_spectral_model",
                tags=[
                    "uncertainty_zone",
                    "auto_routed",
                    "anomaly_detected" if anomaly_flag else "no_anomaly",
                    "unsupervised_used" if unsupervised_ready else "unsupervised_unavailable",
                ],
            )

            # Copy audio to review directory for human inspection.
            review_dir = PATHS.outputs_dir / "uncertainty_review"
            review_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(input_audio), review_dir / input_audio.name)

    return {
        # Supervised outputs
        "supervised_probability": positive_probability,
        "supervised_prediction": supervised_prediction,
        "supervised_label": supervised_label,
        "decision_threshold": decision_threshold,
        "is_uncertain": is_uncertain,
        # Unsupervised outputs
        **{f"unsupervised_{k}": v for k, v in unsupervised_scores.items()},
        # Routing
        "routing_decision": routing_decision,
        "routing_reason": routing_reason,
        "queue_item_id": queue_item.item_id if queue_item else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid HUMAN/AI prediction from one audio file"
    )
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
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=SETTINGS.training.decision_threshold,
        help=(
            "Probability threshold for the supervised positive class; "
            "lower values increase recall at the cost of more false positives"
        ),
    )
    return parser.parse_args()


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

    # Load supervised model
    supervised_model = load_torch_spectral_model(
        PATHS.supervised_torch_weights_path, device="cpu"
    )

    # Load unsupervised detector (gracefully handles missing artifacts)
    unsupervised_detector = UnsupervisedAnomalyDetector(device="cpu")
    if not unsupervised_detector.is_ready:
        print(
            "[INFO] Unsupervised model artifacts not found. "
            "Run `python train_unsupervised.py` to train them.\n"
            "       Uncertain samples will be queued for manual review without "
            "unsupervised tiebreaking."
        )

    uncertainty_queue = UncertaintyQueue(
        queue_dir=PATHS.outputs_dir / "uncertainty_queue",
        lower_bound=UNCERTAINTY_LOWER,
        upper_bound=UNCERTAINTY_UPPER,
    )

    result = run_hybrid_inference(
        input_audio=input_audio,
        supervised_model=supervised_model,
        unsupervised_detector=unsupervised_detector,
        decision_threshold=args.decision_threshold,
        uncertainty_queue=uncertainty_queue,
        keep_temp=args.keep_temp,
    )

    # ------------------------------------------------------------------ #
    # Print results                                                        #
    # ------------------------------------------------------------------ #
    print()
    print("=" * 60)
    print("  HYBRID INFERENCE RESULT")
    print("=" * 60)
    print(f"  Input                  : {input_audio}")
    print()
    print("  [SUPERVISED]")
    print(f"  Prediction             : {result['supervised_prediction']} ({result['supervised_label']})")
    print(f"  Positive probability   : {result['supervised_probability']:.4f}")
    print(f"  Decision threshold     : {result['decision_threshold']}")
    print(f"  Uncertain zone         : {result['is_uncertain']}")
    print()
    print("  [UNSUPERVISED]")
    print(f"  Model ready            : {result['unsupervised_unsupervised_ready']}")
    print(f"  Reconstruction error   : {result['unsupervised_reconstruction_error']:.6f}")
    print(f"  Isolation score        : {result['unsupervised_isolation_score']:.4f}")
    print(f"  AE anomaly             : {result['unsupervised_ae_anomaly']}")
    print(f"  IF anomaly             : {result['unsupervised_if_anomaly']}")
    print(f"  Anomaly flag           : {result['unsupervised_anomaly_flag']}")
    print()
    print("  [ROUTING DECISION]")
    print(f"  Decision               : {result['routing_decision']}")
    print(f"  Reason                 : {result['routing_reason']}")
    if result["queue_item_id"]:
        print(f"  Queue item ID          : {result['queue_item_id']}")
    print("=" * 60)
    print()
    print("  Model weights (supervised)  :", PATHS.supervised_torch_weights_path)
    print()


if __name__ == "__main__":
    main()
