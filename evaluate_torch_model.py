"""Evaluate PyTorch spectral model and overwrite canonical metrics report."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import SETTINGS
from coreML.torch_evaluation import evaluate_spectral_torch_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate spectral PyTorch model")
    parser.add_argument(
        "--manifest",
        required=True,
        help="CSV manifest with columns: audio_path,label",
    )
    parser.add_argument(
        "--weights",
        default=str(SETTINGS.paths.supervised_torch_weights_path),
        help="Path to .pt/.pth model weights",
    )
    parser.add_argument(
        "--report",
        default=str(SETTINGS.paths.supervised_torch_eval_report_path),
        help="Canonical JSON report path used by reporting UI",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest)
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    frame = pd.read_csv(manifest)
    required = {"audio_path", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    report = evaluate_spectral_torch_model(
        audio_paths=frame["audio_path"].astype(str).tolist(),
        labels=frame["label"].astype(int).tolist(),
        report_path=args.report,
        weights_path=args.weights,
    )
    print("Wrote canonical report to:", args.report)
    print("Removed legacy artifacts:", report.get("deprecated_artifacts_removed", []))


if __name__ == "__main__":
    main()

