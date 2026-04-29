"""PyTorch inference helpers for spectral deepfake detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from config import SETTINGS
from data_pipeline.deep_learning_loaders import DeepModelBatchBuilder, TensorConfig


class SpectralCNN(nn.Module):
    """Compact CNN for mel-spectrogram binary classification."""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def load_torch_spectral_model(
    weights_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Load torch model weights from canonical spectral model path."""
    resolved = Path(weights_path or SETTINGS.paths.supervised_torch_weights_path)
    if not resolved.exists():
        raise FileNotFoundError(f"PyTorch spectral weights not found: {resolved}")

    checkpoint = torch.load(str(resolved), map_location=device)
    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        model = SpectralCNN(in_channels=1)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def infer_audio_probability(
    model: nn.Module,
    audio_path: str | Path,
    tensor_config: TensorConfig | None = None,
    device: str | torch.device = "cpu",
) -> float:
    """Infer AI probability from audio path using shared deep data transforms."""
    builder = DeepModelBatchBuilder(tensor_config=tensor_config)
    signal = builder.load_and_preprocess(str(audio_path))
    input_tensor = builder.build_specrnet_tensor(signal).unsqueeze(0).to(device)
    logits = model(input_tensor)
    probability = torch.sigmoid(logits).flatten()[0].item()
    return float(probability)

