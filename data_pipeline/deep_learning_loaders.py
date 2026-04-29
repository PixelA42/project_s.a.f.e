"""Deep learning data loaders and batch utilities for AASIST/SpecRNet architectures.

Provides PyTorch DataLoader implementations with standardized tensor shapes for
advanced deepfake detection models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa

from data_pipeline.audio_utils import AudioNormalizer, AudioPaddingMethod
from coreML.advanced_spectral_analyzer import (
    AdvancedSpectralConfig,
    AASISTFeatureExtractor,
    SpecRNetFeatureExtractor,
)


@dataclass
class TensorConfig:
    """Configuration for tensor shapes required by deep learning models."""

    # AASIST expects: (batch, 1, time_steps) for waveform input
    # Or (batch, freq_bins, time_frames) for spectrogram input
    aasist_waveform_shape: tuple[int, int, int] = (None, 1, 64000)  # (batch, channels, samples)
    aasist_spectrogram_shape: tuple[int, int, int] = (None, 128, 400)  # (batch, freq, time)

    # SpecRNet expects: (batch, 1, height, width) for spectrogram
    specrnet_spectrogram_shape: tuple[int, int, int, int] = (None, 1, 128, 400)

    # Common audio config
    sample_rate: int = 16000
    n_mels: int = 128
    n_fft: int = 512
    hop_length: int = 160
    short_audio_target_seconds: float = 3.0
    padding_method: AudioPaddingMethod = AudioPaddingMethod.REFLECT_MIRROR

    @property
    def aasist_target_samples(self) -> int:
        return self.aasist_waveform_shape[2]

    @property
    def target_time_frames(self) -> int:
        return self.specrnet_spectrogram_shape[3]


def _fix_time_frames(matrix: np.ndarray, target_time_frames: int) -> np.ndarray:
    """Pad/truncate along time axis to enforce deterministic tensor width."""
    return librosa.util.fix_length(matrix, size=target_time_frames, axis=-1).astype(np.float32)


class AudioDataset(Dataset):
    """PyTorch Dataset for audio files with standardized preprocessing."""

    def __init__(
        self,
        audio_paths: list[str],
        labels: list[int] | None = None,
        sample_rate: int = 16000,
        target_samples: int = 64000,
        transform: Callable | None = None,
        return_path: bool = False,
    ):
        """
        Initialize AudioDataset.

        Args:
            audio_paths: List of paths to audio files.
            labels: Optional list of labels (0=human, 1=ai).
            sample_rate: Sample rate for loading (default 16000 Hz).
            target_samples: Target number of samples (default 64000 = 4 seconds at 16kHz).
            transform: Optional callable to transform audio signals.
            return_path: Whether to return file path with sample (default False).
        """
        self.audio_paths = audio_paths
        self.labels = labels or [None] * len(audio_paths)
        self.sample_rate = sample_rate
        self.target_samples = target_samples
        self.transform = transform
        self.return_path = return_path
        self.tensor_config = TensorConfig(
            sample_rate=sample_rate,
            aasist_waveform_shape=(None, 1, target_samples),
        )
        self.audio_normalizer = AudioNormalizer(
            target_duration_seconds=target_samples / sample_rate,
            sample_rate=sample_rate,
            padding_method=self.tensor_config.padding_method,
        )

        if len(self.audio_paths) != len(self.labels):
            raise ValueError("Number of audio_paths and labels must match")

    def __len__(self) -> int:
        return len(self.audio_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int] | tuple[torch.Tensor, int, str]:
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]

        try:
            audio_signal, _ = librosa.load(
                audio_path,
                sr=self.sample_rate,
                mono=True,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load audio: {audio_path}") from exc

        audio_signal = self.audio_normalizer.normalize_signal(audio_signal)
        audio_signal = self.audio_normalizer.pad_short_audio(audio_signal)
        audio_signal = audio_signal[: self.target_samples]

        # Apply transforms if provided
        if self.transform:
            audio_signal = self.transform(audio_signal)

        audio_tensor = torch.from_numpy(audio_signal).float()
        # Add channel dimension: (samples,) -> (1, samples)
        audio_tensor = audio_tensor.unsqueeze(0)

        if self.return_path:
            return audio_tensor, label, audio_path
        return audio_tensor, label


class SpectrogramDataset(Dataset):
    """PyTorch Dataset for spectrogram-based inputs (MFCC, Mel, etc.)."""

    def __init__(
        self,
        audio_paths: list[str],
        labels: list[int] | None = None,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 512,
        hop_length: int = 160,
        transform: Callable | None = None,
        return_path: bool = False,
    ):
        """
        Initialize SpectrogramDataset.

        Args:
            audio_paths: List of paths to audio files.
            labels: Optional list of labels (0=human, 1=ai).
            sample_rate: Sample rate for loading (default 16000 Hz).
            n_mels: Number of mel bins (default 128).
            n_fft: FFT size (default 512).
            hop_length: Hop length for STFT (default 160).
            transform: Optional callable to transform spectrograms.
            return_path: Whether to return file path with sample (default False).
        """
        self.audio_paths = audio_paths
        self.labels = labels or [None] * len(audio_paths)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.transform = transform
        self.return_path = return_path
        self.tensor_config = TensorConfig(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        self.audio_normalizer = AudioNormalizer(
            target_duration_seconds=self.tensor_config.short_audio_target_seconds,
            sample_rate=sample_rate,
            padding_method=self.tensor_config.padding_method,
        )
        spec_config = AdvancedSpectralConfig(
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        self.aasist_extractor = AASISTFeatureExtractor(spec_config)
        self.specrnet_extractor = SpecRNetFeatureExtractor(spec_config)

        if len(self.audio_paths) != len(self.labels):
            raise ValueError("Number of audio_paths and labels must match")

    def __len__(self) -> int:
        return len(self.audio_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int] | tuple[torch.Tensor, int, str]:
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]

        try:
            audio_signal, _ = librosa.load(
                audio_path,
                sr=self.sample_rate,
                mono=True,
            )
            audio_signal = self.audio_normalizer.normalize_signal(audio_signal)
            audio_signal = self.audio_normalizer.pad_short_audio(audio_signal)
            mel_db = self.specrnet_extractor.extract_mel_spectrogram_2d(
                audio_signal,
                self.sample_rate,
            )
            mel_db = _fix_time_frames(mel_db, self.tensor_config.target_time_frames)

        except Exception as exc:
            raise RuntimeError(f"Failed to load audio or compute spectrogram: {audio_path}") from exc

        # Apply transforms if provided
        if self.transform:
            mel_db = self.transform(mel_db)

        spec_tensor = torch.from_numpy(mel_db).float()
        # Add channel dimension: (freq, time) -> (1, freq, time)
        spec_tensor = spec_tensor.unsqueeze(0)

        if self.return_path:
            return spec_tensor, label, audio_path
        return spec_tensor, label


class DataLoaderFactory:
    """Factory for creating standardized DataLoaders."""

    @staticmethod
    def create_waveform_loader(
        audio_paths: list[str],
        labels: list[int] | None = None,
        batch_size: int = 32,
        sample_rate: int = 16000,
        target_samples: int = 64000,
        num_workers: int = 0,
        shuffle: bool = True,
        pin_memory: bool = True,
    ) -> DataLoader:
        """
        Create a DataLoader for waveform inputs (AASIST style).

        Args:
            audio_paths: List of audio file paths.
            labels: Optional list of labels.
            batch_size: Batch size (default 32).
            sample_rate: Sample rate in Hz (default 16000).
            target_samples: Target number of samples (default 64000).
            num_workers: Number of data loading workers (default 0).
            shuffle: Whether to shuffle data (default True).
            pin_memory: Whether to pin memory (default True).

        Returns:
            PyTorch DataLoader for waveform data.
        """
        dataset = AudioDataset(
            audio_paths=audio_paths,
            labels=labels,
            sample_rate=sample_rate,
            target_samples=target_samples,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )


class DeepModelBatchBuilder:
    """Create model-ready tensors for AASIST and SpecRNet training/inference."""

    def __init__(self, tensor_config: TensorConfig | None = None):
        self.tensor_config = tensor_config or TensorConfig()
        self.audio_normalizer = AudioNormalizer(
            target_duration_seconds=self.tensor_config.aasist_target_samples
            / self.tensor_config.sample_rate,
            sample_rate=self.tensor_config.sample_rate,
            padding_method=self.tensor_config.padding_method,
        )
        spectral_config = AdvancedSpectralConfig(
            n_mels=self.tensor_config.n_mels,
            n_fft=self.tensor_config.n_fft,
            hop_length=self.tensor_config.hop_length,
        )
        self.aasist_extractor = AASISTFeatureExtractor(spectral_config)
        self.specrnet_extractor = SpecRNetFeatureExtractor(spectral_config)

    def load_and_preprocess(self, audio_path: str | Path) -> np.ndarray:
        audio_signal, _ = librosa.load(
            str(audio_path),
            sr=self.tensor_config.sample_rate,
            mono=True,
        )
        audio_signal = self.audio_normalizer.normalize_signal(audio_signal)
        audio_signal = self.audio_normalizer.pad_short_audio(audio_signal)
        return audio_signal[: self.tensor_config.aasist_target_samples]

    def build_aasist_waveform_tensor(self, audio_signal: np.ndarray) -> torch.Tensor:
        fixed = librosa.util.fix_length(
            audio_signal,
            size=self.tensor_config.aasist_target_samples,
            axis=0,
        ).astype(np.float32)
        return torch.from_numpy(fixed).unsqueeze(0)

    def build_aasist_spectral_tensor(self, audio_signal: np.ndarray) -> torch.Tensor:
        mel = self.aasist_extractor.extract_mel_spectrogram(
            audio_signal,
            self.tensor_config.sample_rate,
        )
        mel = _fix_time_frames(mel, self.tensor_config.target_time_frames)
        return torch.from_numpy(mel.astype(np.float32))

    def build_specrnet_tensor(self, audio_signal: np.ndarray) -> torch.Tensor:
        mel = self.specrnet_extractor.extract_mel_spectrogram_2d(
            audio_signal,
            self.tensor_config.sample_rate,
        )
        mel = _fix_time_frames(mel, self.tensor_config.target_time_frames)
        return torch.from_numpy(mel.astype(np.float32)).unsqueeze(0)

    @staticmethod
    def create_spectrogram_loader(
        audio_paths: list[str],
        labels: list[int] | None = None,
        batch_size: int = 32,
        sample_rate: int = 16000,
        n_mels: int = 128,
        num_workers: int = 0,
        shuffle: bool = True,
        pin_memory: bool = True,
    ) -> DataLoader:
        """
        Create a DataLoader for spectrogram inputs (SpecRNet style).

        Args:
            audio_paths: List of audio file paths.
            labels: Optional list of labels.
            batch_size: Batch size (default 32).
            sample_rate: Sample rate in Hz (default 16000).
            n_mels: Number of mel bins (default 128).
            num_workers: Number of data loading workers (default 0).
            shuffle: Whether to shuffle data (default True).
            pin_memory: Whether to pin memory (default True).

        Returns:
            PyTorch DataLoader for spectrogram data.
        """
        dataset = SpectrogramDataset(
            audio_paths=audio_paths,
            labels=labels,
            sample_rate=sample_rate,
            n_mels=n_mels,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
