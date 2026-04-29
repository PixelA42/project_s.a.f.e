"""Audio preprocessing utilities for short-audio handling and normalization.

Implements standardized padding and signal-mirroring techniques for audio samples
under 3 seconds to ensure the feature extractor has sufficient temporal data.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

import numpy as np
import librosa


class AudioPaddingMethod(StrEnum):
    """Audio padding strategies for short-audio handling."""

    ZERO_PAD = "zero_pad"  # Pad with zeros
    REFLECT_MIRROR = "reflect_mirror"  # Mirror the signal (numpy reflect mode)
    REPEAT = "repeat"  # Repeat the entire signal
    EDGE_PAD = "edge_pad"  # Pad with edge value


class AudioNormalizer:
    """Normalize and standardize audio signals for consistent feature extraction."""

    def __init__(
        self,
        target_duration_seconds: float = 3.0,
        sample_rate: int = 16000,
        padding_method: AudioPaddingMethod = AudioPaddingMethod.REFLECT_MIRROR,
    ):
        """
        Initialize the AudioNormalizer.

        Args:
            target_duration_seconds: Target duration for all audio clips (default 3.0 seconds).
            sample_rate: Sample rate for librosa operations (default 16000 Hz).
            padding_method: Method to use for padding short audio (default reflect_mirror).
        """
        self.target_duration_seconds = target_duration_seconds
        self.sample_rate = sample_rate
        self.target_samples = int(target_duration_seconds * sample_rate)
        self.padding_method = AudioPaddingMethod(padding_method)

    def normalize_signal(
        self,
        audio_signal: np.ndarray,
        normalize_amplitude: bool = True,
    ) -> np.ndarray:
        """
        Normalize audio signal amplitude to [-1, 1] range.

        Args:
            audio_signal: Input audio signal.
            normalize_amplitude: Whether to normalize to [-1, 1] (default True).

        Returns:
            Normalized audio signal.
        """
        if audio_signal.size == 0:
            return audio_signal

        if normalize_amplitude:
            max_abs = np.abs(audio_signal).max()
            if max_abs > 0:
                audio_signal = audio_signal / max_abs

        return audio_signal.astype(np.float32)

    def pad_short_audio(
        self,
        audio_signal: np.ndarray,
        method: AudioPaddingMethod | None = None,
    ) -> np.ndarray:
        """
        Pad short audio signals to target duration using specified method.

        For audio under target_duration, this ensures stable feature extraction
        by providing sufficient temporal context.

        Args:
            audio_signal: Input audio signal.
            method: Padding method (uses instance default if None).

        Returns:
            Padded audio signal with length == target_samples.
        """
        if method is None:
            method = self.padding_method

        current_samples = len(audio_signal)

        if current_samples == 0:
            return np.zeros(self.target_samples, dtype=np.float32)

        # If audio is already longer than target, return as-is
        if current_samples >= self.target_samples:
            return audio_signal.astype(np.float32)

        pad_amount = self.target_samples - current_samples

        if method == AudioPaddingMethod.ZERO_PAD:
            padded = np.pad(audio_signal, (0, pad_amount), mode="constant", constant_values=0)
        elif method == AudioPaddingMethod.REFLECT_MIRROR:
            # np.pad(..., mode="reflect") requires at least 2 samples; ultra-short clips
            # fall back to edge padding to keep preprocessing stable.
            if current_samples < 2:
                padded = np.pad(audio_signal, (0, pad_amount), mode="edge")
            else:
                padded = np.pad(audio_signal, (0, pad_amount), mode="reflect")
        elif method == AudioPaddingMethod.REPEAT:
            # Repeat the signal until we have enough samples
            repeats_needed = (pad_amount // current_samples) + 1
            repeated = np.tile(audio_signal, repeats_needed)
            padded = repeated[:self.target_samples]
        elif method == AudioPaddingMethod.EDGE_PAD:
            padded = np.pad(audio_signal, (0, pad_amount), mode="edge")
        else:
            raise ValueError(f"Unknown padding method: {method}")

        return padded[:self.target_samples].astype(np.float32)

    def preprocess(
        self,
        audio_file_path: str,
        sr: int | None = None,
        mono: bool = True,
        normalize: bool = True,
        pad: bool = True,
        padding_method: AudioPaddingMethod | None = None,
    ) -> np.ndarray:
        """
        Load, normalize, and pad audio in a single operation.

        This is the primary entry point for audio preprocessing.

        Args:
            audio_file_path: Path to audio file.
            sr: Sample rate for loading (uses instance sr if None).
            mono: Convert to mono (default True).
            normalize: Normalize amplitude (default True).
            pad: Pad to target duration (default True).
            padding_method: Padding method (uses instance default if None).

        Returns:
            Preprocessed audio signal as float32 array.

        Raises:
            ValueError: If audio file cannot be loaded.
        """
        try:
            audio_signal, _ = librosa.load(
                audio_file_path,
                sr=sr or self.sample_rate,
                mono=mono,
            )
        except Exception as exc:
            raise ValueError(f"Failed to load audio file: {audio_file_path}") from exc

        if normalize:
            audio_signal = self.normalize_signal(audio_signal)

        if pad:
            audio_signal = self.pad_short_audio(audio_signal, method=padding_method)

        return audio_signal

    def get_duration_info(self, audio_signal: np.ndarray) -> dict[str, float]:
        """
        Get duration information for an audio signal.

        Args:
            audio_signal: Audio signal array.

        Returns:
            Dictionary with duration_seconds and is_short (< target_duration).
        """
        duration = len(audio_signal) / self.sample_rate
        return {
            "duration_seconds": float(duration),
            "is_short": duration < self.target_duration_seconds,
            "target_duration_seconds": float(self.target_duration_seconds),
            "needs_padding": duration < self.target_duration_seconds,
        }


class AudioValidator:
    """Validate audio signals for quality and suitability."""

    def __init__(
        self,
        min_duration_seconds: float = 0.1,
        max_duration_seconds: float = 120.0,
    ):
        """
        Initialize the AudioValidator.

        Args:
            min_duration_seconds: Minimum acceptable duration (default 0.1 seconds).
            max_duration_seconds: Maximum acceptable duration (default 120 seconds).
        """
        self.min_duration_seconds = min_duration_seconds
        self.max_duration_seconds = max_duration_seconds

    def validate(
        self,
        audio_signal: np.ndarray,
        sample_rate: int,
    ) -> dict[str, bool | str]:
        """
        Validate audio signal quality.

        Args:
            audio_signal: Audio signal to validate.
            sample_rate: Sample rate of the audio.

        Returns:
            Dictionary with validation results.
        """
        results = {
            "is_valid": True,
            "issues": [],
        }

        # Check if empty
        if audio_signal.size == 0:
            results["is_valid"] = False
            results["issues"].append("Audio signal is empty")

        # Check duration
        duration = len(audio_signal) / sample_rate
        if duration < self.min_duration_seconds:
            results["is_valid"] = False
            results["issues"].append(
                f"Duration {duration:.2f}s is below minimum {self.min_duration_seconds}s"
            )

        if duration > self.max_duration_seconds:
            results["is_valid"] = False
            results["issues"].append(
                f"Duration {duration:.2f}s exceeds maximum {self.max_duration_seconds}s"
            )

        # Check for NaN or Inf
        if np.isnan(audio_signal).any():
            results["is_valid"] = False
            results["issues"].append("Audio contains NaN values")

        if np.isinf(audio_signal).any():
            results["is_valid"] = False
            results["issues"].append("Audio contains infinite values")

        return results
