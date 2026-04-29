"""Advanced spectral feature extraction for AASIST and SpecRNet architectures.

Provides high-resolution feature extraction optimized for deep learning models,
including STFT, constant-Q, and other advanced spectral representations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import librosa
import librosa.feature


@dataclass
class AdvancedSpectralConfig:
    """Configuration for advanced spectral feature extraction."""

    # Mel Spectrogram parameters
    n_mels: int = 128
    n_fft: int = 512
    hop_length: int = 160
    f_min: float = 50.0
    f_max: float = 8000.0

    # MFCC parameters
    n_mfcc: int = 40

    # Constant-Q parameters
    n_bins: int = 84  # 7 octaves at 12 bins per octave
    bins_per_octave: int = 12

    # Temporal/spectral derivative orders
    delta_order: int = 1  # First and second derivatives

    # Output normalization
    normalize: bool = True


class AASISTFeatureExtractor:
    """Feature extractor optimized for AASIST architecture.
    
    AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal)
    expects high-resolution spectral representations with both
    instantaneous and temporal derivative information.
    """

    def __init__(self, config: AdvancedSpectralConfig | None = None):
        """
        Initialize AASIST feature extractor.

        Args:
            config: Feature extraction configuration (uses defaults if None).
        """
        self.config = config or AdvancedSpectralConfig()

    def extract_mel_spectrogram(
        self,
        audio_signal: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Extract mel spectrogram (128, time_frames).

        Args:
            audio_signal: Audio waveform.
            sample_rate: Sample rate in Hz.

        Returns:
            Mel spectrogram of shape (n_mels, time_frames).
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio_signal,
            sr=sample_rate,
            n_mels=self.config.n_mels,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
        )
        # Convert to dB scale
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_db.astype(np.float32)

    def extract_mfcc(
        self,
        audio_signal: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Extract MFCC coefficients (40, time_frames).

        Args:
            audio_signal: Audio waveform.
            sample_rate: Sample rate in Hz.

        Returns:
            MFCC features of shape (n_mfcc, time_frames).
        """
        mfcc = librosa.feature.mfcc(
            y=audio_signal,
            sr=sample_rate,
            n_mfcc=self.config.n_mfcc,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
        )
        return mfcc.astype(np.float32)

    def extract_stft_magnitude(
        self,
        audio_signal: np.ndarray,
    ) -> np.ndarray:
        """
        Extract STFT magnitude.

        Args:
            audio_signal: Audio waveform.

        Returns:
            STFT magnitude of shape (freq_bins, time_frames).
        """
        stft = librosa.stft(
            audio_signal,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
        )
        magnitude = np.abs(stft)
        # Convert to dB scale
        magnitude_db = librosa.power_to_db(magnitude**2, ref=np.max)
        return magnitude_db.astype(np.float32)

    def extract_stft_phase(
        self,
        audio_signal: np.ndarray,
    ) -> np.ndarray:
        """
        Extract STFT phase.

        Args:
            audio_signal: Audio waveform.

        Returns:
            STFT phase of shape (freq_bins, time_frames).
        """
        stft = librosa.stft(
            audio_signal,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
        )
        phase = np.angle(stft)
        return phase.astype(np.float32)

    def extract_delta_features(
        self,
        feature_matrix: np.ndarray,
        order: int = 1,
    ) -> np.ndarray:
        """
        Extract delta (temporal derivative) features.

        Args:
            feature_matrix: Base feature matrix of shape (n_features, time_frames).
            order: Derivative order (1=delta, 2=delta-delta).

        Returns:
            Delta features of same shape as input.
        """
        for _ in range(order):
            feature_matrix = librosa.feature.delta(feature_matrix)
        return feature_matrix.astype(np.float32)

    def extract_composite(
        self,
        audio_signal: np.ndarray,
        sample_rate: int,
        include_deltas: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Extract comprehensive feature set for AASIST.

        Includes mel spectrogram, MFCC, and optionally their delta features.

        Args:
            audio_signal: Audio waveform.
            sample_rate: Sample rate in Hz.
            include_deltas: Whether to include delta features (default True).

        Returns:
            Dictionary with keys: 'mel_spec', 'mfcc', optionally 'mel_delta', 'mfcc_delta'.
        """
        features = {}

        # Base features
        features["mel_spec"] = self.extract_mel_spectrogram(audio_signal, sample_rate)
        features["mfcc"] = self.extract_mfcc(audio_signal, sample_rate)

        # Delta features
        if include_deltas:
            features["mel_delta"] = self.extract_delta_features(
                features["mel_spec"],
                order=self.config.delta_order,
            )
            features["mfcc_delta"] = self.extract_delta_features(
                features["mfcc"],
                order=self.config.delta_order,
            )

        # Normalize if configured
        if self.config.normalize:
            features = {
                key: self._normalize_features(value) for key, value in features.items()
            }

        return features

    @staticmethod
    def _normalize_features(features: np.ndarray) -> np.ndarray:
        """Normalize features to zero mean and unit variance."""
        mean = features.mean(axis=1, keepdims=True)
        std = features.std(axis=1, keepdims=True)
        std[std == 0] = 1  # Avoid division by zero
        return ((features - mean) / std).astype(np.float32)

    def stack_features_for_model(
        self,
        features_dict: dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Stack extracted features into a single tensor for AASIST input.

        Expected output shape: (channels, freq, time) where channels is the
        number of stacked feature maps.

        Args:
            features_dict: Dictionary of feature matrices from extract_composite().

        Returns:
            Stacked feature tensor of shape (num_features, freq, time).
        """
        # Stack vertically: (total_features, time_frames)
        stacked = np.vstack(list(features_dict.values()))
        return stacked.astype(np.float32)


class SpecRNetFeatureExtractor:
    """Feature extractor optimized for SpecRNet architecture.
    
    SpecRNet expects high-resolution spectrograms with spatial structure
    suitable for CNN-based processing.
    """

    def __init__(self, config: AdvancedSpectralConfig | None = None):
        """
        Initialize SpecRNet feature extractor.

        Args:
            config: Feature extraction configuration (uses defaults if None).
        """
        self.config = config or AdvancedSpectralConfig()

    def extract_mel_spectrogram_2d(
        self,
        audio_signal: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Extract mel spectrogram as 2D image: (mel_bins, time_frames).

        Args:
            audio_signal: Audio waveform.
            sample_rate: Sample rate in Hz.

        Returns:
            Mel spectrogram of shape (n_mels, time_frames).
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio_signal,
            sr=sample_rate,
            n_mels=self.config.n_mels,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize to [0, 1]
        mel_db = np.clip(mel_db, -80, 0)  # Clip to reasonable dB range
        mel_db = (mel_db + 80) / 80  # Normalize to [0, 1]

        return mel_db.astype(np.float32)

    def extract_stft_2d(
        self,
        audio_signal: np.ndarray,
    ) -> np.ndarray:
        """
        Extract STFT magnitude as 2D image.

        Args:
            audio_signal: Audio waveform.

        Returns:
            STFT magnitude of shape (freq_bins, time_frames).
        """
        stft = librosa.stft(
            audio_signal,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
        )
        magnitude = np.abs(stft)
        magnitude_db = librosa.power_to_db(magnitude**2, ref=np.max)

        # Normalize to [0, 1]
        magnitude_db = np.clip(magnitude_db, -80, 0)
        magnitude_db = (magnitude_db + 80) / 80

        return magnitude_db.astype(np.float32)

    def extract_multichannel_spectrogram(
        self,
        audio_signal: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Extract multi-channel spectrogram representation.

        Returns stacked channels of different spectral representations:
        - Channel 0: Mel spectrogram
        - Channel 1: STFT magnitude
        - Channel 2: Mel delta (temporal derivative)

        Args:
            audio_signal: Audio waveform.
            sample_rate: Sample rate in Hz.

        Returns:
            Multi-channel spectrogram of shape (channels, mel_bins, time_frames).
        """
        mel_spec = self.extract_mel_spectrogram_2d(audio_signal, sample_rate)
        stft_spec = self.extract_stft_2d(audio_signal)

        # Ensure same shape (in case STFT has different freq bins)
        # Resample STFT to match mel bins if needed
        if stft_spec.shape[0] != mel_spec.shape[0]:
            # Use librosa interpolation or simple padding
            stft_spec = librosa.util.fix_length(
                stft_spec,
                size=mel_spec.shape[0],
                axis=0,
            )

        # Extract delta features from mel
        mel_delta = librosa.feature.delta(mel_spec)
        mel_delta = np.clip((mel_delta + 1) / 2, 0, 1)  # Normalize to [0, 1]

        # Stack channels
        channels = np.stack([mel_spec, stft_spec[:mel_spec.shape[0]], mel_delta])
        return channels.astype(np.float32)

    def extract_for_cnn(
        self,
        audio_signal: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Extract spectrogram optimized for CNN input.

        Output shape is (1, height, width) suitable for 2D CNN models.

        Args:
            audio_signal: Audio waveform.
            sample_rate: Sample rate in Hz.

        Returns:
            Single-channel spectrogram of shape (1, n_mels, time_frames).
        """
        mel_spec = self.extract_mel_spectrogram_2d(audio_signal, sample_rate)
        # Add channel dimension
        mel_spec = mel_spec[np.newaxis, ...]
        return mel_spec.astype(np.float32)
