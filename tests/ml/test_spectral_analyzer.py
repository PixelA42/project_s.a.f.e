import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis import HealthCheck
import joblib
import json
import wave

from coreML.errors import AudioProcessingError
from coreML.spectral_analyzer import SpectralAnalyzer


def _write_wav(path: str, signal: np.ndarray, sample_rate: int = 16000) -> None:
    clipped = np.clip(signal, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    duration=st.floats(min_value=0.2, max_value=1.0, allow_nan=False, allow_infinity=False),
    sample_rate=st.sampled_from([8000, 16000, 22050]),
)
def test_mfcc_output_shape_and_dtype(tmp_path, duration, sample_rate):
    analyzer = SpectralAnalyzer()
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = 0.2 * np.sin(2 * np.pi * 440 * t)
    audio_path = tmp_path / "tone.wav"
    _write_wav(str(audio_path), signal, sample_rate)

    features = analyzer.extract_features(str(audio_path))

    assert features.ndim == 2
    assert features.dtype == np.float32
    assert features.shape[0] > 0
    assert features.shape[1] > 0


@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(binary_blob=st.binary(min_size=1, max_size=1024))
def test_invalid_audio_raises_structured_error(tmp_path, binary_blob):
    analyzer = SpectralAnalyzer()
    bad_audio = tmp_path / "bad.wav"
    bad_audio.write_bytes(binary_blob)

    with pytest.raises(AudioProcessingError) as exc_info:
        analyzer.extract_features(str(bad_audio))

    assert exc_info.value.error_code
    assert exc_info.value.description


def test_extract_features_serializes_to_cache(tmp_path):
    cache_dir = tmp_path / "mfcc_cache"
    analyzer = SpectralAnalyzer(feature_cache_dir=str(cache_dir))
    sample_rate = 16000
    t = np.linspace(0, 0.4, int(sample_rate * 0.4), endpoint=False)
    signal = 0.2 * np.sin(2 * np.pi * 440 * t)
    audio_path = tmp_path / "cache_source.wav"
    _write_wav(str(audio_path), signal, sample_rate)

    extracted = analyzer.extract_features(str(audio_path))
    cache_path = analyzer._get_feature_cache_path(str(audio_path))
    assert (cache_dir).exists()
    assert (tmp_path / "mfcc_cache" / (audio_path.name)).exists() is False
    assert np.allclose(extracted, analyzer.deserialize_features(cache_path))


def test_feature_round_trip(tmp_path):
    analyzer = SpectralAnalyzer()
    sample_rate = 16000
    t = np.linspace(0, 0.5, int(sample_rate * 0.5), endpoint=False)
    signal = 0.2 * np.sin(2 * np.pi * 440 * t)
    audio_path = tmp_path / "source.wav"
    feature_path = tmp_path / "features.npy"
    _write_wav(str(audio_path), signal, sample_rate)

    original = analyzer.extract_features(str(audio_path))
    analyzer.serialize_features(original, str(feature_path))
    loaded = analyzer.deserialize_features(str(feature_path))

    assert loaded.shape == original.shape
    assert loaded.dtype == original.dtype
    assert np.allclose(loaded, original)


@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    duration=st.floats(min_value=0.2, max_value=1.0, allow_nan=False, allow_infinity=False),
    base_freq=st.floats(min_value=180.0, max_value=620.0, allow_nan=False, allow_infinity=False),
)
def test_feature_round_trip_property(tmp_path, duration, base_freq):
    analyzer = SpectralAnalyzer(feature_cache_dir=str(tmp_path / "cache"))
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = (
        0.15 * np.sin(2 * np.pi * float(base_freq) * t)
        + 0.05 * np.sin(2 * np.pi * (float(base_freq) * 1.5) * t)
    )
    audio_path = tmp_path / "roundtrip.wav"
    feature_path = tmp_path / "roundtrip_features.npy"
    _write_wav(str(audio_path), signal, sample_rate)

    original = analyzer.extract_features(str(audio_path))
    analyzer.serialize_features(original, str(feature_path))
    loaded = analyzer.deserialize_features(str(feature_path))

    assert loaded.shape == original.shape
    assert loaded.dtype == np.float32
    assert np.allclose(loaded, original)


@pytest.fixture
def trained_spectral_analyzer(tmp_path):
    analyzer = SpectralAnalyzer(
        model_path=str(tmp_path / "spectral_model.joblib"),
        training_report_path=str(tmp_path / "training_report.json"),
        n_clusters=2,
    )
    sample_rate = 16000
    samples = []

    for idx, freq in enumerate([220, 250, 280, 310]):
        t = np.linspace(0, 0.6, int(sample_rate * 0.6), endpoint=False)
        signal = 0.2 * np.sin(2 * np.pi * freq * t)
        path = tmp_path / f"real_{idx}.wav"
        _write_wav(str(path), signal, sample_rate)
        samples.append({"audio_file_path": str(path), "label": "real"})

    for idx, freq in enumerate([440, 480, 520, 560]):
        t = np.linspace(0, 0.6, int(sample_rate * 0.6), endpoint=False)
        signal = 0.2 * np.sign(np.sin(2 * np.pi * freq * t))
        path = tmp_path / f"synthetic_{idx}.wav"
        _write_wav(str(path), signal, sample_rate)
        samples.append({"audio_file_path": str(path), "label": "synthetic"})

    analyzer.train(samples)
    return analyzer


@settings(
    max_examples=6,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(freq=st.floats(min_value=150.0, max_value=700.0, allow_nan=False, allow_infinity=False))
def test_spectral_score_range_invariant(tmp_path, trained_spectral_analyzer, freq):
    sample_rate = 16000
    t = np.linspace(0, 0.4, int(sample_rate * 0.4), endpoint=False)
    signal = 0.2 * np.sin(2 * np.pi * float(freq) * t)
    audio_path = tmp_path / "probe.wav"
    _write_wav(str(audio_path), signal, sample_rate)

    result = trained_spectral_analyzer.analyze(str(audio_path))
    assert 0.0 <= result.spectral_score <= 100.0


def test_spectral_training_rejects_low_f1(tmp_path, monkeypatch):
    analyzer = SpectralAnalyzer(
        model_path=str(tmp_path / "spectral_model.joblib"),
        training_report_path=str(tmp_path / "training_report.json"),
        n_clusters=2,
    )

    sample_rate = 16000
    samples = []
    for idx in range(6):
        t = np.linspace(0, 0.4, int(sample_rate * 0.4), endpoint=False)
        signal = 0.2 * np.sin(2 * np.pi * (220 + idx * 20) * t)
        path = tmp_path / f"sample_{idx}.wav"
        _write_wav(str(path), signal, sample_rate)
        samples.append({"audio_file_path": str(path), "label": "real" if idx < 3 else "synthetic"})

    monkeypatch.setattr("coreML.spectral_analyzer.f1_score", lambda *args, **kwargs: 0.5)

    with pytest.raises(AudioProcessingError) as exc_info:
        analyzer.train(samples, test_size=0.5)

    assert exc_info.value.error_code == "SPECTRAL_MIN_F1_NOT_MET"


def test_spectral_training_serializes_model_and_report(tmp_path):
    analyzer = SpectralAnalyzer(
        model_path=str(tmp_path / "spectral_model.joblib"),
        training_report_path=str(tmp_path / "training_report.json"),
        n_clusters=2,
    )

    sample_rate = 16000
    samples = []

    for idx, freq in enumerate([220, 250, 280, 310, 340, 370]):
        t = np.linspace(0, 0.6, int(sample_rate * 0.6), endpoint=False)
        signal = 0.2 * np.sin(2 * np.pi * freq * t)
        path = tmp_path / f"real_{idx}.wav"
        _write_wav(str(path), signal, sample_rate)
        samples.append({"audio_file_path": str(path), "label": "real"})

    for idx, freq in enumerate([440, 480, 520, 560, 600, 640]):
        t = np.linspace(0, 0.6, int(sample_rate * 0.6), endpoint=False)
        signal = 0.2 * np.sign(np.sin(2 * np.pi * freq * t))
        path = tmp_path / f"synthetic_{idx}.wav"
        _write_wav(str(path), signal, sample_rate)
        samples.append({"audio_file_path": str(path), "label": "synthetic"})

    f1 = analyzer.train(samples, test_size=0.5)

    assert f1 >= 0.75
    assert (tmp_path / "spectral_model.joblib").exists()
    assert (tmp_path / "training_report.json").exists()

    artifact = joblib.load(str(tmp_path / "spectral_model.joblib"))
    assert "classifier" in artifact
    assert "kmeans" in artifact

    report = json.loads((tmp_path / "training_report.json").read_text(encoding="utf-8"))
    assert report["kmeans"]["n_clusters"] >= 1
    assert "assignment_method" in report["kmeans"]


def test_spectral_analysis_completes_within_timeout_for_60s_audio(tmp_path, monkeypatch):
    analyzer = SpectralAnalyzer(timeout_seconds=5)
    monkeypatch.setattr(analyzer, "_predict_synthetic_probability", lambda _pooled: 0.8)
    monkeypatch.setattr(analyzer, "_compute_anomaly_flag", lambda _pooled: False)

    sample_rate = 8000
    t = np.linspace(0, 60.0, int(sample_rate * 60.0), endpoint=False)
    signal = 0.15 * np.sin(2 * np.pi * 220 * t)
    audio_path = tmp_path / "long_60s.wav"
    _write_wav(str(audio_path), signal, sample_rate)

    result = analyzer.analyze(str(audio_path))

    assert 0.0 <= result.spectral_score <= 100.0
    assert result.processing_time_ms <= 5_000
