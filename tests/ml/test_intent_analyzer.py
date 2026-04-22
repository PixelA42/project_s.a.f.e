import numpy as np
import pytest
import soundfile as sf
from hypothesis import given, settings, strategies as st
from hypothesis import HealthCheck

from coreML.intent_analyzer import IntentAnalyzer


class _DummyWhisperModel:
    def __init__(self, text: str):
        self._text = text

    def transcribe(self, _audio_path: str):
        return {"text": self._text}


def _silent_wav(path: str, duration: float = 0.4, sample_rate: int = 16000) -> None:
    signal = np.zeros(int(sample_rate * duration), dtype=np.float32)
    sf.write(path, signal, sample_rate)


@pytest.fixture
def analyzer_factory(monkeypatch, tmp_path):
    monkeypatch.setattr(IntentAnalyzer, "_safe_load_spacy_model", lambda self: None)
    monkeypatch.setattr(IntentAnalyzer, "_ensure_nltk_tokenizer", lambda self: None)

    def _factory(transcript_text: str) -> IntentAnalyzer:
        monkeypatch.setattr(
            IntentAnalyzer,
            "_load_whisper_model",
            staticmethod(lambda: _DummyWhisperModel(transcript_text)),
        )
        return IntentAnalyzer(model_path=str(tmp_path / "intent_model.joblib"))

    return _factory


@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(text=st.text(min_size=0, max_size=200))
def test_intent_score_range_invariant(analyzer_factory, text):
    analyzer = analyzer_factory(text)
    score = analyzer._score_transcript(text)
    assert 0.0 <= score <= 100.0


@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(base_text=st.text(min_size=0, max_size=200))
def test_keyword_density_monotonicity(analyzer_factory, base_text):
    analyzer = analyzer_factory(base_text)
    base_score = analyzer._score_transcript(base_text)
    augmented_score = analyzer._score_transcript(base_text + " transfer otp money")
    assert augmented_score >= base_score


def test_no_speech_returns_zero(analyzer_factory, tmp_path):
    analyzer = analyzer_factory("")
    audio_path = tmp_path / "silent.wav"
    _silent_wav(str(audio_path))

    result = analyzer.analyze(str(audio_path))

    assert result.intent_score == 0.0
    assert result.no_speech_detected is True
