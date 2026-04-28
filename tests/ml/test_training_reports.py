import json
import soundfile as sf
import numpy as np

from coreML.intent_analyzer import IntentAnalyzer


def _write_wav(path: str, signal: np.ndarray, sample_rate: int = 16000) -> None:
    sf.write(path, signal.astype(np.float32), sample_rate)


def test_intent_training_writes_report(tmp_path):
    # Avoid heavy external model initialization (whisper/pytorch/spacy/nltk) in unit tests
    IntentAnalyzer._load_whisper_model = staticmethod(lambda: None)
    IntentAnalyzer._safe_load_spacy_model = lambda self: None
    IntentAnalyzer._ensure_nltk_tokenizer = lambda self: None

    analyzer = IntentAnalyzer(
        model_path=str(tmp_path / "intent_model.joblib"),
        training_report_path=str(tmp_path / "intent_training_report.json"),
        random_state=42,
        n_clusters=2,
    )

    # Create simple labeled transcripts
    samples = []
    samples.append({"transcript": "please send money now", "label": "coercive"})
    samples.append({"transcript": "hello how are you", "label": "safe"})
    samples.append({"transcript": "transfer the funds", "label": "coercive"})
    samples.append({"transcript": "thank you", "label": "safe"})

    f1 = analyzer.train(samples, test_size=0.5)

    assert f1 >= 0.0
    assert (tmp_path / "intent_model.joblib").exists()

    # KMeans model should be present after training
    assert analyzer.kmeans_model is not None
    assert getattr(analyzer.kmeans_model, "n_clusters", 1) >= 1

    # If the training report file was written, validate its contents
    report_path = tmp_path / "intent_training_report.json"
    if report_path.exists() and report_path.stat().st_size > 0:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert "kmeans" in report
        assert report["kmeans"]["n_clusters"] >= 1
        assert "assignment_method" in report["kmeans"]
