"""
Model loaders — called once at app startup.
Each teammate drops their trained model file into app/models/weights/
and this file handles loading it safely.
"""
import os
import joblib
from flask import Flask


def load_all_models(app: Flask) -> None:
    """
    Attempts to load all ML model files into app context.
    If a model file is missing, falls back to mock mode gracefully.
    """
    app.spectral_model = _load_model(
        path=app.config.get("SPECTRAL_MODEL_PATH", ""),
        name="Spectral",
        enabled=app.config.get("USE_REAL_SPECTRAL_MODEL", False),
    )

    app.intent_model = _load_model(
        path=app.config.get("INTENT_MODEL_PATH", ""),
        name="Intent",
        enabled=app.config.get("USE_REAL_INTENT_MODEL", False),
    )


def _load_model(path: str, name: str, enabled: bool):
    """
    Loads a joblib/pickle model from disk.
    Returns the model object, or None if disabled or not found.

    ── HOW YOUR TEAMMATE SAVES A COMPATIBLE MODEL ──────────────────────
    Any sklearn Pipeline, torch Module, or custom class works as long as
    it implements a .predict(features) -> float method.

    Example (sklearn):
        import joblib
        from sklearn.pipeline import Pipeline
        model = Pipeline([...])
        model.fit(X_train, y_train)
        joblib.dump(model, "app/models/weights/spectral_model.pkl")

    Example (PyTorch — wrap in a small adapter class):
        class TorchAdapter:
            def __init__(self, net): self.net = net
            def predict(self, features):
                import torch
                t = torch.tensor(features, dtype=torch.float32)
                return self.net(t).item()
        joblib.dump(TorchAdapter(trained_net), "spectral_model.pkl")
    ─────────────────────────────────────────────────────────────────────
    """
    if not enabled:
        print(f"[SAFE] {name} model: MOCK MODE (USE_REAL_{name.upper()}_MODEL=0)")
        return None

    if not path or not os.path.exists(path):
        print(
            f"[SAFE] WARNING: {name} model not found at '{path}'. "
            f"Falling back to mock mode. Set {name.upper()}_MODEL_PATH in .env "
            f"and ensure the .pkl file exists."
        )
        return None

    try:
        model = joblib.load(path)
        print(f"[SAFE] {name} model loaded from {path}")
        return model
    except Exception as e:
        print(f"[SAFE] ERROR loading {name} model: {e}. Falling back to mock.")
        return None