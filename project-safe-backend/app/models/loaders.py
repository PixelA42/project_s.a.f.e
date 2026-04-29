"""
Model loaders — called once at app startup.
Each teammate drops their trained model file into app/models/weights/
and this file handles loading it safely.
"""
import os
import torch
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
    Loads a PyTorch model checkpoint from disk via torch.load.
    Returns the model object, or None if disabled or not found.
    """
    if not enabled:
        print(f"[SAFE] {name} model: MOCK MODE (USE_REAL_{name.upper()}_MODEL=0)")
        return None

    if not path or not os.path.exists(path):
        print(
            f"[SAFE] WARNING: {name} model not found at '{path}'. "
            f"Falling back to mock mode. Set {name.upper()}_MODEL_PATH in .env "
            f"and ensure the .pt/.pth file exists."
        )
        return None

    try:
        model = torch.load(path, map_location="cpu")
        if hasattr(model, "eval"):
            model.eval()
        print(f"[SAFE] {name} model loaded from {path}")
        return model
    except Exception as e:
        print(f"[SAFE] ERROR loading {name} model: {e}. Falling back to mock.")
        return None