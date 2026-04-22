"""Core ML helper utilities."""

from typing import Any

import joblib
import numpy as np


def verify_serialization(model: Any, model_path: str, x_test: np.ndarray) -> None:
    """Assert serialized model predictions match in-memory model predictions."""
    original_predictions = model.predict(x_test)
    loaded_model = joblib.load(model_path)
    loaded_predictions = loaded_model.predict(x_test)

    if not np.array_equal(original_predictions, loaded_predictions):
        raise AssertionError(
            "Serialized model predictions do not match original in-memory predictions."
        )
