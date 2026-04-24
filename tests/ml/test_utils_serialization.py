import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from config import SETTINGS
from coreML.utils import verify_serialization


def test_model_serialization_round_trip(tmp_path):
    x_train = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0, 1, 1, 0])

    model = RandomForestClassifier(
        n_estimators=50,
        random_state=SETTINGS.general.random_seed,
    )
    model.fit(x_train, y_train)

    model_path = tmp_path / "rf.joblib"
    joblib.dump(model, model_path)

    verify_serialization(model, str(model_path), x_train)
