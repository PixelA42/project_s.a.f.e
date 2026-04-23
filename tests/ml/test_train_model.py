import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

import train_model


def test_split_without_leakage_stratifies_by_original_audio_label():
    rows = []
    for label in [0, 0, 0, 0, 1, 1, 1, 1]:
        group_id = f"group_{len(rows)}_{label}"
        rows.append(
            {
                "file_path": f"{group_id}_a.png",
                "label": label,
                "original_audio": f"{group_id}.wav",
            }
        )
        rows.append(
            {
                "file_path": f"{group_id}_b.png",
                "label": label,
                "original_audio": f"{group_id}.wav",
            }
        )

    df = pd.DataFrame(rows)

    train_df, test_df, split_metrics = train_model._split_without_leakage(
        df,
        test_size=0.25,
        random_state=42,
    )

    assert set(train_df["original_audio"]).isdisjoint(set(test_df["original_audio"]))
    assert set(train_df["label"]) == {0, 1}
    assert set(test_df["label"]) == {0, 1}
    assert split_metrics["strategy"] == "group_stratified"


def test_split_without_leakage_rejects_conflicting_group_labels():
    df = pd.DataFrame(
        [
            {"file_path": "a.png", "label": 0, "original_audio": "same.wav"},
            {"file_path": "b.png", "label": 1, "original_audio": "same.wav"},
            {"file_path": "c.png", "label": 1, "original_audio": "other.wav"},
        ]
    )

    with pytest.raises(ValueError, match="exactly one label"):
        train_model._split_without_leakage(df, test_size=0.3, random_state=42)


def test_compute_classification_metrics_handles_single_class_probability_output():
    x_train = np.zeros((4, 2), dtype=np.float32)
    y_train = np.zeros(4, dtype=np.int64)
    x_test = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    y_test = np.array([0, 1], dtype=np.int64)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(x_train, y_train)

    metrics = train_model._compute_classification_metrics(model, x_test, y_test)

    assert metrics["class_labels"] == [0, 1]
    assert metrics["confusion_matrix"] == [[1, 0], [1, 0]]
    assert metrics["log_loss"] is None
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    assert metrics["metric_notes"]


def test_train_unsupervised_allows_single_sample():
    x_train = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

    model, metrics = train_model._train_unsupervised(x_train, random_state=42)

    assert int(model.n_clusters) == 1
    assert metrics["n_clusters"] == 1
    assert metrics["silhouette_score"] is None


def test_train_semisupervised_handles_zero_unlabeled_ratio():
    x_train = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0, 0, 1, 1], dtype=np.int64)
    x_test = x_train.copy()
    y_test = y_train.copy()

    base_model = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
    model, metrics = train_model._train_semisupervised(
        base_model,
        x_train,
        y_train,
        x_test,
        y_test,
        unlabeled_ratio=0.0,
        threshold=0.85,
        random_state=42,
    )

    assert hasattr(model, "classes_")
    assert metrics["pseudo_labels_added"] == 0
    assert metrics["seed_samples"] == 4
    assert metrics["unlabeled_pool_samples"] == 0
    assert metrics["seed_label_distribution"] == {"0": 2, "1": 2}


def test_train_semisupervised_rejects_invalid_unlabeled_ratio():
    x_train = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    y_train = np.array([0, 1], dtype=np.int64)
    base_model = RandomForestClassifier(n_estimators=10, random_state=42)

    with pytest.raises(ValueError, match="semi-unlabeled-ratio"):
        train_model._train_semisupervised(
            base_model,
            x_train,
            y_train,
            x_train,
            y_train,
            unlabeled_ratio=1.0,
            threshold=0.85,
            random_state=42,
        )
