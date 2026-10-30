"""Tests for the anomaly detectors."""

import numpy as np
import pytest
from sklearn.metrics import f1_score

from detector.detectors import IsolationForestDetector, ZScoreDetector
from telemetry.generator import FEATURE_COLUMNS, LABEL_COLUMN, generate_telemetry


@pytest.fixture(scope="module")
def labeled():
    # 5 days, chronological 70/30 split like production.
    df = generate_telemetry(hours=120, n_anomalies=20, seed=11)
    split = int(len(df) * 0.7)
    return df.iloc[:split], df.iloc[split:]


def test_isolation_forest_f1_on_labeled_data(labeled):
    train, test = labeled
    det = IsolationForestDetector(contamination=0.08).train(train[FEATURE_COLUMNS])
    pred = det.predict(test[FEATURE_COLUMNS])
    f1 = f1_score(test[LABEL_COLUMN], pred, zero_division=0)
    assert f1 > 0.7, f"Isolation Forest F1={f1:.3f}, expected > 0.7"


def test_zscore_baseline_runs(labeled):
    train, test = labeled
    det = ZScoreDetector(threshold=3.0).train(train[FEATURE_COLUMNS])
    pred = det.predict(test[FEATURE_COLUMNS])
    assert set(np.unique(pred)) <= {0, 1}
    assert len(pred) == len(test)


def test_predict_and_score_shapes(labeled):
    train, test = labeled
    X = test[FEATURE_COLUMNS]
    for det in (IsolationForestDetector().train(train[FEATURE_COLUMNS]),
                ZScoreDetector().train(train[FEATURE_COLUMNS])):
        assert det.predict(X).shape == (len(X),)
        assert det.score(X).shape == (len(X),)


def test_score_ordering_spike_scores_highest(labeled):
    """An extreme point spike should out-score a normal point."""
    train, test = labeled
    X_train, X_test = train[FEATURE_COLUMNS], test[FEATURE_COLUMNS]
    spikes = test[test["anomaly_type"] == "spike"]
    normals = test[test[LABEL_COLUMN] == 0]
    assert len(spikes) > 0 and len(normals) > 0
    det = IsolationForestDetector().train(X_train)
    assert det.score(X_test.loc[spikes.index]).mean() > det.score(
        X_test.loc[normals.index]
    ).mean()


def test_save_load_roundtrip(labeled, tmp_path):
    train, test = labeled
    X_train, X_test = train[FEATURE_COLUMNS], test[FEATURE_COLUMNS]
    for cls, kwargs in ((IsolationForestDetector, {}), (ZScoreDetector, {})):
        det = cls(**kwargs).train(X_train)
        path = tmp_path / "model.joblib"
        det.save(path)
        loaded = cls.load(path)
        np.testing.assert_array_equal(det.predict(X_test), loaded.predict(X_test))


def test_untrained_zscore_raises(labeled):
    _, test = labeled
    with pytest.raises(RuntimeError):
        ZScoreDetector().predict(test[FEATURE_COLUMNS])
