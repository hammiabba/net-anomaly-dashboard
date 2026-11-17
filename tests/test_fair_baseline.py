"""Tests for the daily-pattern removal and the robust Z-score."""

import numpy as np
import pandas as pd
import pytest

from detector.detectors import ZScoreDetector
from detector.evaluate import evaluate
from detector.preprocess import DailyProfile
from telemetry.generator import FEATURE_COLUMNS, generate_telemetry


def test_daily_profile_removes_most_of_the_daily_swing():
    df = generate_telemetry(hours=96, n_anomalies=0, seed=3)
    resid = DailyProfile(FEATURE_COLUMNS).fit_transform(df)
    # Throughput swings ~440 Mbps over a day; what's left should be mostly noise.
    assert resid["throughput_mbps"].std() < 0.5 * df["throughput_mbps"].std()
    assert abs(resid["latency_ms"].median()) < 1.0


def test_daily_profile_needs_fit_and_a_full_day():
    df = generate_telemetry(hours=48, n_anomalies=0, seed=3)
    with pytest.raises(RuntimeError):
        DailyProfile(FEATURE_COLUMNS).transform(df)
    with pytest.raises(ValueError):
        DailyProfile(FEATURE_COLUMNS).fit(df.iloc[:600])  # 10 hours only


def test_robust_zscore_ignores_outliers_in_training():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"m": rng.normal(0, 1, 1000)})
    X.iloc[:20, 0] = 500.0  # a few huge spikes in the training data
    plain = ZScoreDetector(threshold=3.0).train(X)
    robust = ZScoreDetector(threshold=3.0, robust=True).train(X)
    probe = pd.DataFrame({"m": [6.0]})  # clearly abnormal for N(0, 1)
    assert plain.predict(probe)[0] == 0   # std blown up by spikes -> missed
    assert robust.predict(probe)[0] == 1  # median/MAD unaffected -> caught


def test_robust_flag_survives_save_load(tmp_path):
    X = generate_telemetry(hours=48, n_anomalies=4, seed=5)[FEATURE_COLUMNS]
    det = ZScoreDetector(robust=True).train(X)
    det.save(tmp_path / "z.joblib")
    loaded = ZScoreDetector.load(tmp_path / "z.joblib")
    assert loaded.robust is True
    np.testing.assert_array_equal(det.predict(X), loaded.predict(X))


def test_fixes_improve_zscore_on_drift():
    raw = evaluate(n_seeds=3, seed=0, verbose=False)
    fair = evaluate(n_seeds=3, seed=0, deseason=True, robust_z=True, verbose=False)
    assert (fair["zscore"]["recall_by_type"]["drift"]
            > raw["zscore"]["recall_by_type"]["drift"] + 0.3)
    assert fair["zscore"]["f1"] > raw["zscore"]["f1"]
