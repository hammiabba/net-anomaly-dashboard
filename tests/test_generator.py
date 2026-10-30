"""Tests for the synthetic telemetry generator."""

import pandas as pd
import pytest

from telemetry.generator import FEATURE_COLUMNS, LABEL_COLUMN, generate_telemetry


def test_determinism_same_seed_same_output():
    a = generate_telemetry(hours=24, n_anomalies=6, seed=42)
    b = generate_telemetry(hours=24, n_anomalies=6, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_differ():
    a = generate_telemetry(hours=24, n_anomalies=6, seed=1)
    b = generate_telemetry(hours=24, n_anomalies=6, seed=2)
    assert not a[FEATURE_COLUMNS].equals(b[FEATURE_COLUMNS])


def test_expected_columns_and_no_nans():
    df = generate_telemetry(hours=24, n_anomalies=6, seed=42)
    assert list(df.columns) == [
        "timestamp", *FEATURE_COLUMNS, LABEL_COLUMN, "anomaly_type",
    ]
    assert not df.isna().any().any()
    assert len(df) == 24 * 60


def test_anomaly_labels_present():
    df = generate_telemetry(hours=72, n_anomalies=12, seed=42)
    assert df[LABEL_COLUMN].sum() > 0
    labeled = df[df[LABEL_COLUMN] == 1]
    assert (labeled["anomaly_type"] != "").all()
    assert set(labeled["anomaly_type"].unique()) <= {"spike", "shift", "drift"}


def test_zero_anomalies_means_no_labels():
    df = generate_telemetry(hours=24, n_anomalies=0, seed=42)
    assert df[LABEL_COLUMN].sum() == 0


def test_invalid_args_rejected():
    with pytest.raises(ValueError):
        generate_telemetry(hours=0)
    with pytest.raises(ValueError):
        generate_telemetry(hours=24, n_anomalies=-1)
