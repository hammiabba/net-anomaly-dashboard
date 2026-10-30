"""Synthetic network telemetry generator with labeled anomalies.

Produces realistic-looking network metrics (latency, packet loss, throughput,
error rate) with diurnal patterns and noise, plus injected *labeled* anomalies
of three kinds:

- ``spike``: a single-point extreme value (e.g. a latency burst).
- ``shift``: a sustained level change over tens of minutes (e.g. a degraded link).
- ``drift``: a gradual ramp over 1-2 hours (e.g. a slowly saturating queue).

Labels exist so detectors can be evaluated honestly (precision/recall/F1)
instead of eyeballed. All randomness flows through a seeded
``numpy.random.Generator``, so the same seed always yields the same dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["latency_ms", "packet_loss_pct", "throughput_mbps", "error_rate_pct"]
LABEL_COLUMN = "is_anomaly"
ANOMALY_TYPES = ("spike", "shift", "drift")

# Fixed epoch so identical seeds produce identical timestamps on any machine/date.
EPOCH = pd.Timestamp("2026-01-01 00:00:00")


def _diurnal(hours: np.ndarray, peak_hour: float, amplitude: float, base: float) -> np.ndarray:
    """Sinusoidal daily pattern peaking at ``peak_hour`` (0-24)."""
    return base + amplitude * np.sin(2 * np.pi * (hours - peak_hour + 6) / 24)


def _base_telemetry(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Healthy baseline metrics with diurnal rhythm + Gaussian noise."""
    hours = np.arange(n) / 60.0  # 1-minute samples

    latency = _diurnal(hours, peak_hour=14, amplitude=14, base=42) + rng.normal(0, 2.5, n)
    loss = _diurnal(hours, peak_hour=14, amplitude=0.12, base=0.15) + rng.normal(0, 0.05, n)
    throughput = _diurnal(hours, peak_hour=21, amplitude=220, base=780) + rng.normal(0, 30, n)
    errors = _diurnal(hours, peak_hour=14, amplitude=0.02, base=0.04) + rng.normal(0, 0.015, n)

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(EPOCH, periods=n, freq="1min"),
            "latency_ms": np.clip(latency, 5, None),
            "packet_loss_pct": np.clip(loss, 0, None),
            "throughput_mbps": np.clip(throughput, 50, None),
            "error_rate_pct": np.clip(errors, 0, None),
            LABEL_COLUMN: np.zeros(n, dtype=int),
            "anomaly_type": "",
        }
    )
    return df


def _col(df: pd.DataFrame, name: str) -> int:
    return df.columns.get_loc(name)


def _inject_spike(df: pd.DataFrame, idx: int, rng: np.random.Generator) -> None:
    # NOTE: positional .iloc throughout — .loc slices with integer labels are
    # endpoint-inclusive, which breaks vectorized ramps.
    df.iloc[idx, _col(df, "latency_ms")] *= rng.uniform(6, 10)
    df.iloc[idx, _col(df, "packet_loss_pct")] += rng.uniform(2, 5)
    df.iloc[idx, _col(df, "throughput_mbps")] *= rng.uniform(0.4, 0.6)
    df.iloc[idx, _col(df, "error_rate_pct")] *= rng.uniform(6, 10)
    df.iloc[idx, _col(df, LABEL_COLUMN)] = 1
    df.iloc[idx, _col(df, "anomaly_type")] = "spike"


def _inject_shift(df: pd.DataFrame, start: int, rng: np.random.Generator) -> None:
    duration = int(rng.integers(20, 61))  # 20-60 minutes of degraded service
    sl = slice(start, min(start + duration, len(df)))
    df.iloc[sl, _col(df, "latency_ms")] += rng.uniform(25, 45)
    df.iloc[sl, _col(df, "packet_loss_pct")] += rng.uniform(0.8, 2.0)
    df.iloc[sl, _col(df, "throughput_mbps")] *= rng.uniform(0.65, 0.8)
    df.iloc[sl, _col(df, "error_rate_pct")] *= rng.uniform(3, 5)
    df.iloc[sl, _col(df, LABEL_COLUMN)] = 1
    df.iloc[sl, _col(df, "anomaly_type")] = "shift"


def _inject_drift(df: pd.DataFrame, start: int, rng: np.random.Generator) -> None:
    duration = int(rng.integers(60, 151))  # slow ramp over 1-2.5 hours
    sl = slice(start, min(start + duration, len(df)))
    n = sl.stop - sl.start
    ramp = np.linspace(0, 1, n)
    lat, loss, tput, err = (_col(df, c) for c in FEATURE_COLUMNS)
    df.iloc[sl, lat] += ramp * rng.uniform(20, 35)
    df.iloc[sl, loss] += ramp * rng.uniform(0.5, 1.2)
    df.iloc[sl, tput] *= 1 - ramp * rng.uniform(0.15, 0.3)
    df.iloc[sl, err] += ramp * rng.uniform(0.1, 0.3)
    df.iloc[sl, _col(df, LABEL_COLUMN)] = 1
    df.iloc[sl, _col(df, "anomaly_type")] = "drift"


_INJECTORS = {"spike": _inject_spike, "shift": _inject_shift, "drift": _inject_drift}


def generate_telemetry(
    hours: float = 72,
    n_anomalies: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a labeled telemetry DataFrame.

    Anomalies are spread evenly: the timeline is split into ``n_anomalies``
    segments and one anomaly lands at a random offset inside each segment,
    which guarantees they never overlap while staying reproducible.
    """
    if hours <= 0:
        raise ValueError("hours must be positive")
    if n_anomalies < 0:
        raise ValueError("n_anomalies must be non-negative")

    rng = np.random.Generator(np.random.PCG64(seed))
    n = int(hours * 60)
    df = _base_telemetry(n, rng)

    if n_anomalies == 0:
        return df

    segment = n // n_anomalies
    for i in range(n_anomalies):
        kind = ANOMALY_TYPES[i % len(ANOMALY_TYPES)]
        # Leave a margin at the segment end so multi-point anomalies fit.
        margin = 200
        lo = i * segment + 5
        hi = max(lo + 1, (i + 1) * segment - margin)
        start = int(rng.integers(lo, hi))
        _INJECTORS[kind](df, start, rng)

    return df.reset_index(drop=True)
