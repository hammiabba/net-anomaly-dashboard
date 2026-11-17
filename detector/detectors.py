"""Anomaly detectors with a shared train/predict/score interface.

- :class:`IsolationForestDetector` — the primary detector. An ensemble method
  that isolates observations by random partitioning; points that are easy to
  isolate (short average path length) are anomalous. Works on the *joint*
  distribution of the metrics, so it catches multivariate deviations (e.g. a
  subtle drift across latency + loss + throughput) that per-metric thresholds
  miss.
- :class:`ZScoreDetector` — the statistical baseline. Flags a point when any
  single metric deviates more than ``threshold`` standard deviations from its
  training mean. Simple, explainable, and blind to cross-metric patterns.

Both expose ``train(X)`` / ``predict(X)`` / ``score(X)`` where ``predict``
returns 0/1 labels and ``score`` returns a continuous anomaly score
(higher = more anomalous).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class IsolationForestDetector:
    """Multivariate anomaly detection via Isolation Forest."""

    def __init__(self, contamination: float = 0.08, random_state: int = 42):
        if not 0 < contamination < 0.5:
            raise ValueError("contamination must be in (0, 0.5)")
        self.contamination = contamination
        self.random_state = random_state
        self._pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "forest",
                    IsolationForest(
                        contamination=contamination,
                        random_state=random_state,
                        n_estimators=200,
                    ),
                ),
            ]
        )

    def train(self, X: pd.DataFrame) -> "IsolationForestDetector":
        self._pipe.fit(X)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """0 = normal, 1 = anomaly."""
        return (self._pipe.predict(X) == -1).astype(int)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Higher = more anomalous (negated decision function)."""
        return -self._pipe.decision_function(X)

    def save(self, path: str | Path) -> None:
        joblib.dump(
            {"contamination": self.contamination, "random_state": self.random_state,
             "pipe": self._pipe},
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "IsolationForestDetector":
        blob = joblib.load(path)
        det = cls(contamination=blob["contamination"], random_state=blob["random_state"])
        det._pipe = blob["pipe"]
        return det


class ZScoreDetector:
    """Univariate statistical baseline: max |z| across metrics vs. threshold.

    ``robust=False`` (default) uses the mean and standard deviation.
    ``robust=True`` uses the median and MAD (median absolute deviation, scaled
    by 1.4826 so it matches the std for normal data). Spikes, shifts and drifts
    in the training window inflate the std a lot and push the threshold too
    high; the median and MAD barely move.
    """

    MAD_SCALE = 1.4826

    def __init__(self, threshold: float = 3.0, robust: bool = False):
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.threshold = threshold
        self.robust = robust
        self._mean: pd.Series | None = None  # center (median when robust)
        self._std: pd.Series | None = None   # spread (scaled MAD when robust)

    def train(self, X: pd.DataFrame) -> "ZScoreDetector":
        if self.robust:
            self._mean = X.median()
            spread = self.MAD_SCALE * (X - self._mean).abs().median()
        else:
            self._mean = X.mean()
            spread = X.std()
        self._std = spread.replace(0, 1e-9)  # guard against constant columns
        return self

    def _zscores(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._mean is None or self._std is None:
            raise RuntimeError("detector must be trained before predict/score")
        return (X - self._mean).abs() / self._std

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self._zscores(X).max(axis=1) > self.threshold).astype(int).to_numpy()

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Higher = more anomalous (max |z| across metrics)."""
        return self._zscores(X).max(axis=1).to_numpy()

    def save(self, path: str | Path) -> None:
        joblib.dump(
            {"threshold": self.threshold, "robust": self.robust,
             "mean": self._mean, "std": self._std},
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ZScoreDetector":
        blob = joblib.load(path)
        det = cls(threshold=blob["threshold"], robust=blob.get("robust", False))
        det._mean, det._std = blob["mean"], blob["std"]
        return det
