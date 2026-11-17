"""Remove the daily pattern from telemetry before detection.

Network metrics rise and fall every day (busy afternoons, quiet nights). If a
detector sees the raw values, that daily swing makes the "normal" range very
wide, and slow problems hide inside it.

``DailyProfile`` learns the typical value of each metric for every hour of the
day (the median, so injected anomalies in the training window barely move it)
and subtracts it. What's left is "how far is this reading from what's normal
for this time of day", which is what the detectors should be looking at.

The profile is fitted on the training window only, so no information from the
test window leaks into it.
"""

from __future__ import annotations

import pandas as pd


class DailyProfile:
    """Per-hour-of-day median profile, subtracted from each metric."""

    def __init__(self, features: list[str], time_col: str = "timestamp"):
        self.features = list(features)
        self.time_col = time_col
        self._profile: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame) -> "DailyProfile":
        hours = pd.to_datetime(df[self.time_col]).dt.hour
        self._profile = df[self.features].groupby(hours.to_numpy()).median()
        if len(self._profile) < 24:
            raise ValueError("need at least one full day of data to fit the daily profile")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._profile is None:
            raise RuntimeError("DailyProfile must be fitted before transform")
        hours = pd.to_datetime(df[self.time_col]).dt.hour.to_numpy()
        expected = self._profile.loc[hours].to_numpy()
        return pd.DataFrame(
            df[self.features].to_numpy() - expected,
            columns=self.features,
            index=df.index,
        )

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
