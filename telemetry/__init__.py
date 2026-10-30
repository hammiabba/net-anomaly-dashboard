"""Synthetic network telemetry generation with labeled anomalies."""

from .generator import generate_telemetry, FEATURE_COLUMNS, LABEL_COLUMN

__all__ = ["generate_telemetry", "FEATURE_COLUMNS", "LABEL_COLUMN"]
