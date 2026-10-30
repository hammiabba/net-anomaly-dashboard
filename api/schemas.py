"""Pydantic schemas for the detection API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TelemetryPoint(BaseModel):
    timestamp: datetime
    latency_ms: float = Field(gt=0, description="Round-trip latency in milliseconds")
    packet_loss_pct: float = Field(ge=0, description="Packet loss as a percentage")
    throughput_mbps: float = Field(gt=0, description="Throughput in megabits per second")
    error_rate_pct: float = Field(ge=0, description="Error rate as a percentage")


class DetectRequest(BaseModel):
    points: list[TelemetryPoint] = Field(
        min_length=1, max_length=10000, description="Batch of telemetry points"
    )
    detector: Literal["isolation_forest", "zscore"] = "isolation_forest"


class DetectionResult(BaseModel):
    timestamp: datetime
    score: float = Field(description="Anomaly score; higher = more anomalous")
    is_anomaly: bool


class DetectResponse(BaseModel):
    detector: str
    results: list[DetectionResult]
