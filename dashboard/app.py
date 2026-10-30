"""Streamlit dashboard: network telemetry anomaly monitoring.

Run with:
    streamlit run dashboard/app.py

Features:
- Generate (or re-generate) labeled synthetic telemetry from the sidebar.
- Pick the detector; see Isolation Forest vs. Z-score side by side.
- Plotly time-series charts with detected anomalies highlighted.
- Alert table of the most recent anomalies + summary metric cards.
- "Live stream" simulation that replays the series as if it were arriving now.
"""

from __future__ import annotations

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import precision_recall_fscore_support

from detector.detectors import IsolationForestDetector, ZScoreDetector
from telemetry.generator import FEATURE_COLUMNS, LABEL_COLUMN, generate_telemetry

st.set_page_config(page_title="Net Anomaly Dashboard", layout="wide")
st.title("Network Anomaly Detection Dashboard")

# ---------------------------------------------------------------- sidebar ---
with st.sidebar:
    st.header("Data")
    hours = st.slider("Hours of telemetry", 24, 168, 72, step=24)
    n_anomalies = st.slider("Injected anomalies", 0, 40, 12)
    seed = st.number_input("Random seed", value=42, step=1)
    st.header("Detection")
    detector_name = st.selectbox("Detector", ["isolation_forest", "zscore"])
    contamination = st.slider("IF contamination", 0.01, 0.20, 0.08, step=0.01)
    z_threshold = st.slider("Z-score threshold", 1.5, 5.0, 3.0, step=0.5)
    live = st.checkbox("Simulate live stream", value=False)


@st.cache_data(show_spinner="Generating telemetry…")
def load_data(hours: int, n_anomalies: int, seed: int) -> pd.DataFrame:
    return generate_telemetry(hours=hours, n_anomalies=n_anomalies, seed=int(seed))


@st.cache_resource(show_spinner="Training detector…")
def train_detector(
    name: str, contamination: float, z_threshold: float, _df: pd.DataFrame
):
    det = (
        IsolationForestDetector(contamination=contamination)
        if name == "isolation_forest"
        else ZScoreDetector(threshold=z_threshold)
    )
    # Train on the first 70% (the past), detect on all of it for display.
    split = int(len(_df) * 0.7)
    det.train(_df[FEATURE_COLUMNS].iloc[:split])
    return det


df = load_data(hours, n_anomalies, seed)
detector = train_detector(detector_name, contamination, z_threshold, df)
df = df.copy()
df["score"] = detector.score(df[FEATURE_COLUMNS])
df["predicted"] = detector.predict(df[FEATURE_COLUMNS])

# ------------------------------------------------------------ metric cards ---
n_points = len(df)
n_detected = int(df["predicted"].sum())
col1, col2, col3, col4 = st.columns(4)
col1.metric("Data points", f"{n_points:,}")
col2.metric("Anomalies detected", f"{n_detected:,}")
col3.metric("Detection rate", f"{100 * n_detected / n_points:.2f}%")
if df[LABEL_COLUMN].sum() > 0:  # labels exist in synthetic mode: show real quality
    p, r, f1, _ = precision_recall_fscore_support(
        df[LABEL_COLUMN], df["predicted"], average="binary", zero_division=0
    )
    col4.metric("F1 vs. ground truth", f"{f1:.3f}")
    st.caption(f"Precision {p:.3f} · Recall {r:.3f} — computed against injected labels.")
else:
    col4.metric("F1 vs. ground truth", "n/a")


def _timeseries_figure(data: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Latency (ms)", "Packet loss (%)", "Throughput (Mbps)", "Error rate (%)"),
        shared_xaxes=True, vertical_spacing=0.08,
    )
    anomalies = data[data["predicted"] == 1]
    for i, col in enumerate(FEATURE_COLUMNS):
        row, c = divmod(i, 2)
        row, c = row + 1, c + 1
        fig.add_trace(
            go.Scatter(x=data["timestamp"], y=data[col], mode="lines",
                       name=col, line=dict(width=1), showlegend=False),
            row=row, col=c,
        )
        fig.add_trace(
            go.Scatter(x=anomalies["timestamp"], y=anomalies[col], mode="markers",
                       name="anomaly", marker=dict(color="red", size=6, symbol="x"),
                       showlegend=(i == 0)),
            row=row, col=c,
        )
    fig.update_layout(height=600, title_text="Telemetry with detected anomalies")
    return fig


# ------------------------------------------------------------------ stream ---
if live:
    st.subheader("Live stream simulation")
    tail = df.tail(240)  # last 4 hours, replayed
    speed = st.slider("Points per second", 1, 20, 5)
    placeholder = st.empty()
    alert_box = st.empty()
    if st.button("▶ Replay stream"):
        for k in range(10, len(tail) + 1):
            window = tail.iloc[:k]
            with placeholder.container():
                st.plotly_chart(_timeseries_figure(window), use_container_width=True,
                                key=f"live-{k}")
            latest = tail.iloc[k - 1]
            if latest["predicted"] == 1:
                alert_box.error(
                    f"ALERT {latest['timestamp']}: anomaly score "
                    f"{latest['score']:.2f} "
                    f"(latency={latest['latency_ms']:.1f} ms, "
                    f"loss={latest['packet_loss_pct']:.2f}%)"
                )
            time.sleep(1 / speed)
else:
    st.plotly_chart(_timeseries_figure(df), use_container_width=True)

# -------------------------------------------------------------- alert table ---
st.subheader("Recent alerts")
alerts = df[df["predicted"] == 1].sort_values("timestamp", ascending=False)
display_cols = ["timestamp", "score", *FEATURE_COLUMNS]
if not alerts.empty:
    st.dataframe(alerts[display_cols].head(50), use_container_width=True,
                 hide_index=True)
else:
    st.info("No anomalies detected with the current settings.")

with st.expander("How detection works"):
    st.markdown(
        "- **Isolation Forest** isolates observations by random partitioning; points "
        "with short average path lengths are anomalous. It sees the *joint* "
        "distribution across all four metrics.\n"
        "- **Z-score** flags a point when any single metric is more than the "
        "threshold standard deviations from its training mean.\n"
        "- Both train on the first 70% of the series (the past) and score the rest."
    )
