from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


EVENTS_PATH = Path("outputs/events.csv")
SUMMARY_PATH = Path("outputs/frame_summary.csv")


def read_csv(uploaded_file, default_path: Path) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if default_path.exists():
        st.caption(f"Membaca {default_path}")
        return pd.read_csv(default_path)
    return pd.DataFrame()


st.set_page_config(page_title="Nexar Near-Miss Dashboard", layout="wide")
st.title("Nexar YOLO Risk Dashboard")
st.caption("Model 1: YOLO pretrained + tracking + rule-based risk engine")

with st.sidebar:
    st.header("Input")
    uploaded_events = st.file_uploader("Unggah events.csv", type=["csv"])
    uploaded_summary = st.file_uploader(
        "Unggah frame_summary.csv",
        type=["csv"],
    )

events = read_csv(uploaded_events, EVENTS_PATH)
summary = read_csv(uploaded_summary, SUMMARY_PATH)

if events.empty and summary.empty:
    st.info("Jalankan inferensi atau unggah CSV hasil model 1.")
    st.stop()

risk_count = 0
safe_count = 0
average_risk = 0.0
average_ttc = "N/A"

if not events.empty:
    risk_count = int((events["status"] == "RISK").sum())
    average_risk = float(events["risk_score"].mean())
    valid_ttc = pd.to_numeric(events["ttc_seconds"], errors="coerce").dropna()
    if not valid_ttc.empty:
        average_ttc = f"{valid_ttc.mean():.2f}s"
elif not summary.empty:
    risk_count = int((summary["max_risk_status"] == "RISK").sum())
    safe_count = int((summary["max_risk_status"] == "SAFE").sum())
    average_risk = float(summary["max_risk_score"].mean())
    valid_ttc = pd.to_numeric(
        summary["max_risk_ttc_seconds"],
        errors="coerce",
    ).dropna()
    if not valid_ttc.empty:
        average_ttc = f"{valid_ttc.mean():.2f}s"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total event", len(events))
col2.metric("RISK", risk_count)
col3.metric("SAFE frame", safe_count)
col4.metric("TTC rata-rata", average_ttc)

st.metric("Risk rata-rata", f"{average_risk:.1f}")

if not summary.empty:
    st.subheader("Risk over time")
    timeline = summary.set_index("time_seconds")[["max_risk_score"]]
    st.line_chart(timeline)
elif not events.empty:
    st.subheader("Risk over time")
    timeline = events.set_index("time_seconds")[["risk_score"]]
    st.line_chart(timeline)

left, right = st.columns(2)

with left:
    st.subheader("Objek paling berisiko")
    if not events.empty:
        st.bar_chart(events["object"].value_counts())
    elif "max_risk_object" in summary:
        objects = summary["max_risk_object"].replace("", pd.NA).dropna()
        st.bar_chart(objects.value_counts())

with right:
    st.subheader("Distribusi status")
    if not events.empty:
        st.bar_chart(events["status"].value_counts())
    else:
        st.bar_chart(summary["max_risk_status"].value_counts())

if not events.empty:
    st.subheader("Daftar event")
    st.dataframe(
        events.sort_values("risk_score", ascending=False),
        use_container_width=True,
    )

if not summary.empty:
    st.subheader("Ringkasan frame")
    st.dataframe(
        summary.sort_values("max_risk_score", ascending=False).head(300),
        use_container_width=True,
    )
