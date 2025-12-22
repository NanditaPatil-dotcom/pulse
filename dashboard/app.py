import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import time

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="Pulse", layout="wide")
st.title("Pulse — Live Health Monitor")

API_BASE = "http://127.0.0.1:8000"

# --------------------------------------------------
# Controls
# --------------------------------------------------
user_id = st.text_input("Enter User ID", value="pulse_001")
REFRESH_SECONDS = 5        # dashboard refresh rate
WINDOW_MINUTES = 10        # last 10 minutes of data

# --------------------------------------------------
# Data fetch
# --------------------------------------------------
@st.cache_data(ttl=REFRESH_SECONDS)
def fetch_history(user_id: str) -> pd.DataFrame:
    r = requests.get(
        f"{API_BASE}/api/v1/history",
        params={
            "user_id": user_id,
            "hours": 1,   # IMPORTANT: backend expects int hours
        },
        timeout=5,
    )
    if r.status_code != 200:
        return pd.DataFrame()
    return pd.DataFrame(r.json())


df = fetch_history(user_id)

if df.empty:
    st.warning("No vitals found.")
    st.stop()

# --------------------------------------------------
# Time filtering (frontend, precise)
# --------------------------------------------------
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
df = df.sort_values("timestamp")

cutoff = (
    pd.Timestamp.utcnow()
    .tz_localize(None)
    - pd.Timedelta(minutes=WINDOW_MINUTES)
)
df = df[df["timestamp"] >= cutoff].tail(200)

if df.empty:
    st.warning("No vitals in the selected time window.")
    st.stop()

# --------------------------------------------------
# Smoothing
# --------------------------------------------------
df["heart_rate_smooth"] = df["heart_rate"].rolling(
    window=5, min_periods=1
).mean()

df["spo2_smooth"] = df["spo2"].rolling(
    window=5, min_periods=1
).mean()

# --------------------------------------------------
# Charts
# --------------------------------------------------
st.subheader("Vitals over time")

col1, col2 = st.columns(2)

with col1:
    st.caption("Heart Rate (smoothed)")
    st.line_chart(
        df.set_index("timestamp")["heart_rate_smooth"],
        height=300,
    )

with col2:
    st.caption("SpO₂ (smoothed)")
    st.line_chart(
        df.set_index("timestamp")["spo2_smooth"],
        height=300,
    )

# --------------------------------------------------
# Prediction (latest point)
# --------------------------------------------------
latest = df.iloc[-1]

payload = {
    "heart_rate": int(latest["heart_rate"]),
    "spo2": int(latest["spo2"]),
    "temp_c": float(latest.get("temp_c", 0) or 0),
    "steps": int(latest.get("steps", 0) or 0),
}

resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=5)

if resp.status_code != 200:
    st.error("Prediction failed")
    st.stop()

pred = resp.json()

# --------------------------------------------------
# Risk summary
# --------------------------------------------------
st.subheader("Risk Assessment")

c1, c2, c3 = st.columns(3)
c1.metric("Risk Label", pred["risk_label"])
c2.metric("Confidence", f"{pred['confidence']:.2f}")
c3.metric("Risk Score", f"{pred['risk_score']:.2f}")

# --------------------------------------------------
# SHAP explanation
# --------------------------------------------------
st.subheader("Model Explanation (SHAP)")

shap_df = (
    pd.DataFrame(pred["shap"].items(), columns=["feature", "impact"])
    .sort_values("impact")
)

fig, ax = plt.subplots(figsize=(6, 3))
ax.barh(shap_df["feature"], shap_df["impact"])
ax.set_xlabel("Impact on Risk")
ax.invert_yaxis()
st.pyplot(fig)

# --------------------------------------------------
# Auto refresh
# --------------------------------------------------
time.sleep(REFRESH_SECONDS)
st.rerun()


