import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

st.set_page_config(page_title="Pulse", layout="wide")
st.title("Pulse")

API_BASE = os.getenv("API_BASE", "http://backend:8000")


user_id = st.text_input("Enter User ID", value="pulse_001")
REFRESH_SECONDS = 5        
WINDOW_MINUTES = 10        


@st.cache_data(ttl=REFRESH_SECONDS)
def fetch_history(user_id: str) -> pd.DataFrame:
    r = requests.get(
        f"{API_BASE}/api/v1/history",
        params={
            "user_id": user_id,
            "hours": 1,   
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


df["heart_rate"] = df["heart_rate"].rolling(
    window=5, min_periods=1
).mean()

df["spo2"] = df["spo2"].rolling(
    window=5, min_periods=1
).mean()


st.subheader("Vitals over time")

col1, col2 = st.columns(2)

with col1:
    st.caption("Heart Rate (smoothed)")
    st.line_chart(
        df.set_index("timestamp")["heart_rate"],
        height=300,
    )

with col2:
    st.caption("SpO₂ (smoothed)")
    st.line_chart(
        df.set_index("timestamp")["spo2"],
        height=300,
    )

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

# ---- Dark themed SHAP plot ----
plt.style.use("dark_background")

fig, ax = plt.subplots(figsize=(6, 4))

# Colors
BAR_COLOR = "#7EC8FF"      # soft light blue
TEXT_COLOR = "#E6E6E6"
GRID_COLOR = "#2A2A2A"
BG_COLOR = "#0E1117"       # Streamlit dark bg match

fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

ax.barh(
    shap_df["feature"],
    shap_df["impact"],
    color=BAR_COLOR,
    alpha=0.85
)

ax.invert_yaxis()

# Labels & styling
ax.set_xlabel("Impact on Risk", color=TEXT_COLOR, fontsize=10)
ax.set_ylabel("Feature", color=TEXT_COLOR, fontsize=10)
ax.tick_params(colors=TEXT_COLOR)

# Subtle grid
ax.xaxis.grid(True, linestyle="--", alpha=0.2, color=GRID_COLOR)
ax.yaxis.grid(False)

# Remove harsh borders
for spine in ax.spines.values():
    spine.set_visible(False)

st.pyplot(fig)

time.sleep(REFRESH_SECONDS)
st.rerun()


