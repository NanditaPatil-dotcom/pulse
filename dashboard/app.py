import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Pulse", layout="wide")
st.title("Pulse — Live Health Monitor")

user_id = st.text_input("Enter User ID", value="pulse_001")
API_BASE = "http://127.0.0.1:8000"

@st.cache_data(ttl=5)
def fetch_history(user_id):
    r = requests.get(
        f"{API_BASE}/api/v1/history",
        params={"user_id": user_id, "hours": 0.166}  # last 10 minutes
    )
    if r.status_code != 200:
        return pd.DataFrame()
    return pd.DataFrame(r.json())

df = fetch_history(user_id)

if df.empty:
    st.warning("No vitals found.")
    st.stop()

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").tail(200)

# smoothing
df["heart_rate_smooth"] = df["heart_rate"].rolling(5, min_periods=1).mean()
df["spo2_smooth"] = df["spo2"].rolling(5, min_periods=1).mean()

st.subheader("Vitals over time")
col1, col2 = st.columns(2)

with col1:
    st.line_chart(df.set_index("timestamp")["heart_rate_smooth"])

with col2:
    st.line_chart(df.set_index("timestamp")["spo2_smooth"])

latest = df.iloc[-1]

payload = {
    "heart_rate": int(latest["heart_rate"]),
    "spo2": int(latest["spo2"]),
    "temp_c": float(latest.get("temp_c", 0) or 0),
    "steps": int(latest.get("steps", 0) or 0),
}

resp = requests.post(f"{API_BASE}/predict", json=payload)
pred = resp.json()

st.subheader("Risk Assessment")
st.metric("Risk Label", pred["risk_label"])
st.metric("Confidence", f"{pred['confidence']:.2f}")
st.metric("Risk Score", f"{pred['risk_score']:.2f}")

st.subheader("Model Explanation (SHAP)")
shap_df = pd.DataFrame(
    pred["shap"].items(), columns=["feature", "impact"]
).sort_values("impact")

fig, ax = plt.subplots()
ax.barh(shap_df["feature"], shap_df["impact"])
ax.invert_yaxis()
st.pyplot(fig)

