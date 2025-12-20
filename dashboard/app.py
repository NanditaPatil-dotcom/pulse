import streamlit as st
import requests
import pandas as pd
import time

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Pulse", layout="wide")
st.title("Pulse — Live Health Monitor")

device_id = st.text_input("Device ID", value="pulse_001")

placeholder = st.empty()

while True:
    try:
        resp = requests.get(
            f"{API_BASE}/api/v1/history",
            params={"user_id": device_id, "hours": 1},
            timeout=5
        )

        if resp.status_code == 200:
            data = resp.json()
            df = pd.DataFrame(data)

            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp")

                with placeholder.container():
                    col1, col2 = st.columns(2)

                    col1.metric(
                        "Heart Rate",
                        int(df.iloc[-1]["heart_rate"])
                    )

                    col2.metric(
                        "SpO₂",
                        int(df.iloc[-1]["spo2"])
                    )

                    st.line_chart(
                        df.set_index("timestamp")[["heart_rate", "spo2"]]
                    )
            else:
                st.warning("No data yet for this device")

    except Exception as e:
        st.error(f"Backend not reachable: {e}")

    time.sleep(2)
