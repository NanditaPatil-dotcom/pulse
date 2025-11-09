# Alpine--pipline

Node 1 — Smartwatch Simulator
      ↓ (Kafka Producer)
Kafka Broker (Topic: pulse_readings)
      ↓ (Kafka Consumer)
Node 2 — Backend (FastAPI + ML + DB)
      ↓ (REST APIs)
Node 3 — Streamlit Dashboard
