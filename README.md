# Pulse
```
Node 1 — Smartwatch Simulator
      ↓ (Kafka Producer)
Kafka Broker (Topic: pulse_readings)
      ↓ (Kafka Consumer)
Node 2 — Backend (FastAPI + ML + DB)
      ↓ (Fast APIs)
Node 3 — Streamlit Dashboard
```
---
# To run the nodes
 # 1. infra
 ```
  cd infra
  docker compose up -d
 ```
 # 2. backend
 ```
  cd backend
  uvicorn app.main:app --reload
 ```
# 3. simulator
 ```
  cd simulator
  python producer.py
 ```
# 4. dashboard
 ```
  cd dashboard
  streamlit run app.py
 ```
