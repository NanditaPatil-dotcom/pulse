# Pulse – Live Health Monitoring System

Pulse is a real-time health monitoring system built using a distributed, event-driven architecture. It simulates wearable health data, streams it through Kafka, processes it using a FastAPI backend with an ML risk model, stores results in PostgreSQL, and visualizes live vitals and predictions via a Streamlit dashboard.

This project is intentionally designed to demonstrate **system design, Kafka-based streaming, async backends, ML inference, and Docker-based orchestration**.

---

## Architecture Overview

```
Simulator  ──▶  Kafka  ──▶  Backend (FastAPI + ML)  ──▶  PostgreSQL
                                      │
                                      └──▶  Streamlit Dashboard
```

### Components

* **Simulator**: Generates synthetic pulse/SpO2 data and publishes to Kafka
* **Kafka + Zookeeper**: Event streaming backbone
* **Backend**: FastAPI service that

  * Consumes Kafka events
  * Stores vitals in PostgreSQL
  * Runs ML risk prediction + SHAP explainability
* **PostgreSQL**: Persistent storage for vitals and predictions
* **Dashboard**: Streamlit app for live visualization

---

## Tech Stack

* Python 3
* Apache Kafka
* FastAPI
* PostgreSQL + SQLAlchemy (async)
* Streamlit
* Scikit-learn + SHAP
* Docker & Docker Compose

---

## Prerequisites

* Docker (v20+)
* Docker Compose (v2+)
* Git

No local Python or Kafka installation is required when using Docker.

---

## How to Fork and Run the Project

### 1. Fork the Repository

* Click **Fork** on GitHub
* Clone your fork:

```bash
git clone https://github.com/NanditaPatil-dotcom/pulse.git
cd pulse
```

---

### 2. Environment Configuration

Create an `.env` file inside the `infra/` folder:

```env
KAFKA_BOOTSTRAP=kafka:9092
KAFKA_TOPIC=pulse_readings
DATABASE_URL=postgresql+asyncpg://pulse_admin:pulse_pass@postgres:5432/pulse_db
MODEL_PATH=/models/model.pkl
```

The `.env` file is intentionally ignored by Git.

---

### 3. Build and Start All Services

From the `infra/` directory:

```bash
cd infra
docker compose build
docker compose up
```

This starts:

* Zookeeper
* Kafka broker
* PostgreSQL
* Backend API
* Simulator (producer)
* Dashboard

---

### 4. Access the System

#### Dashboard (Streamlit)

```
http://localhost:8501
```

#### Backend API

```
http://localhost:8000
```

Health check:

```
GET /health
```

---

## Accessing Docker Containers

### List running containers

```bash
docker compose ps
```

### View logs

```bash
docker compose logs backend
docker compose logs simulator
docker compose logs kafka
```

### Exec into a container

```bash
docker compose exec backend bash
docker compose exec postgres psql -U pulse_admin -d pulse_db
```

---

## Kafka Topics

The simulator publishes vitals to:

```
pulse_readings
```

To inspect Kafka manually:

```bash
docker compose exec kafka kafka-topics \
  --bootstrap-server kafka:9092 \
  --list
```

---

## Database Schema

### vitals

* device_id
* user_id
* heart_rate
* spo2
* temperature
* steps
* timestamp

### predictions

* vitals_id
* risk_label
* probability
* shap_json
* model_version
* created_at

---

## Key Design Decisions

* Kafka decouples ingestion from processing
* Async FastAPI prevents blocking during DB + ML operations
* ML inference is offloaded to executors
* SHAP values stored for explainability
* Docker Compose ensures reproducibility

---

## Notes

* This project is intended for **local development and learning**
* Kafka makes this unsuitable for simple PaaS deployment, but ideal for system design demos
* All secrets are handled via environment variables

---


## Author

Built as a systems-focused engineering project demonstrating real-time data pipelines, ML inference, and production-style infrastructure.
Not for clinical use.
