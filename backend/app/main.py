import asyncio
from datetime import datetime
from threading import Thread
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from app.consumer import start_consumer
from app.db import (
    AsyncSessionLocal,
    Vitals,
    init_db,
    insert_vital,
    fetch_latest_vital,
    fetch_history,
    fetch_metrics,
)
from app.schemas import VitalOut, PredictionResponse
from app.ml import pulse_model

# ============================================================
# App init
# ============================================================

app = FastAPI(title="Pulse Backend API")

# ============================================================
# Kafka handling
# ============================================================

async def handle_kafka_message(data: dict):
    """
    Handles Kafka message → inserts vitals → optional ML inference
    """
    try:
        async with AsyncSessionLocal() as session:
            v = Vitals(
                device_id=data["device_id"],
                user_id=data.get("user_id", data["device_id"]),
                heart_rate=data["heart_rate"],
                spo2=data["spo2"],
                temp_c=data.get("temp_c"),
                steps=data.get("steps"),
                timestamp=datetime.fromisoformat(data["timestamp"]),
            )
            session.add(v)
            await session.commit()

            # ML inference (non-blocking)
            features = {
                "heart_rate": v.heart_rate,
                "spo2": v.spo2,
                "temp_c": v.temp_c or 0.0,
                "steps": v.steps or 0,
            }
            pulse_model.predict(features)

    except Exception as e:
        print("[Kafka] Failed:", e, "| data:", data)


@app.on_event("startup")
async def startup_event():
    await init_db()

    loop = asyncio.get_running_loop()

    Thread(
        target=start_consumer,
        args=(loop, handle_kafka_message),
        daemon=True,
    ).start()

# ============================================================
# HTTP ingestion (manual / testing)
# ============================================================

class IngestPayload(BaseModel):
    device_id: str
    user_id: str
    heart_rate: int
    spo2: int
    timestamp: Optional[datetime] = None
    temp_c: Optional[float] = None
    steps: Optional[int] = None


@app.post("/api/v1/ingest")
async def http_ingest(payload: IngestPayload):
    vitals_id = await insert_vital(payload)

    features = {
        "heart_rate": payload.heart_rate,
        "spo2": payload.spo2,
        "temp_c": payload.temp_c or 0.0,
        "steps": payload.steps or 0,
    }

    result = pulse_model.predict(features)

    return {
        "status": "ok",
        "vitals_id": vitals_id,
        "prediction": result,
    }

# ============================================================
# Prediction endpoint (pure ML)
# ============================================================

@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: dict):
    result = pulse_model.predict(payload)

    if not result:
        raise HTTPException(status_code=500, detail="Model not loaded")

    return result

# ============================================================
# Read APIs
# ============================================================

@app.get("/readings/latest")
async def latest_readings():
    async with AsyncSessionLocal() as session:
        q = select(Vitals).order_by(Vitals.timestamp.desc()).limit(20)
        res = await session.execute(q)
        return res.scalars().all()


@app.get("/api/v1/live", response_model=VitalOut)
async def get_live(user_id: str):
    row = await fetch_latest_vital(user_id)
    if not row:
        raise HTTPException(status_code=404, detail="No vitals found")
    return row


@app.get("/api/v1/history", response_model=List[VitalOut])
async def get_history(user_id: str, hours: int = 24):
    return await fetch_history(user_id, hours)

# ============================================================
# System health
# ============================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "ml_loaded": pulse_model.is_loaded,
    }


@app.get("/api/v1/metrics")
async def get_model_metrics():
    return await fetch_metrics()
