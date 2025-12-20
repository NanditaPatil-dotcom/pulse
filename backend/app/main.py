import asyncio
from fastapi import FastAPI, HTTPException
from threading import Thread
from typing import List, Optional
from datetime import datetime
from sqlalchemy import select
from pydantic import BaseModel

from app.consumer import start_consumer
from app.db import (
    AsyncSessionLocal,
    Vitals,
    init_db,
    insert_vital,
    insert_prediction,
    fetch_latest_vital,
    fetch_history,
    fetch_metrics,
)
from app.schemas import VitalIn, VitalOut, PredictionOut
from app.ml import predict_risk, model_wrapper

# -------------------------------------------------------------------
# App init
# -------------------------------------------------------------------

app = FastAPI(title="Pulse Backend API")


# -------------------------------------------------------------------
# Startup: init DB + start Kafka consumer
# -------------------------------------------------------------------
async def handle_kafka_message(data: dict):
    """
    Convert Kafka JSON dict -> VitalIn -> DB
    """
    try:
        vital = VitalIn(**data)
        await insert_vital(vital)
    except Exception as e:
        print("[Kafka] Failed to insert vital:", e, "| data:", data)


@app.on_event("startup")
async def startup_event():
    await init_db()
    Thread(
    target=lambda: asyncio.run(start_consumer(handle_kafka_message)),
    daemon=True
    ).start()



# -------------------------------------------------------------------
# Basic read endpoint (debug)
# -------------------------------------------------------------------

@app.get("/readings/latest")
async def latest_readings():
    async with AsyncSessionLocal() as session:
        q = select(Vitals).order_by(Vitals.timestamp.desc()).limit(20)
        res = await session.execute(q)
        rows = res.scalars().all()
        return rows


# -------------------------------------------------------------------
# ML prediction endpoint (simple)
# -------------------------------------------------------------------

@app.post("/predict", response_model=PredictionOut)
def predict_endpoint(data: VitalIn):
    score, explanation = predict_risk(data.heart_rate, data.spo2)
    return {"risk_score": score, "explanation": explanation}


# -------------------------------------------------------------------
# HTTP ingestion payload (optional path alongside Kafka)
# -------------------------------------------------------------------

class IngestPayload(BaseModel):
    device_id: str
    user_id: str
    timestamp: Optional[datetime] = None
    heart_rate: int
    spo2: int
    temp_c: Optional[float] = None
    steps: Optional[int] = None


@app.post("/api/v1/ingest")
async def http_ingest(payload: IngestPayload):
    """
    Allow HTTP ingestion for prototyping / testing
    """
    vitals_id = await insert_vital(payload)

    # Feature vector for ML
    features = {
        "heart_rate": payload.heart_rate,
        "spo2": payload.spo2,
        "temp_c": payload.temp_c or 0.0,
        "steps": payload.steps or 0,
    }

    # Run ML model if loaded
    if model_wrapper and model_wrapper.is_loaded:
        pred_label, prob, shap_json, model_version = (
            model_wrapper.predict_with_shap(features)
        )
        await insert_prediction(
            vitals_id, pred_label, prob, shap_json, model_version
        )
    else:
        print("Model not loaded; skipping prediction")

    return {"status": "ok", "vitals_id": vitals_id}


# -------------------------------------------------------------------
# Health check
# -------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": bool(model_wrapper and model_wrapper.is_loaded),
    }


# -------------------------------------------------------------------
# Live vitals (latest per user)
# -------------------------------------------------------------------

@app.get("/api/v1/live", response_model=VitalOut)
async def get_live(user_id: str):
    row = await fetch_latest_vital(user_id)
    if not row:
        raise HTTPException(status_code=404, detail="No vitals found")
    return row


# -------------------------------------------------------------------
# Historical vitals
# -------------------------------------------------------------------

@app.get("/api/v1/history", response_model=List[VitalOut])
async def get_history(user_id: str, hours: int = 24):
    rows = await fetch_history(user_id, hours)
    return rows


# -------------------------------------------------------------------
# Model metrics
# -------------------------------------------------------------------

@app.get("/api/v1/metrics")
async def get_model_metrics():
    metrics = await fetch_metrics()
    return metrics
