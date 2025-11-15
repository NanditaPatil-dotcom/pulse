from fastapi import FastAPI, HTTPException
from threading import Thread
from app.consumer import start_consumer
from app.db import SessionLocal, PulseRecord
from app.schemas import PulseReading, PredictionResult
from app.ml import predict_risk, model_wrapper
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
from app.db import insert_vital, insert_prediction, fetch_latest_vital, fetch_history, fetch_metrics
from app.schemas import VitalOut

app = FastAPI(title="Pulse Backend API")

@app.on_event("startup")
def startup_event():
    Thread(target=start_consumer, daemon=True).start()

@app.get("/readings/latest")
def latest_readings():
    db = SessionLocal()
    rows = db.query(PulseRecord).order_by(PulseRecord.timestamp.desc()).limit(20).all()
    return rows

@app.post("/predict", response_model=PredictionResult)
def predict_endpoint(data: PulseReading):
    score, explanation = predict_risk(data.heart_rate, data.spo2)
    return {"risk_score": score, "explanation": explanation}

class IngestPayload(BaseModel):
device_id: str
user_id: str
timestamp: Optional[datetime]
heart_rate: int
spo2: int
temp_c: Optional[float] = None
steps: Optional[int] = None

@app.post("/api/v1/ingest")
async def http_ingest(payload: IngestPayload):
# allow HTTP ingestion for prototyping
vitals_id = await insert_vital(payload)
v = payload
# Create feature vector (simple: use current values)
features = {"heart_rate": v.heart_rate,"spo2": v.spo2,"temp_c": v.temp_c or 0.0,"steps": v.steps or 0 }
# Run model inference
if model_wrapper and model_wrapper.is_loaded:
pred_label, prob, shap_json, model_version = model_wrapper.predict_with_shap(features)
await insert_prediction(vitals_id, pred_label, prob, shap_json, model_version)
else:
print("Model not loaded; skipping prediction")
return {"status": "ok", "vitals_id": vitals_id}

@app.get("/health")
async def health():
return {"status": "ok", "model_loaded": bool(model_wrapper and model_wrapper.is_loaded)}

@app.get("/api/v1/live", response_model=VitalOut)
async def get_live(user_id: str):
row = await fetch_latest_vital(user_id)
if not row:
raise HTTPException(status_code=404, detail="No vitals found")
return row

@app.get("/api/v1/history", response_model=List[VitalOut])
async def get_history(user_id: str, hours: int = 24):
rows = await fetch_history(user_id, hours)
return rows

@app.get("/api/v1/metrics")
async def get_model_metrics():
metrics = await fetch_metrics()
return metrics