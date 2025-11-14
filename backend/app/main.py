from fastapi import FastAPI
from threading import Thread
from app.consumer import start_consumer
from app.db import SessionLocal, PulseRecord
from app.schemas import PulseReading, PredictionResult
from app.ml import predict_risk

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