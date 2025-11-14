from pydantic import BaseModel
from datetime import datetime


class PulseReading(BaseModel):
    device_id: str
    heart_rate: int
    spo2: float
    timestamp: datetime


class PredictionResult(BaseModel):
    risk_score: float
    explanation: dict