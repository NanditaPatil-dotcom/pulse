from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class VitalIn(BaseModel):
device_id: str
user_id: str
timestamp: Optional[datetime]
heart_rate: int
spo2: int
temp_c: Optional[float] = None
steps: Optional[int] = None

class VitalOut(BaseModel):
id: int
device_id: str
user_id: str
timestamp: datetime
heart_rate: int
spo2: int
temp_c: Optional[float]
steps: Optional[int]

class Config:
orm_mode = True

class PredictionOut(BaseModel):
id: int
vitals_id: int
prediction_label: str
probability: float
shap_json: dict
model_version: str
created_at: datetime

class Config:
orm_mode = True