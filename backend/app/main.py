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
    fetch_latest_vital,
    fetch_history,
    fetch_metrics,
)


try:
    from app.db import insert_prediction
except Exception:
    insert_prediction = None

from app.schemas import VitalOut, PredictionResponse
from app.ml import pulse_model

app = FastAPI(title="Pulse Backend API")


async def insert_vital_record_from_dict(data: dict) -> int:
    """
    Insert a vital record into DB and return the new id.
    Must be called from the main asyncio loop (i.e. not from a raw thread).
    """
    async with AsyncSessionLocal() as session:
        v = Vitals(
            device_id=data["device_id"],
            user_id=data.get("user_id", data["device_id"]),
            heart_rate=int(data["heart_rate"]),
            spo2=int(data["spo2"]),
            temp_c=float(data["temp_c"]) if data.get("temp_c") is not None else None,
            steps=int(data["steps"]) if data.get("steps") is not None else None,
            timestamp=datetime.fromisoformat(data["timestamp"])
            if data.get("timestamp")
            else datetime.utcnow(),
        )
        session.add(v)
        await session.commit()
        await session.refresh(v)
        return v.id

async def kafka_queue_worker(q: asyncio.Queue):
    """
    Runs on the main asyncio loop. Consumes message dicts from queue,
    writes them to DB and runs model inference (in executor).
    """
    print("[KafkaWorker] started")
    loop = asyncio.get_running_loop()

    while True:
        data = await q.get()
        try:

            vitals_id = await insert_vital_record_from_dict(data)
            print("[KafkaWorker] inserted vitals_id:", vitals_id)


            if pulse_model and getattr(pulse_model, "is_loaded", True):
                try:

                    result = await loop.run_in_executor(
                        None,
                        lambda: pulse_model.predict(
                            {
                                "heart_rate": data["heart_rate"],
                                "spo2": data["spo2"],
                                "temp_c": data.get("temp_c", 0.0),
                                "steps": data.get("steps", 0),
                            }
                        ),
                    )
                    print("[KafkaWorker] model result:", result)


                    if insert_prediction and result:
                        label = result.get("risk_label") or result.get("label") or str(result)
                        prob = float(result.get("confidence", result.get("probability", 0.0)))
                        shap_json = result.get("shap", {})
                        version = result.get("model_version", "v1")
                        try:
                            await insert_prediction(vitals_id, label, prob, shap_json, version)
                        except Exception as e:
                            print("[KafkaWorker] failed to insert prediction:", e)

                except Exception as e:
                    print("[KafkaWorker] model inference failed:", e)
            else:
                print("[KafkaWorker] model not loaded; skipping inference")

        except Exception as e:
            print("[Kafka] Failed:", e, "| data:", data)
        finally:
            q.task_done()


@app.on_event("startup")
async def startup_event():
    await init_db()

    loop = asyncio.get_running_loop()


    app.state.kafka_queue = asyncio.Queue()


    asyncio.create_task(kafka_queue_worker(app.state.kafka_queue))

    Thread(
        target=start_consumer,
        args=(loop, app.state.kafka_queue),
        daemon=True,
    ).start()



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

    data = payload.dict()
    if not data.get("timestamp"):
        data["timestamp"] = datetime.utcnow().isoformat()

    vitals_id = await insert_vital_record_from_dict(data)


    loop = asyncio.get_running_loop()
    result = None
    if pulse_model and getattr(pulse_model, "is_loaded", True):
        result = await loop.run_in_executor(None, lambda: pulse_model.predict({
            "heart_rate": data["heart_rate"],
            "spo2": data["spo2"],
            "temp_c": data.get("temp_c", 0.0),
            "steps": data.get("steps", 0),
        }))

        if insert_prediction and result:
            try:
                label = result.get("risk_label") or result.get("label") or str(result)
                prob = float(result.get("confidence", result.get("probability", 0.0)))
                shap_json = result.get("shap", {})
                version = result.get("model_version", "v1")
                await insert_prediction(vitals_id, label, prob, shap_json, version)
            except Exception as e:
                print("[HTTP] insert_prediction failed:", e)

    return {"status": "ok", "vitals_id": vitals_id, "prediction": result}



@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: dict):
    if not (pulse_model and getattr(pulse_model, "is_loaded", True)):
        raise HTTPException(status_code=500, detail="Model not loaded")

    loop = asyncio.get_running_loop()

    result = await loop.run_in_executor(None, lambda: pulse_model.predict(payload))

    if not result:
        raise HTTPException(status_code=500, detail="Prediction failed")

    return result


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


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "ml_loaded": bool(getattr(pulse_model, "is_loaded", False)),
    }


@app.get("/api/v1/metrics")
async def get_model_metrics():
    return await fetch_metrics()

