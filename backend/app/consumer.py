from kafka import KafkaConsumer
import json
from datetime import datetime
from app.db import SessionLocal, PulseRecord
from app.ml import predict_risk

def start_consumer():
    consumer = KafkaConsumer(
        "pulse_readings",
        bootstrap_servers="localhost:9092",
        value_deserializer=lambda v: json.loads(v.decode("utf-8"))
    )

    db = SessionLocal()

    print("Yoo Kafka Consumer listening...")

    for msg in consumer:
        data = msg.value

        # Save to DB
        entry = PulseRecord(
            device_id=data["device_id"],
            heart_rate=data["heart_rate"],
            spo2=data["spo2"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )
        db.add(entry)
        db.commit()

        # Run ML inference for each reading
        score, expl = predict_risk(data["heart_rate"], data["spo2"])
        print(f"ML Risk Score: {score:.4f} | Explanation: {expl}")