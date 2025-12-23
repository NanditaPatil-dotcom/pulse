import json
import random
import time
import os
from datetime import datetime
from kafka import KafkaProducer



KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "pulse_readings")
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")

DEVICE_IDS = ["pulse_001", "pulse_002", "pulse_003"]

MODES = ["normal", "stressed", "critical"]
WEIGHTS = [0.5, 0.3, 0.2]



def generate_vitals(mode: str):
    if mode == "normal":
        return {"heart_rate": random.randint(60, 90), "spo2": random.randint(96, 100)}
    elif mode == "stressed":
        return {"heart_rate": random.randint(95, 120), "spo2": random.randint(90, 94)}
    else:
        return {"heart_rate": random.randint(120, 150), "spo2": random.randint(80, 88)}



def create_producer():
    while True:
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            print("Connected to Kafka at", KAFKA_BOOTSTRAP)
            return producer
        except Exception as e:
            print("Kafka not ready, retrying in 3s...", e)
            time.sleep(3)



def main():
    producer = create_producer()
    print("Starting Pulse data stream...")

    while True:
        device_id = random.choice(DEVICE_IDS)
        mode = random.choices(MODES, weights=WEIGHTS, k=1)[0]
        vitals = generate_vitals(mode)

        payload = {
            "device_id": device_id,
            "user_id": device_id,
            "heart_rate": vitals["heart_rate"],
            "spo2": vitals["spo2"],
            "timestamp": datetime.utcnow().isoformat(),
        }

        producer.send(KAFKA_TOPIC, value=payload)
        print(f"[{mode.upper()}] Sent → {payload}")

        time.sleep(1)

if __name__ == "__main__":
    main()


