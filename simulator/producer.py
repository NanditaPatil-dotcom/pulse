import json
import random
import time
from datetime import datetime
from kafka import KafkaProducer

# ------------------------
# Configuration
# ------------------------

KAFKA_TOPIC = "pulse_readings"
KAFKA_BOOTSTRAP = "localhost:9092"

DEVICE_IDS = ["pulse_001", "pulse_002", "pulse_003"]

# Risk scenario control
MODES = ["normal", "stressed", "critical"]
WEIGHTS = [0.5, 0.3, 0.2]   # controls dataset balance


# ------------------------
# Vital generation logic
# ------------------------

def generate_vitals(mode: str):
    if mode == "normal":
        return {
            "heart_rate": random.randint(60, 90),
            "spo2": random.randint(96, 100),
        }

    elif mode == "stressed":
        return {
            "heart_rate": random.randint(95, 120),
            "spo2": random.randint(90, 94),
        }

    elif mode == "critical":
        return {
            "heart_rate": random.randint(120, 150),
            "spo2": random.randint(80, 88),
        }


# ------------------------
# Main producer loop
# ------------------------

def main():
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    print("Starting scaled Pulse data stream...")

    try:
        while True:
            mode = random.choices(MODES, weights=WEIGHTS, k=1)[0]
            vitals = generate_vitals(mode)

            payload = {
                "device_id": random.choice(DEVICE_IDS),
                "user_id": "pulse_user",
                "heart_rate": vitals["heart_rate"],
                "spo2": vitals["spo2"],
                "timestamp": datetime.utcnow().isoformat(),
            }

            producer.send(KAFKA_TOPIC, value=payload)
            print(f"[{mode.upper()}] Sent → {payload}")

            time.sleep(1)   # 1 event per second

    except KeyboardInterrupt:
        print("\nStream stopped by user")

    finally:
        producer.close()


if __name__ == "__main__":
    main()

