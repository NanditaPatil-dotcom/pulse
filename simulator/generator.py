import random
import time
import json
from datetime import datetime


def generate_data(device_id: str):
    heart_rate = random.randint(60, 100)
    spo2 = random.randint(95, 100)
    timestamp = datetime.now().isoformat()
    return {
        "device_id": device_id,
        "heart_rate": heart_rate,
        "spo2": spo2,
        "timestamp": timestamp,
    }


def stream_data(device_ids):
    while True:
        for device_id in device_ids:
            yield generate_data(device_id)
        time.sleep(5)
