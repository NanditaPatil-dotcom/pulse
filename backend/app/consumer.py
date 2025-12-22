# app/consumer.py

import json
import os
from kafka import KafkaConsumer

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC = os.environ.get("KAFKA_TOPIC", "pulse_readings")
GROUP_ID = os.environ.get("KAFKA_GROUP", "pulse_consumer_group")


def start_consumer(loop, queue):
    """
    Kafka runs in a background thread.
    It pushes messages safely into the asyncio queue owned by FastAPI loop.
    """
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=GROUP_ID,
        auto_offset_reset="latest",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )

    print(f"[Kafka] Consumer started | topic={TOPIC}")

    for msg in consumer:
        # thread-safe push into async loop
        loop.call_soon_threadsafe(queue.put_nowait, msg.value)





