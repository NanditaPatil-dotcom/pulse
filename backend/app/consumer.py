"""
Kafka consumer wrapper for Pulse.
Consumes JSON messages from topic `pulse_readings`
and schedules async callbacks for processing.
"""

import asyncio
import json
import os
from kafka import KafkaConsumer


KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC = os.environ.get("KAFKA_TOPIC", "pulse_readings")
GROUP_ID = os.environ.get("KAFKA_GROUP", "pulse_consumer_group")

def start_consumer(loop, callback):
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=GROUP_ID,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=True,
    )

    print("[Kafka] Consumer started. Listening to topic:", TOPIC)

    for msg in consumer:
        asyncio.run_coroutine_threadsafe(
            callback(msg.value),
            loop
        )

