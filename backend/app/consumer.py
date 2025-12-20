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


async def start_consumer(callback):
    """
    Start Kafka consumer in background.
    `callback` must be an async function that accepts a dict.
    """

    loop = asyncio.get_running_loop()

    def run_sync_consumer():
        consumer = KafkaConsumer(
            TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            group_id=GROUP_ID,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=True,
        )

        print(f"[Kafka] Consumer started. Listening to topic: {TOPIC}")

        for msg in consumer:
            data = msg.value
            try:
                # Schedule async processing safely
                asyncio.run_coroutine_threadsafe(callback(data), loop)
            except Exception as e:
                print("[Kafka] Callback scheduling error:", e)

    # Run Kafka consumer in background thread
    await loop.run_in_executor(None, run_sync_consumer)
