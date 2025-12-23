import json
import os
import time
from kafka import KafkaConsumer

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "pulse_readings")
GROUP_ID = os.getenv("KAFKA_GROUP", "pulse_consumer_group")


def start_consumer(loop, queue):
    """
    Kafka runs in a background thread.
    It retries until Kafka is reachable,
    then continuously consumes messages.
    """
    consumer = None

    print("[Kafka] Starting consumer...")


    while consumer is None:
        try:
            consumer = KafkaConsumer(
                TOPIC,
                bootstrap_servers=KAFKA_BOOTSTRAP,
                group_id=GROUP_ID,
                auto_offset_reset="latest",
                enable_auto_commit=True,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            )
            print(f"[Kafka] Connected to {KAFKA_BOOTSTRAP}")
        except Exception as e:
            print("[Kafka] Broker not ready, retrying in 3s...", e)
            time.sleep(3)


    for msg in consumer:
        loop.call_soon_threadsafe(queue.put_nowait, msg.value)






