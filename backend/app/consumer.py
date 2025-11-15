"""A simple Kafka consumer wrapper that connects to topic 'pulse_readings'and calls a callback for each parsed JSON message. This uses kafka-pythonsynchronously but is wrapped in awaitable tasks for simplicity."""
import asyncio
import json
import os
from kafka import KafkaConsumer

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC = os.environ.get("KAFKA_TOPIC", "pulse_readings")
GROUP_ID = os.environ.get("KAFKA_GROUP", "pulse_consumer_group")

async def start_consumer(callback):
"""Start consumer in background. callback is async function taking dict."""
def run_sync_consumer(loop):
consumer = KafkaConsumer(TOPIC,
bootstrap_servers=KAFKA_BOOTSTRAP,
group_id=GROUP_ID,
value_deserializer=lambda m: json.loads(m.decode("utf-8")),
auto_offset_reset='earliest',
enable_auto_commit=True )
print("Kafka consumer started, listening to topic:", TOPIC)
for msg in consumer:
try:
loop.create_task(callback(msg.value))
except Exception as e:
print("Callback scheduling error:", e)
loop = asyncio.get_event_loop()
# run sync consumer in executor to not block event loop
await loop.run_in_executor(None, run_sync_consumer, loop)