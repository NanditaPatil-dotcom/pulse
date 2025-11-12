from kafka import KafkaProducer
import json
import time

# Initialize producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Send dummy sensor data
data = {"device_id": "pulse01", "temp": 26.4, "humidity": 58}
producer.send('sensor_data', value=data)
producer.flush()

print("Sent data to Kafka:", data)
time.sleep(1)

