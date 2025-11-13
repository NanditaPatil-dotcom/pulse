from kafka import KafkaProducer
import json
from generator import stream_data


def main():
    producer = KafkaProducer(
        bootstrap_servers="localhost:9092",
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    device_ids = ["pulse_001", "pulse_002", "pulse_003"]
    print("Starting data stream to Kafka topic: 'pulse_readings'")
    try:
        for data in stream_data(device_ids):
            producer.send("pulse_readings", value=data)
            print(f"Sent: {data}")
    except KeyboardInterrupt:
        print("\nStream stopped by user.")
    finally:
        producer.close()


if __name__ == "__main__":
    main()
