import pandas as pd
import json
import time
from kafka import KafkaProducer

KAFKA_BROKER = "localhost:9092"
TOPIC = "deliveries"
DATA_PATH = "../../data/processed/delivery_features.parquet" 

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"), 
)
print(f"Connected to Kafka at {KAFKA_BROKER}, topic '{TOPIC}'")

df = pd.read_parquet(DATA_PATH)
print(f"Loaded {len(df):,} deliveries to stream")

SEND_COLS = ['order_id','city','courier_id','delivery_duration_min','distance_km',
             'duration_robust_city','distance_robust_city','implied_speed_kmh',
             'accept_hour','is_instant_delivery','is_ghost_dispatch','aoi_type']

sample = df[SEND_COLS].head(1000)  

print("Streaming deliveries... (Ctrl+C to stop)")
for i, row in enumerate(sample.to_dict(orient="records")):
    producer.send(TOPIC, value=row)
    if i % 100 == 0:
        print(f"  Sent {i} deliveries...")
    time.sleep(0.05)  

producer.flush()  
print(f"Done. Sent {len(sample)} deliveries to '{TOPIC}'")