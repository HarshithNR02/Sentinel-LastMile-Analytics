import json
import time
import joblib
import logging
import numpy as np
from datetime import datetime
from kafka import KafkaConsumer
from sqlalchemy import create_engine, text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler("consumer.log")]  
)
log = logging.getLogger("anomaly_consumer")

KAFKA_BROKER = "localhost:9092"
TOPIC = "deliveries"
MODEL_PATH = "../../models/delivery_isolation_forest.pkl"
MODEL_VERSION = "isoforest_v1_20260617"         
DB_URL = "postgresql+psycopg2://sentinel:sentinel@localhost:5433/sentinel"

iso = joblib.load(MODEL_PATH)
engine = create_engine(DB_URL)
log.info(f"Loaded model {MODEL_VERSION}, connected to Postgres")

IF_FEATURES = ['duration_robust_city','distance_robust_city','implied_speed_kmh',
               'accept_hour','is_instant_delivery','is_ghost_dispatch','aoi_type']

def check_rules(d):
    reasons = []
    if d['implied_speed_kmh'] > 100:       reasons.append('impossible_speed')
    if d['is_instant_delivery'] == 1:      reasons.append('instant_delivery')
    if d['is_ghost_dispatch'] == 1:        reasons.append('ghost_dispatch')
    if d['delivery_duration_min'] > 5000:  reasons.append('extreme_duration')
    return reasons

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=KAFKA_BROKER,
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    auto_offset_reset="earliest",
    group_id="anomaly-scorer-mlops",
)
log.info(f"Listening on '{TOPIC}'...")

total, anomalies, errors = 0, 0, 0
start_time = time.time()

for message in consumer:
    try:
        d = message.value
        total += 1

        t0 = time.time()
        X = np.array([[d[f] for f in IF_FEATURES]])
        if_anomaly = (iso.predict(X)[0] == -1)
        rule_reasons = check_rules(d)
        latency_ms = (time.time() - t0) * 1000

        is_anomaly = if_anomaly or len(rule_reasons) > 0
        if is_anomaly:
            anomalies += 1
            reasons = rule_reasons.copy()
            if if_anomaly:
                reasons.append('IF_flagged')
            reason_str = ', '.join(reasons)

            with engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO live_anomalies
                    (order_id, city, duration_min, distance_km, speed_kmh,
                     reasons, model_version, scoring_latency_ms)
                    VALUES (:oid, :city, :dur, :dist, :spd, :rsn, :ver, :lat)
                """), {
                    "oid": int(d['order_id']), "city": d['city'],
                    "dur": float(d['delivery_duration_min']),
                    "dist": float(d['distance_km']),
                    "spd": float(d['implied_speed_kmh']),
                    "rsn": reason_str, "ver": MODEL_VERSION, "lat": latency_ms
                })
                conn.commit()

            log.info(f"ANOMALY order={d['order_id']} {d['city']} "
                     f"speed={d['implied_speed_kmh']:.0f} reasons={reason_str} "
                     f"latency={latency_ms:.1f}ms")

        if total % 100 == 0:
            elapsed = time.time() - start_time
            rate = total / elapsed
            log.info(f"METRICS | processed={total} anomalies={anomalies} "
                     f"errors={errors} throughput={rate:.1f}/s")

    except Exception as e:

        errors += 1
        log.error(f"Failed to process message: {e}")
        continue