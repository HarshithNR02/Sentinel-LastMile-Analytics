"""
Sentinel AI Insights — hybrid Text-to-SQL + FAISS RAG system.
Routes numeric questions to Text-to-SQL (live Postgres), conceptual questions
to FAISS-based retrieval, and off-topic questions to a polite redirect.
"""
import os
import json
import numpy as np
import pandas as pd
import faiss
from openai import OpenAI
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
client = OpenAI()
engine = create_engine("postgresql+psycopg2://sentinel:sentinel@localhost:5433/sentinel")

_INDEX_DIR = os.path.join(os.path.dirname(__file__), "faiss_index")
faiss_index = faiss.read_index(os.path.join(_INDEX_DIR, "sentinel.index"))
with open(os.path.join(_INDEX_DIR, "knowledge_base.json")) as f:
    knowledge_base = json.load(f)

SCHEMA = """
Tables in the Sentinel logistics database:
1. deliveries (4.5M rows): order_id (PK), city, courier_id, ds, accept_hour,
   delivery_duration_min, distance_km, implied_speed_kmh, aoi_type
2. delivery_anomalies (78,957 rows, ONE ROW PER REASON): anomaly_id (PK),
   order_id (FK to deliveries), reason, anomaly_score
   reasons: IF_subtle, impossible_speed, instant_delivery, ghost_dispatch, extreme_duration, IF_confirmed
3. couriers (4,872 rows): courier_id (PK), city, total_deliveries, avg_duration, anomaly_count, anomaly_rate
4. courier_flags (4,872 rows): courier_id, city, total_deliveries, anomaly_count, anomaly_pct, flag_status
5. pickups (6M rows): order_id (PK), city, courier_id, ds, is_disrupted, accept_distance_km
View: city_summary (city, total_deliveries, anomalous_deliveries, anomaly_pct, avg_duration, avg_distance)
Cities: Shanghai, Hangzhou, Chongqing, Yantai, Jilin
"""

SCHEMA_RULES = """
BUSINESS RULES (follow strictly to write correct SQL):

RATES AND RANKINGS:
- Any rate (anomaly_rate, disruption_rate, anomaly_pct) for ranking must filter for volume:
  WHERE total_deliveries >= 100 (or HAVING COUNT(*) >= 100). A 1-delivery courier with 1 anomaly
  has a meaningless 100% rate. Never rank by rate without a volume floor.
- An anomaly rate = (distinct anomalous deliveries) / (TOTAL deliveries including non-anomalous).
  Always LEFT JOIN from deliveries to delivery_anomalies so non-anomalous deliveries count in the
  denominator. NEVER compute anomalies/anomalies (that wrongly gives 100%).

PREFER PRE-BUILT TABLES/VIEWS (already correct):
- City-level anomaly rate: use city_summary view (SELECT city, anomaly_pct FROM city_summary ORDER BY anomaly_pct DESC).
- Courier counts/rates: use couriers table (total_deliveries, anomaly_count, anomaly_rate).
- Couriers needing review: use courier_flags (flag_status = 'review_operator'/'monitor'/'normal').

COUNTING ANOMALIES:
- delivery_anomalies has ONE ROW PER REASON. For unique anomalous deliveries use COUNT(DISTINCT order_id).
- "How many anomalies" usually means unique deliveries -> COUNT(DISTINCT order_id).
- "How many of each reason" -> GROUP BY reason, COUNT(*).

PICKUPS vs DELIVERIES (two separate datasets, DO NOT mix):
- pickups: supervised disruption (is_disrupted, 1=missed deadline). Disruption rate = AVG(is_disrupted). Base ~1.82%.
- deliveries: unsupervised anomalies (via delivery_anomalies). DIFFERENT orders.
- NEVER JOIN pickups to deliveries on order_id; they are different events.

DATES:
- ds is an integer MMDD (604 = June 4, 1028 = Oct 28). Month = ds/100, day = ds%100. Range 501-1031 (May-Oct 2022).
- "by month" -> (ds / 100). "by day of month" -> (ds % 100).

CITIES: Only Shanghai, Hangzhou, Chongqing, Yantai, Jilin exist. Use ILIKE for case-insensitive matching.
If a user names another city, there is no data for it.

GENERAL:
- If ambiguous, pick the most common reasonable interpretation.
- LIMIT large result sets to 20 rows unless the user asks for all or a specific number.
"""

def embed(texts):
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([d.embedding for d in resp.data], dtype="float32")

def text_to_sql(question):
    prompt = f"""You are a PostgreSQL expert. Write a SQL query to answer the question.
{SCHEMA}
{SCHEMA_RULES}
Return ONLY the SQL, no markdown fences.
Question: {question}
SQL:"""
    sql = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0
    ).choices[0].message.content.strip().replace("```sql", "").replace("```", "").strip()
    try:
        result = pd.read_sql(sql, engine)
        return {"sql": sql, "result": result}
    except Exception as e:
        return {"error": f"SQL failed: {e}", "sql": sql}

def ask_sql(question):
    r = text_to_sql(question)
    if "error" in r:
        return "I couldn't compute that from the data. Try rephrasing the question."
    if r["result"].empty:
        return "The query ran but returned no matching data. There may be no records for that condition."
    prompt = f"""The user asked: "{question}"
SQL run: {r['sql']}
Result:
{r['result'].to_string(index=False)}
Answer in 2-3 sentences using ONLY these numbers. Do not invent information."""
    return client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0.3
    ).choices[0].message.content.strip()

def retrieve(query, k=3):
    DISTANCE_THRESHOLD = 1.2
    distances, indices = faiss_index.search(embed([query]), k)
    return [knowledge_base[i] for i, d in zip(indices[0], distances[0]) if d < DISTANCE_THRESHOLD]

def ask_rag(question, k=3):
    context = "\n\n".join(retrieve(question, k))
    prompt = f"""Answer using ONLY the context below. If it doesn't contain the answer,
say "I don't have information about that in my knowledge base." Do not make up information.
Context:
{context}
Question: {question}
Answer:"""
    return client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0.3
    ).choices[0].message.content.strip()

def smart_answer(question):
    classify = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0, max_tokens=10,
        messages=[{"role": "user", "content": f"""Classify this question about the Sentinel
logistics anomaly system into exactly one category:
SQL = specific numbers, counts, rankings, comparisons, data lookups.
CONCEPTUAL = explanations, definitions, methodology, why/how about the system.
OFFTOPIC = unrelated to this logistics/anomaly project, or impossible to answer from the data.
Question: {question}
Reply with ONLY: SQL, CONCEPTUAL, or OFFTOPIC."""}]
    ).choices[0].message.content.strip().upper()

    if "OFFTOPIC" in classify:
        return {"route": "OFFTOPIC",
                "answer": "I can only answer questions about the Sentinel logistics anomaly "
                          "detection system — its data, models, anomalies, couriers, and methodology. "
                          "Try asking about anomaly counts, courier patterns, or how the system works."}
    if "SQL" in classify:
        return {"route": "SQL", "answer": ask_sql(question)}
    return {"route": "CONCEPTUAL", "answer": ask_rag(question)}


if __name__ == "__main__":
    tests = [
        "How many anomalies in shanghai?",
        "What about Beijing?",
        "Anomalies by month",
        "What's the weather today?",
        "What does ghost dispatch mean?",
        "Which city has the highest anomaly rate?",
    ]
    for q in tests:
        r = smart_answer(q)
        print(f"\nQ: {q}\n[{r['route']}] {r['answer']}")