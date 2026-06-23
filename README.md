cat > README.md << 'SENTINEL_README_EOF'
# Sentinel — Last-Mile Delivery Disruption Intelligence

End-to-end ML platform on **6M+ real deliveries** across 5 Chinese cities. Two models run in parallel — a supervised disruption classifier and an unsupervised anomaly detector — wrapped in a full production-style stack: streaming, serving, monitoring, and a hybrid AI assistant.

🔗 **Live API:** http://3.92.4.253:8000/docs
🔗 **Live Dashboard:** http://3.92.4.253:8501

> Dataset: [Cainiao LaDe](https://huggingface.co/datasets/Cainiao-AI/LaDe) (Alibaba, 2022) — 6.1M pickups + 4.5M deliveries across Shanghai, Hangzhou, Chongqing, Yantai, Jilin.

---

## Key Findings

- **30 couriers → 13,258 anomalies.** 0.6% of the network produced ~20% of all flagged events. Concentrated anomaly detection into a 30-courier review list — actionable for operators.
- **City inversion across the two models.** Shanghai ranked *highest* for pickup disruption (2.45%) and *lowest* for delivery anomalies (0.88%). Jilin ranked the opposite (0.20% / 3.29%). Two failure modes: dense cities fail on *time*, small cities fail on *telemetry*.
- **Caught data leakage during feature engineering.** A rolling "recent disruption rate" feature pushed PR-AUC from 0.22 to 0.29. Investigated, found it was leaking test-set outcomes between concurrent rows (couriers with no training history showed roll5 values of 1.0). Dropped the feature, kept the honest 0.22.

---

## Architecture

```text
                       +----------------------------------------+
                       |        AWS Cloud (us-east-1)           |
                       |                                        |
   Browser --HTTP-->   |  EC2 (t3.micro, Ubuntu)                |
                       |   |-- FastAPI  :8000  (model serving)  |
                       |   |-- Streamlit :8501 (dashboard)      |
                       |   +-- PostgreSQL :5432 (1.6GB restored)|
                       |                ^                       |
                       |                | boto3                 |
                       |                v                       |
                       |  S3  (sentinel-models-harshith)        |
                       |   |-- pickup_disruption_lgbm.txt       |
                       |   |-- feature_store.pkl                |
                       |   +-- delivery_isolation_forest.pkl    |
                       |                                        |
                       |  IAM Role: sentinel-ec2-s3-role        |
                       +----------------------------------------+
```

---

## What's Inside

| Component | Tech | What it does |
|---|---|---|
| **Pickup Disruption Model** | LightGBM, Optuna | Supervised. Predicts if a pickup will miss its time window. PR-AUC 0.22 at 1.82% base rate (12x lift). |
| **Delivery Anomaly Detection** | Isolation Forest + rules | Unsupervised. Flags abnormal deliveries (ghost dispatches, impossible speeds, extreme durations). 66K anomalies surfaced from 4.5M deliveries. |
| **REST API** | FastAPI, Pydantic, boto3 | Model-serving API with a feature-store lookup. Validated input, typed responses, structured logging, `/health` endpoint. Deployed on AWS EC2 with models in S3. |
| **Streaming** | Kafka, Postgres | Producer streams deliveries to a Kafka topic; consumer scores them with the anomaly model and persists predictions with latency tracking. |
| **SQL Analytics** | PostgreSQL | Window functions, CTEs, indexing (29x speedup verified), partitioning by month, SCD Type 2 for couriers. 5 normalized tables. |
| **Hybrid AI Assistant** | OpenAI, LangChain, FAISS, Text-to-SQL | Routes natural-language questions: SQL queries to live Postgres; conceptual questions to FAISS RAG over a curated knowledge base. |
| **A/B Testing** | scipy, statsmodels | Simulated randomized controlled trial. Two-proportion z-test + sensitivity analysis across intervention strengths. |
| **Drift Monitoring** | Evidently | Compares training-period vs current-period feature distributions. Wasserstein distance. Distinguishes expected drift (tenure accumulation) from operational drift. |
| **Dashboard** | Streamlit, pydeck, Mapbox | Live operational view: metrics, city anomaly map, AI chat, risk checker, two-model comparison, anomaly feed. |

---

## Tech Stack

**Languages:** Python, SQL
**ML / Modeling:** LightGBM, XGBoost, scikit-learn, SHAP, Optuna, Isolation Forest
**LLM / AI:** OpenAI API, LangChain, FAISS, RAG, Text-to-SQL
**MLOps:** MLflow, Docker, FastAPI, Evidently
**Data:** PostgreSQL, Kafka, Parquet
**Cloud:** AWS (EC2, S3, IAM)
**Apps:** Streamlit, Pydeck

---

## Running Locally

```bash
git clone https://github.com/HarshithNR02/Sentinel-LastMile-Analytics.git
cd Sentinel-LastMile-Analytics

pip install -r requirements.txt

# Create a .env file with: OPENAI_API_KEY=your-key-here

docker-compose up -d

# API
cd api && uvicorn main:app --reload --port 8000

# Dashboard (separate terminal)
cd dashboard && streamlit run app.py
```

Models are excluded from the repo (too large) — they're hosted in S3 and downloaded by the API on startup. To train from scratch, run the notebooks in order (01a -> 12).

---

## Project Structure

```text
sentinel/
|-- api/                  # FastAPI model-serving
|   +-- main.py
|-- consumer/             # Kafka consumer (scores deliveries live)
|   +-- delivery_consumer.py
|-- producer/             # Kafka producer (streams historical deliveries)
|   +-- delivery_producer.py
|-- dashboard/            # Streamlit operational dashboard
|   +-- app.py
|-- rag/                  # Hybrid Text-to-SQL + FAISS RAG
|   |-- sentinel_rag.py
|   +-- faiss_index/
|-- notebooks/            # 13 notebooks: EDA -> FE -> modeling -> SQL -> RAG -> A/B -> drift
|-- requirements.txt
|-- docker-compose.yml
+-- .gitignore
```

---

## Honest Notes

- **PR-AUC of 0.22 is the empirical ceiling** for this 1.82% base-rate behavioral prediction problem. 8 candidate features were tested across two rounds; one looked promising but was leaking (caught and rejected). The honest number stands.
- **The A/B test is a simulation** demonstrating the methodology (randomization, hypothesis testing, sensitivity analysis) on historical data. A real deployment would run as a live RCT.
- **The dashboard's streaming feed** plays anomalies in real dataset time (2022-05 onward); the underlying real-time pipeline is the Kafka consumer.
- **Portfolio-grade, not production-grade.** Scoped out: auth, rate limiting, async scaling, HTTPS. Easy to add for real traffic; deliberately omitted for a demo.

---

## Author

**Harshith Nerlikere Ramesh** — MS Data Science, UMass Dartmouth (Aug 2026)
Open to Data Scientist / ML Engineer / Data Analyst roles.

[LinkedIn](https://www.linkedin.com/in/harshithnr/) · [GitHub](https://github.com/HarshithNR02)
SENTINEL_README_EOF