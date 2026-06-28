from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import lightgbm as lgb
import joblib
import pandas as pd
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("sentinel-api")

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup
    try:
        ml_models["model"] = lgb.Booster(model_file="../../models/pickup_disruption_lgbm.txt")
        ml_models["store"] = joblib.load("../../models/feature_store.pkl")
        logger.info("Model and feature store loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")
        raise
    yield
    ml_models.clear()
    logger.info("Shutdown: cleared model artifacts.")

app = FastAPI(
    title="Sentinel Pickup Disruption API",
    description="Predicts last-mile pickup disruption risk using a LightGBM model "
                "with a feature-store lookup. Returns a relative risk score and tier.",
    version="1.0.0",
    lifespan=lifespan,
)

MODEL_FEATURES = ['hour_of_day','courier_orders_so_far','day_of_week','accept_distance_km',
                  'courier_late_rate','aoi_disruption_rate','city_hour_rate','courier_city_rate',
                  'courier_zone_familiarity','courier_tenure_days','courier_load_3h',
                  'mins_since_last_accept','velocity_target','courier_running_rate',
                  'zone_running_rate','orders_per_courier','city']

VALID_CITIES = {"Shanghai", "Hangzhou", "Chongqing", "Jilin", "Yantai"}

class PickupRequest(BaseModel):
    courier_id: int = Field(..., ge=0, description="Courier ID")
    city: str = Field(..., description="City name (Shanghai, Hangzhou, Chongqing, Jilin, Yantai)")
    aoi_id: int = Field(..., ge=0, description="Area/zone ID")
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour 0-23")
    day_of_week: int = Field(..., ge=0, le=6, description="Day 0=Mon..6=Sun")
    accept_distance_km: float = Field(..., ge=0, le=200, description="Distance in km")

class PredictionResponse(BaseModel):
    courier_id: int
    disruption_risk: float = Field(..., description="Relative risk score (not a calibrated probability)")
    risk_level: str = Field(..., description="high / medium / low (percentile-based)")
    features_source: str = Field(..., description="feature_store or global_fallback")

def build_features(req: PickupRequest, store: dict):
    g = store['global']
    courier = store['courier'].get(req.courier_id, {})
    aoi_rate = store['aoi'].get(req.aoi_id, g['aoi_disruption_rate'])
    ch_key = f"{req.city}_{req.hour_of_day}"
    ch_rate = store['city_hour'].get(ch_key, g['city_hour_rate'])

    return {
        'hour_of_day': req.hour_of_day,
        'courier_orders_so_far': courier.get('courier_orders_so_far', 0),
        'day_of_week': req.day_of_week,
        'accept_distance_km': req.accept_distance_km,
        'courier_late_rate': courier.get('courier_late_rate', g['courier_late_rate']),
        'aoi_disruption_rate': aoi_rate,
        'city_hour_rate': ch_rate,
        'courier_city_rate': courier.get('courier_city_rate', g['courier_city_rate']),
        'courier_zone_familiarity': courier.get('courier_zone_familiarity', 0),
        'courier_tenure_days': courier.get('courier_tenure_days', 0),
        'courier_load_3h': courier.get('courier_load_3h', 0),
        'mins_since_last_accept': courier.get('mins_since_last_accept', 0),
        'velocity_target': courier.get('velocity_target', 0),
        'courier_running_rate': courier.get('courier_running_rate', g['courier_running_rate']),
        'zone_running_rate': courier.get('zone_running_rate', g['zone_running_rate']),
        'orders_per_courier': courier.get('orders_per_courier', 0),
        'city': req.city,
    }

@app.get("/")
def root():
    return {
        "service": "Sentinel Pickup Disruption API",
        "version": "1.0.0",
        "model": "pickup_disruption_lgbm",
        "docs": "/docs",
    }

@app.get("/health")
def health():
    model_ok = "model" in ml_models and "store" in ml_models
    if not model_ok:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PickupRequest):
    start = time.time()

    if req.city not in VALID_CITIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown city '{req.city}'. Valid cities: {sorted(VALID_CITIES)}"
        )

    try:
        store = ml_models["store"]
        model = ml_models["model"]
        feat = build_features(req, store)
        X = pd.DataFrame([feat])[MODEL_FEATURES]
        city_cats = pd.CategoricalDtype(categories=["Shanghai","Hangzhou","Chongqing","Jilin","Yantai"], ordered=False)
        dow_cats  = pd.CategoricalDtype(categories=list(range(7)), ordered=False)
        hour_cats = pd.CategoricalDtype(categories=list(range(24)), ordered=False)
        X['city']        = X['city'].astype(city_cats)
        X['day_of_week'] = X['day_of_week'].astype(dow_cats)
        X['hour_of_day'] = X['hour_of_day'].astype(hour_cats)

        risk = float(model.predict(X)[0])
        level = "high" if risk >= 0.768 else "medium" if risk >= 0.646 else "low"
        source = "feature_store" if req.courier_id in store['courier'] else "global_fallback"

        latency_ms = (time.time() - start) * 1000
        logger.info(
            f"predict courier={req.courier_id} city={req.city} "
            f"risk={risk:.4f} level={level} source={source} latency={latency_ms:.1f}ms"
        )

        return PredictionResponse(
            courier_id=req.courier_id,
            disruption_risk=round(risk, 4),
            risk_level=level,
            features_source=source,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Prediction failed for courier={req.courier_id}")
        raise HTTPException(status_code=500, detail="Internal prediction error")