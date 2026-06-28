import streamlit as st
import pandas as pd
import numpy as np
import sys, os
import lightgbm as lgb
import joblib
import pydeck as pdk
from datetime import datetime
from sqlalchemy import create_engine

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from rag.sentinel_rag import smart_answer

st.set_page_config(page_title="Sentinel", layout="wide", initial_sidebar_state="collapsed")

@st.cache_resource
def get_engine():
    return create_engine("postgresql+psycopg2://sentinel:sentinel@localhost:5433/sentinel")

@st.cache_resource
def load_pickup_model():
    base = os.path.join(os.path.dirname(__file__), "..", "..", "models")
    model = lgb.Booster(model_file=os.path.join(base, "pickup_disruption_lgbm.txt"))
    store = joblib.load(os.path.join(base, "feature_store.pkl"))
    return model, store

engine = get_engine()
pickup_model, feature_store = load_pickup_model()

MODEL_FEATURES = ['hour_of_day','courier_orders_so_far','day_of_week','accept_distance_km',
                  'courier_late_rate','aoi_disruption_rate','city_hour_rate','courier_city_rate',
                  'courier_zone_familiarity','courier_tenure_days','courier_load_3h',
                  'mins_since_last_accept','velocity_target','courier_running_rate',
                  'zone_running_rate','orders_per_courier','city']

CITY_COORDS = {
    "Shanghai": (31.2304, 121.4737), "Hangzhou": (30.2741, 120.1551),
    "Chongqing": (29.4316, 106.9123), "Jilin": (43.8378, 126.5497),
    "Yantai": (37.4638, 121.4479),
}

@st.cache_data(ttl=60)
def load_metrics():
    return pd.read_sql("""
        SELECT
            (SELECT COUNT(*) FROM pickups) AS total_pickups,
            (SELECT AVG(is_disrupted)*100 FROM pickups) AS disruption_rate,
            (SELECT COUNT(*) FROM deliveries) AS total_deliveries,
            (SELECT COUNT(DISTINCT order_id) FROM delivery_anomalies) AS total_anomalies,
            (SELECT COUNT(*) FROM courier_flags WHERE flag_status='review_operator') AS review_couriers
    """, engine).iloc[0]

@st.cache_data(ttl=60)
def load_city_data():
    anom = pd.read_sql("SELECT city, total_deliveries, anomaly_pct FROM city_summary", engine)
    disr = pd.read_sql("SELECT city, AVG(is_disrupted)*100 AS disruption_pct FROM pickups GROUP BY city", engine)
    return anom.merge(disr, on="city", how="outer")

@st.cache_data(ttl=60)
def load_feed_sorted():
    return pd.read_sql("""
        SELECT a.order_id, d.city, a.reason, d.ds, d.accept_hour,
               ROUND(d.delivery_duration_min::numeric,0) AS duration_min,
               ROUND(d.implied_speed_kmh::numeric,0) AS speed_kmh
        FROM delivery_anomalies a
        JOIN deliveries d ON a.order_id = d.order_id
        ORDER BY d.ds ASC, d.accept_hour ASC
        LIMIT 500
    """, engine).to_dict("records")

st.markdown("""
<style>
    .stApp { background-color: #0E1116; }
    .main .block-container { padding-top: 2rem; max-width: 1500px; }
    html, body, [class*="css"] { color: #E6EDF3; }
    .sentinel-header { font-family:'SF Mono','Menlo',monospace; font-size:1.5rem;
        font-weight:600; letter-spacing:0.15em; color:#E6EDF3;
        border-bottom:1px solid #21262D; padding-bottom:0.75rem; margin-bottom:0.25rem; }
    .sentinel-sub { font-family:'SF Mono','Menlo',monospace; font-size:0.75rem;
        color:#7D8590; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:1.5rem; }
    .metric-strip { display:flex; gap:2.5rem; margin:1rem 0 2rem 0; padding:1rem 1.25rem;
        background:#161B22; border:1px solid #21262D; border-radius:4px; flex-wrap:wrap; }
    .metric-item { display:flex; flex-direction:column; }
    .metric-value { font-family:'SF Mono','Menlo',monospace; font-size:1.5rem;
        font-weight:600; color:#E6EDF3; line-height:1.1; }
    .metric-label { font-family:'SF Mono','Menlo',monospace; font-size:0.68rem;
        color:#7D8590; text-transform:uppercase; letter-spacing:0.08em; margin-top:0.3rem; }
    .metric-accent { color:#3FB950; } .metric-alert { color:#F85149; }
    .panel-title { font-family:'SF Mono','Menlo',monospace; font-size:0.8rem; color:#7D8590;
        text-transform:uppercase; letter-spacing:0.1em; margin:1rem 0 0.75rem 0; }
    [data-testid="stChatInput"] textarea, .stChatInput textarea {
        background-color:#161B22 !important; color:#E6EDF3 !important; }
    [data-testid="stChatInput"] textarea::placeholder { color:#7D8590 !important; }
    [data-testid="stForm"] { background:#161B22 !important; border:1px solid #21262D !important; border-radius:6px; }
    .stSelectbox div[data-baseweb="select"] > div,
    .stNumberInput input, .stTextInput input {
        background-color:#0E1116 !important; color:#E6EDF3 !important; border-color:#30363D !important; }
    .stForm button { background:#161B22 !important; color:#3FB950 !important; border:1px solid #3FB950 !important; }
    label { color:#E6EDF3 !important; }
    #MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="sentinel-header">SENTINEL</div>', unsafe_allow_html=True)
st.markdown('<div class="sentinel-sub">Last-Mile Disruption & Anomaly Intelligence — 5 cities</div>', unsafe_allow_html=True)

m = load_metrics()
st.markdown(f"""
<div class="metric-strip">
    <div class="metric-item"><span class="metric-value">{int(m['total_pickups']):,}</span>
        <span class="metric-label">Pickups (Supervised)</span></div>
    <div class="metric-item"><span class="metric-value metric-alert">{m['disruption_rate']:.2f}%</span>
        <span class="metric-label">Disruption Rate</span></div>
    <div class="metric-item"><span class="metric-value">{int(m['total_deliveries']):,}</span>
        <span class="metric-label">Deliveries (Unsupervised)</span></div>
    <div class="metric-item"><span class="metric-value metric-alert">{int(m['total_anomalies']):,}</span>
        <span class="metric-label">Anomalies Detected</span></div>
    <div class="metric-item"><span class="metric-value metric-accent">{int(m['review_couriers'])}</span>
        <span class="metric-label">Couriers Flagged</span></div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="panel-title">Ask Sentinel — Hybrid AI (Text-to-SQL + RAG)</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.7rem;color:#7D8590;margin-bottom:0.5rem;">Try: "Which city has the highest anomaly rate?" · "What does ghost dispatch mean?"</div>', unsafe_allow_html=True)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if prompt := st.chat_input("Ask about pickups, anomalies, couriers..."):
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.spinner("Analyzing..."):
            result = smart_answer(prompt)
        st.session_state.messages.append({"role":"assistant","content":result["answer"],"route":result["route"]})
    chat_box = st.container(height=380)
    with chat_box:
        if not st.session_state.messages:
            st.markdown('<div style="color:#7D8590;font-size:0.85rem;padding:1rem;">Ask a question to begin.</div>', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div style="background:#1C2128;border:1px solid #30363D;border-radius:6px;'
                    f'padding:0.6rem 0.9rem;margin:0.4rem 0;color:#E6EDF3;font-size:0.9rem;">'
                    f'<span style="color:#7D8590;font-size:0.65rem;font-family:monospace;">YOU</span><br>{msg["content"]}</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="background:#161B22;border:1px solid #21262D;border-radius:6px;'
                    f'padding:0.6rem 0.9rem;margin:0.4rem 0 1rem 0;color:#E6EDF3;font-size:0.9rem;line-height:1.55;">'
                    f'<span style="color:#3FB950;font-size:0.65rem;font-family:monospace;letter-spacing:0.05em;">SENTINEL · {msg.get("route","")}</span><br>{msg["content"]}</div>',
                    unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel-title">Anomaly Map — 5 Cities</div>', unsafe_allow_html=True)
    city_df = load_city_data()
    city_df["lat"] = city_df["city"].map(lambda c: CITY_COORDS.get(c,(None,None))[0])
    city_df["lon"] = city_df["city"].map(lambda c: CITY_COORDS.get(c,(None,None))[1])
    city_df["anom_count"] = (city_df["total_deliveries"]*city_df["anomaly_pct"]/100).round().astype(int)
    layer = pdk.Layer("ScatterplotLayer", data=city_df, get_position=["lon","lat"],
        get_radius="anom_count * 8", get_fill_color=[248,81,73,140], pickable=True)
    view = pdk.ViewState(latitude=33, longitude=118, zoom=3.3, pitch=0)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view,
        map_style="mapbox://styles/mapbox/dark-v10",
        tooltip={"text":"{city}\nAnomaly rate: {anomaly_pct}%"}), use_container_width=True)

c1, c2 = st.columns([1, 1], gap="large")

with c1:
    st.markdown('<div class="panel-title">Pickup Risk Checker — Disruption Model</div>', unsafe_allow_html=True)
    with st.form("pickup_form"):
        fc1, fc2 = st.columns(2)
        with fc1:
            in_city = st.selectbox("City", list(CITY_COORDS.keys()))
            in_courier = st.number_input("Courier ID", value=164, step=1)
            in_aoi = st.number_input("Zone (AOI) ID", value=450, step=1)
        with fc2:
            in_hour = st.slider("Hour of day", 0, 23, 14)
            in_dow = st.slider("Day of week (0=Mon)", 0, 6, 2)
            in_dist = st.number_input("Distance (km)", value=5.0, step=0.5)
        submitted = st.form_submit_button("Score Pickup")
    if submitted:
        g = feature_store['global']
        courier = feature_store['courier'].get(int(in_courier), {})
        aoi_rate = feature_store['aoi'].get(int(in_aoi), g['aoi_disruption_rate'])
        ch_rate = feature_store['city_hour'].get(f"{in_city}_{in_hour}", g['city_hour_rate'])
        feat = {
            'hour_of_day': in_hour, 'courier_orders_so_far': courier.get('courier_orders_so_far',0),
            'day_of_week': in_dow, 'accept_distance_km': in_dist,
            'courier_late_rate': courier.get('courier_late_rate', g['courier_late_rate']),
            'aoi_disruption_rate': aoi_rate, 'city_hour_rate': ch_rate,
            'courier_city_rate': courier.get('courier_city_rate', g['courier_city_rate']),
            'courier_zone_familiarity': courier.get('courier_zone_familiarity',0),
            'courier_tenure_days': courier.get('courier_tenure_days',0),
            'courier_load_3h': courier.get('courier_load_3h',0),
            'mins_since_last_accept': courier.get('mins_since_last_accept',0),
            'velocity_target': courier.get('velocity_target',0),
            'courier_running_rate': courier.get('courier_running_rate', g['courier_running_rate']),
            'zone_running_rate': courier.get('zone_running_rate', g['zone_running_rate']),
            'orders_per_courier': courier.get('orders_per_courier',0), 'city': in_city,
        }
        X = pd.DataFrame([feat])[MODEL_FEATURES]
        city_cats = pd.CategoricalDtype(categories=["Shanghai","Hangzhou","Chongqing","Jilin","Yantai"], ordered=False)
        dow_cats  = pd.CategoricalDtype(categories=list(range(7)), ordered=False)
        hour_cats = pd.CategoricalDtype(categories=list(range(24)), ordered=False)
        X['city']        = X['city'].astype(city_cats)
        X['day_of_week'] = X['day_of_week'].astype(dow_cats)
        X['hour_of_day'] = X['hour_of_day'].astype(hour_cats)
        score = float(pickup_model.predict(X)[0])
        level = "HIGH" if score>=0.768 else "MEDIUM" if score>=0.646 else "LOW"
        color = "#F85149" if level=="HIGH" else "#D29922" if level=="MEDIUM" else "#3FB950"
        found = "feature store" if int(in_courier) in feature_store['courier'] else "global fallback"
        st.markdown(f'<div style="background:#161B22;border:1px solid #21262D;border-radius:6px;padding:1rem;">'
            f'<span style="color:#7D8590;font-size:0.7rem;font-family:monospace;">DISRUPTION RISK SCORE</span><br>'
            f'<span style="color:{color};font-size:2rem;font-family:monospace;font-weight:600;">{score:.3f}</span> '
            f'<span style="color:{color};font-family:monospace;">{level}</span><br>'
            f'<span style="color:#7D8590;font-size:0.65rem;font-family:monospace;">features: {found}</span></div>',
            unsafe_allow_html=True)

with c2:
    st.markdown('<div class="panel-title">Two-Model Insight — Pickup Disruption vs Delivery Anomaly</div>', unsafe_allow_html=True)
    city_df2 = load_city_data().sort_values("city")
    chart_df = city_df2[["city","disruption_pct","anomaly_pct"]].set_index("city")
    chart_df.columns = ["Pickup Disruption %", "Delivery Anomaly %"]
    st.bar_chart(chart_df, height=300, color=["#F85149","#3FB950"])
    st.markdown('<div style="font-size:0.7rem;color:#7D8590;">Inversion: Shanghai high disruption / low anomaly; Jilin the reverse.</div>', unsafe_allow_html=True)

st.markdown('<div class="panel-title">Live Anomaly Feed — Dataset Playback</div>', unsafe_allow_html=True)

if "feed_data" not in st.session_state:
    raw = load_feed_sorted() 
    for i, rec in enumerate(raw):
        rec["disp_min"] = i % 60
    st.session_state.feed_data = raw
    st.session_state.feed_pos = 12

stream_on = st.toggle("Stream live", value=False)

def fmt_time(row):
    month = int(row["ds"]) // 100
    day = int(row["ds"]) % 100
    return f"2022-{month:02d}-{day:02d} {int(row['accept_hour']):02d}:{row['disp_min']:02d}"

def render_feed():
    pos = st.session_state.feed_pos
    visible = st.session_state.feed_data[max(0, pos-12):pos][::-1]
    feed_html = '<div style="height:300px;overflow-y:auto;background:#0E1116;border:1px solid #21262D;border-radius:6px;padding:0.5rem 0.75rem;">'
    for row in visible:
        feed_html += (f'<div style="font-family:monospace;font-size:0.8rem;color:#E6EDF3;'
            f'border-bottom:1px solid #21262D;padding:0.35rem 0;">'
            f'<span style="color:#6E7681;">{fmt_time(row)}</span>  '
            f'<span style="color:#F85149;">●</span> '
            f'<span style="color:#7D8590;">order={row["order_id"]}</span> '
            f'<span style="color:#E6EDF3;">{row["city"]}</span> · '
            f'<span style="color:#D29922;">{row["reason"]}</span> · '
            f'{row["duration_min"]:.0f}min · {row["speed_kmh"]:.0f}km/h</div>')
    feed_html += '</div>'
    st.markdown(feed_html, unsafe_allow_html=True)

if stream_on:
    @st.fragment(run_every=1)
    def animated_feed():
        st.session_state.feed_pos += 1
        if st.session_state.feed_pos >= len(st.session_state.feed_data):
            st.session_state.feed_pos = 12
        render_feed()
    animated_feed()
else:
    render_feed()