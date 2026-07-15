import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="AQI Forecaster India",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw')

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0A0F1E; color: #F0F4F8; }
.stApp { background-color: #0A0F1E; }
[data-testid="stSidebarNav"] { display: none !important; }
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] { transform: none !important; width: 260px !important; min-width: 260px !important; background-color: #111827; border-right: 1px solid #1E2D40; }
[data-testid="stMetric"] { background-color: #111827; border: 1px solid #1E2D40; border-radius: 12px; padding: 20px; }
[data-testid="stMetricLabel"] { color: #8899AA; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetricValue"] { color: #00D4AA; font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 700; }
h1, h2, h3 { color: #F0F4F8; font-weight: 700; }
hr { border-color: #1E2D40; }
.stat-card { background-color: #111827; border: 1px solid #1E2D40; border-radius: 12px; padding: 24px; margin-bottom: 16px; }
.stat-card h4 { color: #8899AA; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; margin: 0 0 8px 0; }
.stat-card p { color: #00D4AA; font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; font-weight: 700; margin: 0; }
.info-box { background-color: #0D1F35; border-left: 3px solid #00D4AA; border-radius: 0 8px 8px 0; padding: 16px 20px; margin: 16px 0; }
.page-header { background: linear-gradient(135deg, #111827 0%, #0D1F35 100%); border: 1px solid #1E2D40; border-radius: 12px; padding: 28px 32px; margin-bottom: 28px; }
.page-header h1 { margin: 0 0 4px 0; font-size: 1.8rem; }
.page-header p { color: #8899AA; margin: 0; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)
# ── Sidebar Navigation ─────────────────────────────────────────────────────────
st.sidebar.markdown("## 🌫️ AQI Forecaster")
st.sidebar.markdown("---")
st.sidebar.markdown("[🏠 Home](https://aqi-forecaster-y9czaqshoqdwbgapp9dxyre.streamlit.app)")
st.sidebar.markdown("[🏙️ City Analysis](https://aqi-forecaster-y9czaqshoqdwbgapp9dxyre.streamlit.app/City_Analysis)")
st.sidebar.markdown("[📈 Forecast](https://aqi-forecaster-y9czaqshoqdwbgapp9dxyre.streamlit.app/Forecast)")
st.sidebar.markdown("[📍 Station Analysis](https://aqi-forecaster-y9czaqshoqdwbgapp9dxyre.streamlit.app/Station_Analysis)")
st.sidebar.markdown("[🤖 Model Insights](https://aqi-forecaster-y9czaqshoqdwbgapp9dxyre.streamlit.app/Model_Insigths)")
# ── Load Data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    city_day = pd.read_csv(os.path.join(DATA_RAW, 'city_day.csv'))
    city_day['Date'] = pd.to_datetime(city_day['Date'])
    return city_day

df = load_data()

# ── Hero Section ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>🌫️ AQI Forecaster — India</h1>
    <p>Air quality monitoring and forecasting across 26 major Indian cities · 2015–2020</p>
</div>
""", unsafe_allow_html=True)

# ── Top Metrics ────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Cities Monitored", "26")
with col2: st.metric("Total Records", "1,37,000+")
with col3: st.metric("Model MAE", "15.74")
with col4: st.metric("Forecast Horizon", "7 Days")

st.markdown("---")

# ── Most Polluted Cities ───────────────────────────────────────────────────────
st.markdown("### Most Polluted Cities")
latest = df.dropna(subset=['AQI']).sort_values('Date').groupby('City').last().reset_index()
top5   = latest.nlargest(5, 'AQI')[['City', 'AQI', 'AQI_Bucket']]

cols = st.columns(5)
for i, (_, row) in enumerate(top5.iterrows()):
    with cols[i]:
        st.markdown(f"""
        <div class="stat-card">
            <h4>{row['City']}</h4>
            <p>{row['AQI']:.0f}</p>
            <span style="color:#8899AA; font-size:0.75rem;">{row['AQI_Bucket']}</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── AQI Scale Reference ────────────────────────────────────────────────────────
st.markdown("### AQI Scale Reference")

scale_cols = st.columns(6)
scales = [
    ("Good",        "0–50",    "#00C853", "#000"),
    ("Satisfactory","51–100",  "#64DD17", "#000"),
    ("Moderate",    "101–200", "#FFD600", "#000"),
    ("Poor",        "201–300", "#FF6D00", "#fff"),
    ("Very Poor",   "301–400", "#DD2C00", "#fff"),
    ("Severe",      "401+",    "#6200EA", "#fff"),
]

for col, (label, rang, bg, fg) in zip(scale_cols, scales):
    with col:
        st.markdown(f"""
        <div style="background:{bg}; border-radius:10px; padding:16px; text-align:center;">
            <p style="color:{fg}; font-weight:700; margin:0; font-size:0.85rem;">{label}</p>
            <p style="color:{fg}; margin:0; font-size:0.75rem; opacity:0.85;">{rang}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── Info Box ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="info-box">
    <strong style="color:#00D4AA;">Navigate using the sidebar</strong>
    <p style="color:#8899AA; margin:4px 0 0 0; font-size:0.85rem;">
    Use City Analysis to explore trends · Forecast for 7-day predictions ·
    Station Analysis for location-level data · Model Insights for ML details
    </p>
</div>
""", unsafe_allow_html=True)
