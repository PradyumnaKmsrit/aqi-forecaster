import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
from datetime import datetime, timedelta

st.set_page_config(page_title="AQI Forecast", page_icon="📈", layout="wide")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0A0F1E; color: #F0F4F8; }
.stApp { background-color: #0A0F1E; }
[data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1E2D40; }
[data-testid="stMetric"] { background-color: #111827; border: 1px solid #1E2D40; border-radius: 12px; padding: 20px; }
[data-testid="stMetricLabel"] { color: #8899AA; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetricValue"] { color: #00D4AA; font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 700; }
h1, h2, h3 { color: #F0F4F8; font-weight: 700; }
hr { border-color: #1E2D40; }
.page-header { background: linear-gradient(135deg, #111827 0%, #0D1F35 100%); border: 1px solid #1E2D40; border-radius: 12px; padding: 28px 32px; margin-bottom: 28px; }
.page-header h1 { margin: 0 0 4px 0; font-size: 1.8rem; }
.page-header p { color: #8899AA; margin: 0; font-size: 0.9rem; }
.forecast-card { background-color: #111827; border: 1px solid #1E2D40; border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 8px; }
.forecast-card .day { color: #8899AA; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; margin: 0 0 8px 0; }
.forecast-card .aqi-val { font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 700; margin: 0 0 8px 0; }
.forecast-card .bucket { font-size: 0.75rem; font-weight: 600; padding: 3px 10px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Load Data and Model ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(DATA_PROCESSED, 'city_day_processed.csv'))
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_resource
def load_model():
    model    = joblib.load(os.path.join(MODELS_DIR, 'xgb_aqi_model.pkl'))
    features = joblib.load(os.path.join(MODELS_DIR, 'feature_names.pkl'))
    return model, features

df              = load_data()
model, features = load_model()

# ── Helpers ────────────────────────────────────────────────────────────────────
def get_bucket(aqi):
    if aqi <= 50:    return 'Good',        '#00C853', '#000'
    elif aqi <= 100: return 'Satisfactory', '#64DD17', '#000'
    elif aqi <= 200: return 'Moderate',    '#FFD600', '#000'
    elif aqi <= 300: return 'Poor',        '#FF6D00', '#fff'
    elif aqi <= 400: return 'Very Poor',   '#DD2C00', '#fff'
    else:            return 'Severe',      '#6200EA', '#fff'

def get_season(month):
    if month in [12,1,2]:    return 0
    elif month in [3,4,5]:   return 1
    elif month in [6,7,8,9]: return 2
    else:                     return 3

# ── Page Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>📈 7-Day AQI Forecast</h1>
    <p>XGBoost-powered next 7-day air quality predictions for any Indian city</p>
</div>
""", unsafe_allow_html=True)

# ── City Selector ──────────────────────────────────────────────────────────────
cities = sorted(df['City'].unique().tolist())
selected_city = st.selectbox("Select City", cities, index=cities.index('Delhi'))

city_df  = df[df['City'] == selected_city].copy().sort_values('Date')
last_row = city_df.iloc[-1]

st.markdown("---")

# ── Generate 7-Day Forecast ────────────────────────────────────────────────────
city_encoded  = city_df['City_encoded'].iloc[0]
last_date     = city_df['Date'].iloc[-1]
forecast_rows = []
recent_aqi    = list(city_df['AQI'].tail(30))

for i in range(1, 8):
    future_date = last_date + timedelta(days=i)
    month  = future_date.month
    day    = future_date.day
    dow    = future_date.dayofweek
    year   = future_date.year
    qtr    = (month - 1) // 3 + 1
    season = get_season(month)

    lag1   = recent_aqi[-1]
    lag3   = recent_aqi[-3]
    lag7   = recent_aqi[-7]
    roll7  = np.mean(recent_aqi[-7:])
    roll30 = np.mean(recent_aqi[-30:])

    row = {
        'PM2.5': last_row['PM2.5'], 'PM10': last_row['PM10'],
        'NO': last_row['NO'], 'NO2': last_row['NO2'],
        'NOx': last_row['NOx'], 'NH3': last_row['NH3'],
        'CO': last_row['CO'], 'SO2': last_row['SO2'],
        'O3': last_row['O3'],
        'AQI_lag1': lag1, 'AQI_lag3': lag3, 'AQI_lag7': lag7,
        'AQI_roll7': roll7, 'AQI_roll30': roll30,
        'PM_ratio': last_row['PM_ratio'], 'NO_ratio': last_row['NO_ratio'],
        'Year': year, 'Month': month, 'Day': day,
        'DayOfWeek': dow, 'Quarter': qtr, 'Season': season,
        'City_encoded': city_encoded
    }

    X    = pd.DataFrame([row])[features]
    pred = model.predict(X)[0]
    pred = max(0, round(pred, 1))

    recent_aqi.append(pred)
    forecast_rows.append({'Date': future_date, 'AQI': pred})

forecast_df = pd.DataFrame(forecast_rows)

# ── Current AQI ───────────────────────────────────────────────────────────────
current_aqi          = last_row['AQI']
bucket, bg_color, fg = get_bucket(current_aqi)

col1, col2, col3 = st.columns(3)
with col1: st.metric("Current AQI", f"{current_aqi:.0f}")
with col2: st.metric("AQI Category", bucket)
with col3: st.metric("7-Day Avg Forecast", f"{forecast_df['AQI'].mean():.0f}")

st.markdown("---")

# ── 7-Day Forecast Cards ───────────────────────────────────────────────────────
st.markdown("### Next 7 Days")

cols = st.columns(7)
for i, (_, row) in enumerate(forecast_df.iterrows()):
    b, bg, fg = get_bucket(row['AQI'])
    with cols[i]:
        st.markdown(f"""
        <div class="forecast-card">
            <p class="day">{row['Date'].strftime('%a')}<br>{row['Date'].strftime('%b %d')}</p>
            <p class="aqi-val" style="color:{bg};">{row['AQI']:.0f}</p>
            <span class="bucket" style="background:{bg}; color:{fg};">{b}</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── Forecast Chart ─────────────────────────────────────────────────────────────
st.markdown("### Forecast Chart")

hist = city_df.tail(30)[['Date', 'AQI']].copy()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hist['Date'], y=hist['AQI'],
    mode='lines', name='Historical (last 30 days)',
    line=dict(color='#00D4AA', width=2)
))
fig.add_trace(go.Scatter(
    x=forecast_df['Date'], y=forecast_df['AQI'],
    mode='lines+markers', name='7-Day Forecast',
    line=dict(color='#FF6B35', width=2, dash='dot'),
    marker=dict(size=8, color='#FF6B35')
))
fig.add_vline(x=last_date, line_dash='dash', line_color='#8899AA',
              annotation_text='Today', annotation_font_color='#8899AA')
fig.update_layout(
    paper_bgcolor='#111827', plot_bgcolor='#111827',
    font_color='#F0F4F8', font_family='Inter', height=380,
    xaxis=dict(gridcolor='#1E2D40'),
    yaxis=dict(gridcolor='#1E2D40', title='AQI'),
    legend=dict(bgcolor='#111827', bordercolor='#1E2D40'),
    margin=dict(l=0, r=0, t=10, b=0)
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Health Advisory ────────────────────────────────────────────────────────────
st.markdown("### Health Advisory")

advisories = {
    'Good':        ('🟢', 'Air quality is good. Ideal for outdoor activities.', '#00C853'),
    'Satisfactory':('🟡', 'Air quality is acceptable. Sensitive people should consider reducing outdoor exertion.', '#64DD17'),
    'Moderate':    ('🟡', 'Sensitive groups may experience health effects. General public less likely to be affected.', '#FFD600'),
    'Poor':        ('🟠', 'Everyone may begin to experience health effects. Sensitive groups should avoid outdoor exertion.', '#FF6D00'),
    'Very Poor':   ('🔴', 'Health alert — everyone may experience serious health effects. Avoid outdoor activities.', '#DD2C00'),
    'Severe':      ('🟣', 'Health emergency. Everyone likely to be affected. Stay indoors and keep windows closed.', '#6200EA'),
}

icon, advice, color = advisories.get(bucket, ('⚪', 'No advisory available.', '#8899AA'))

st.markdown(f"""
<div style="background:#111827; border-left: 4px solid {color}; border-radius: 0 12px 12px 0; padding: 20px 24px;">
    <p style="font-size:1.1rem; font-weight:600; color:{color}; margin:0 0 8px 0;">{icon} {bucket} — {selected_city}</p>
    <p style="color:#F0F4F8; margin:0;">{advice}</p>
</div>
""", unsafe_allow_html=True)