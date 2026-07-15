import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Station Analysis", page_icon="📍", layout="wide")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw')

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
</style>
""", unsafe_allow_html=True)

# ── Load Data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    station_day = pd.read_csv(os.path.join(DATA_RAW, 'station_day.csv'))
    stations    = pd.read_csv(os.path.join(DATA_RAW, 'stations.csv'))
    station_day['Date'] = pd.to_datetime(station_day['Date'])
    return station_day, stations

station_day, stations = load_data()

PLOTLY_THEME = dict(
    paper_bgcolor='#111827',
    plot_bgcolor='#111827',
    font_color='#F0F4F8',
    font_family='Inter'
)

def aqi_color(value):
    if value <= 50:    return '#00C853'
    elif value <= 100: return '#64DD17'
    elif value <= 200: return '#FFD600'
    elif value <= 300: return '#FF6D00'
    elif value <= 400: return '#DD2C00'
    else:              return '#6200EA'

# ── Page Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>📍 Station Analysis</h1>
    <p>Drill down to individual monitoring station data across India</p>
</div>
""", unsafe_allow_html=True)

# ── Merge station info ─────────────────────────────────────────────────────────
merged = station_day.merge(stations, on='StationId', how='left')

# ── City Selector ──────────────────────────────────────────────────────────────
cities = sorted(merged['City'].dropna().unique().tolist())
selected_city = st.selectbox("Select City", cities, index=cities.index('Delhi') if 'Delhi' in cities else 0)

city_stations = merged[merged['City'] == selected_city]
station_list  = city_stations['StationId'].unique().tolist()

st.markdown(f"**{len(station_list)} monitoring stations** found in {selected_city}")
st.markdown("---")

# ── Top Metrics ────────────────────────────────────────────────────────────────
city_aqi    = city_stations.dropna(subset=['AQI'])
avg_aqi     = city_aqi['AQI'].mean()
max_station = city_aqi.groupby('StationId')['AQI'].mean().idxmax()
min_station = city_aqi.groupby('StationId')['AQI'].mean().idxmin()
total_recs  = len(city_aqi)

col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Avg AQI", f"{avg_aqi:.0f}")
with col2: st.metric("Most Polluted Station", max_station)
with col3: st.metric("Cleanest Station", min_station)
with col4: st.metric("Total Records", f"{total_recs:,}")

st.markdown("---")

# ── Station Comparison ─────────────────────────────────────────────────────────
st.markdown("### Average AQI by Station")

station_avg = city_aqi.groupby('StationId')['AQI'].mean().sort_values(ascending=False).reset_index()
station_avg.columns = ['Station', 'AQI']
colors = [aqi_color(v) for v in station_avg['AQI']]

fig1 = go.Figure(go.Bar(
    x=station_avg['Station'], y=station_avg['AQI'],
    marker_color=colors,
    text=station_avg['AQI'].round(0).astype(int),
    textposition='outside',
    textfont=dict(color='#F0F4F8', size=11)
))
fig1.update_layout(
    **PLOTLY_THEME, height=350,
    xaxis=dict(gridcolor='#1E2D40', tickangle=45),
    yaxis=dict(gridcolor='#1E2D40', title='Average AQI'),
    margin=dict(l=0, r=0, t=10, b=80)
)
st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")

# ── Station Trend ──────────────────────────────────────────────────────────────
st.markdown("### AQI Trend by Station")

selected_station = st.selectbox("Select Station", station_list)
station_df = city_stations[city_stations['StationId'] == selected_station].sort_values('Date')

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=station_df['Date'], y=station_df['AQI'],
    mode='lines',
    line=dict(color='#00D4AA', width=1.2),
    fill='tozeroy',
    fillcolor='rgba(0,212,170,0.08)',
    name='Daily AQI'
))
fig2.update_layout(
    **PLOTLY_THEME, height=320,
    xaxis=dict(gridcolor='#1E2D40'),
    yaxis=dict(gridcolor='#1E2D40', title='AQI'),
    margin=dict(l=0, r=0, t=10, b=0)
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── Pollutant Heatmap ──────────────────────────────────────────────────────────
st.markdown("### Pollutant Levels by Station")

pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
available  = [p for p in pollutants if p in city_aqi.columns]
heatmap_df = city_aqi.groupby('StationId')[available].mean().round(2)

fig3 = go.Figure(go.Heatmap(
    z=heatmap_df.values,
    x=heatmap_df.columns.tolist(),
    y=heatmap_df.index.tolist(),
    colorscale='RdYlGn_r',
    text=heatmap_df.values.round(1),
    texttemplate='%{text}',
    textfont=dict(size=10),
    colorbar=dict(tickfont=dict(color='#F0F4F8'))
))
fig3.update_layout(
    **PLOTLY_THEME, height=400,
    xaxis=dict(title='Pollutant'),
    yaxis=dict(title='Station'),
    margin=dict(l=0, r=0, t=10, b=0)
)
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ── Station Summary Table ──────────────────────────────────────────────────────
st.markdown("### All Stations in Dataset")

station_summary = merged.groupby('StationId').agg(
    City=('City', 'first'),
    Avg_AQI=('AQI', 'mean'),
    Total_Records=('AQI', 'count')
).reset_index()
station_summary['Avg_AQI'] = station_summary['Avg_AQI'].round(1)
station_summary = station_summary.sort_values('Avg_AQI', ascending=False).reset_index(drop=True)
station_summary.columns = ['Station ID', 'City', 'Avg AQI', 'Records']

st.dataframe(station_summary, use_container_width=True, hide_index=True)