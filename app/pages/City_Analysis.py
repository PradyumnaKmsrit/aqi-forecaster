import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

st.set_page_config(page_title="City Analysis", page_icon="🏙️", layout="wide", initial_sidebar_state="expanded")

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
    df = pd.read_csv(os.path.join(DATA_RAW, 'city_day.csv'))
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    return df

df = load_data()

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
    <h1>🏙️ City Analysis</h1>
    <p>Explore AQI trends, seasonal patterns, and pollutant breakdown for any Indian city</p>
</div>
""", unsafe_allow_html=True)

# ── City Selector ──────────────────────────────────────────────────────────────
cities = sorted(df['City'].unique().tolist())
selected_city = st.selectbox("Select City", cities, index=cities.index('Delhi'))

city_df = df[df['City'] == selected_city].copy()

# ── Top Metrics ────────────────────────────────────────────────────────────────
avg_aqi   = city_df['AQI'].mean()
max_aqi   = city_df['AQI'].max()
min_aqi   = city_df['AQI'].min()
worst_mon = city_df.groupby('Month')['AQI'].mean().idxmax()
month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
               7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Average AQI", f"{avg_aqi:.0f}")
with col2: st.metric("Highest AQI", f"{max_aqi:.0f}")
with col3: st.metric("Lowest AQI", f"{min_aqi:.0f}")
with col4: st.metric("Worst Month", month_names[worst_mon])

st.markdown("---")

# ── AQI Trend Over Time ────────────────────────────────────────────────────────
st.markdown("### AQI Trend Over Time")

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=city_df['Date'], y=city_df['AQI'],
    mode='lines',
    line=dict(color='#00D4AA', width=1.2),
    fill='tozeroy',
    fillcolor='rgba(0, 212, 170, 0.08)',
    name='Daily AQI'
))

rolling = city_df['AQI'].rolling(30).mean()
fig1.add_trace(go.Scatter(
    x=city_df['Date'], y=rolling,
    mode='lines',
    line=dict(color='#FF6B35', width=2, dash='dot'),
    name='30-day Average'
))

fig1.update_layout(
    **PLOTLY_THEME, height=350,
    xaxis=dict(gridcolor='#1E2D40', showgrid=True),
    yaxis=dict(gridcolor='#1E2D40', showgrid=True, title='AQI'),
    legend=dict(bgcolor='#111827', bordercolor='#1E2D40'),
    margin=dict(l=0, r=0, t=10, b=0)
)
st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")

# ── Monthly and Yearly ─────────────────────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.markdown("### Monthly Average AQI")
    monthly = city_df.groupby('Month')['AQI'].mean().reset_index()
    monthly['Month_Name'] = monthly['Month'].map(month_names)
    colors = [aqi_color(v) for v in monthly['AQI']]

    fig2 = go.Figure(go.Bar(
        x=monthly['Month_Name'], y=monthly['AQI'],
        marker_color=colors,
        text=monthly['AQI'].round(0).astype(int),
        textposition='outside',
        textfont=dict(color='#F0F4F8', size=11)
    ))
    fig2.update_layout(**PLOTLY_THEME, height=320,
        xaxis=dict(gridcolor='#1E2D40'),
        yaxis=dict(gridcolor='#1E2D40', title='AQI'),
        margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)

with col_r:
    st.markdown("### Yearly Average AQI")
    yearly = city_df.groupby('Year')['AQI'].mean().reset_index()

    fig3 = go.Figure(go.Bar(
        x=yearly['Year'].astype(str), y=yearly['AQI'],
        marker_color='#00D4AA',
        text=yearly['AQI'].round(0).astype(int),
        textposition='outside',
        textfont=dict(color='#F0F4F8', size=11)
    ))
    fig3.update_layout(**PLOTLY_THEME, height=320,
        xaxis=dict(gridcolor='#1E2D40'),
        yaxis=dict(gridcolor='#1E2D40', title='AQI'),
        margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ── Pollutant Breakdown ────────────────────────────────────────────────────────
st.markdown("### Pollutant Breakdown")

pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3']
poll_avg = city_df[pollutants].mean().reset_index()
poll_avg.columns = ['Pollutant', 'Average']
poll_avg = poll_avg.sort_values('Average', ascending=True)

fig4 = go.Figure(go.Bar(
    x=poll_avg['Average'], y=poll_avg['Pollutant'],
    orientation='h',
    marker_color='#FF6B35',
    text=poll_avg['Average'].round(2),
    textposition='outside',
    textfont=dict(color='#F0F4F8', size=11)
))
fig4.update_layout(**PLOTLY_THEME, height=320,
    xaxis=dict(gridcolor='#1E2D40', title='Average Concentration'),
    yaxis=dict(gridcolor='#1E2D40'),
    margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ── AQI Category Distribution ──────────────────────────────────────────────────
st.markdown("### AQI Category Distribution")

bucket_counts = city_df['AQI_Bucket'].value_counts().reset_index()
bucket_counts.columns = ['Category', 'Count']
order   = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
bcolors = ['#00C853', '#64DD17', '#FFD600', '#FF6D00', '#DD2C00', '#6200EA']
bucket_counts['Category'] = pd.Categorical(bucket_counts['Category'], categories=order, ordered=True)
bucket_counts = bucket_counts.sort_values('Category')

fig5 = go.Figure(go.Bar(
    x=bucket_counts['Category'], y=bucket_counts['Count'],
    marker_color=bcolors[:len(bucket_counts)],
    text=bucket_counts['Count'],
    textposition='outside',
    textfont=dict(color='#F0F4F8', size=11)
))
fig5.update_layout(**PLOTLY_THEME, height=300,
    xaxis=dict(gridcolor='#1E2D40'),
    yaxis=dict(gridcolor='#1E2D40', title='Days'),
    margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig5, use_container_width=True)
