import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="AQI Forecaster India",
    page_icon="🌫️",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw')

@st.cache_data
def load_data():
    city_day = pd.read_csv(os.path.join(DATA_RAW, 'city_day.csv'))
    city_day['Date'] = pd.to_datetime(city_day['Date'])
    return city_day

df = load_data()

st.title("🌫️ AQI Forecaster — India")
st.markdown("Air quality monitoring and forecasting across 26 major Indian cities · 2015–2020")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Cities Monitored", "26")
with col2: st.metric("Total Records", "1,37,000+")
with col3: st.metric("Model MAE", "15.74")
with col4: st.metric("Forecast Horizon", "7 Days")

st.markdown("---")
st.markdown("### Most Polluted Cities")
latest = df.dropna(subset=['AQI']).sort_values('Date').groupby('City').last().reset_index()
top5 = latest.nlargest(5, 'AQI')[['City', 'AQI', 'AQI_Bucket']]

cols = st.columns(5)
for i, (_, row) in enumerate(top5.iterrows()):
    with cols[i]:
        st.metric(row['City'], f"{row['AQI']:.0f}", row['AQI_Bucket'])
