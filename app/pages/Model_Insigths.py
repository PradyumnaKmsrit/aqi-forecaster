import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Model Insights", page_icon="🤖", layout="wide")

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
.info-box { background-color: #0D1F35; border-left: 3px solid #00D4AA; border-radius: 0 8px 8px 0; padding: 16px 20px; margin: 16px 0; }
</style>
""", unsafe_allow_html=True)

# ── Load ───────────────────────────────────────────────────────────────────────
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

PLOTLY_THEME = dict(
    paper_bgcolor='#111827',
    plot_bgcolor='#111827',
    font_color='#F0F4F8',
    font_family='Inter'
)

# ── Page Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>🤖 Model Insights</h1>
    <p>XGBoost model performance, feature importance, and prediction analysis</p>
</div>
""", unsafe_allow_html=True)

# ── Train/Test Split ───────────────────────────────────────────────────────────
train = df[df['Date'] < '2019-01-01']
test  = df[df['Date'] >= '2019-01-01']

X_test = test[features]
y_test = test['AQI']
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-5))) * 100
r2   = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2)

# ── Model Metrics ──────────────────────────────────────────────────────────────
st.markdown("### Model Performance")

col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("MAE",      f"{mae:.2f}")
with col2: st.metric("RMSE",     f"{rmse:.2f}")
with col3: st.metric("MAPE",     f"{mape:.2f}%")
with col4: st.metric("R² Score", f"{r2:.4f}")

st.markdown("""
<div class="info-box">
    <strong style="color:#00D4AA;">What these metrics mean</strong>
    <p style="color:#8899AA; margin:6px 0 0 0; font-size:0.85rem;">
    MAE of 15.74 means predictions are off by ~16 AQI points on average.
    MAPE of 12% means the model is accurate within 12% of the actual value.
    R² close to 1.0 means the model explains most of the variance in AQI.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Feature Importance ─────────────────────────────────────────────────────────
st.markdown("### Feature Importance")

feat_imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
colors   = ['#00D4AA' if i >= len(feat_imp) - 5 else '#1E3A5F' for i in range(len(feat_imp))]

fig1 = go.Figure(go.Bar(
    x=feat_imp.values,
    y=feat_imp.index,
    orientation='h',
    marker_color=colors,
    text=[f'{v:.3f}' for v in feat_imp.values],
    textposition='outside',
    textfont=dict(color='#F0F4F8', size=10)
))
fig1.update_layout(
    **PLOTLY_THEME, height=500,
    xaxis=dict(gridcolor='#1E2D40', title='Importance Score'),
    yaxis=dict(gridcolor='#1E2D40'),
    margin=dict(l=0, r=60, t=10, b=0)
)
st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")

# ── Predicted vs Actual ────────────────────────────────────────────────────────
st.markdown("### Predicted vs Actual AQI")

cities   = sorted(test['City'].unique().tolist())
sel_city = st.selectbox("Filter by City", ['All Cities'] + cities)

if sel_city == 'All Cities':
    plot_test = test.copy()
    plot_pred = y_pred
else:
    mask      = test['City'] == sel_city
    plot_test = test[mask].copy()
    plot_pred = model.predict(plot_test[features])

plot_test = plot_test.head(200)
plot_pred = plot_pred[:200]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=plot_test['Date'], y=plot_test['AQI'],
    mode='lines', name='Actual',
    line=dict(color='#00D4AA', width=1.5)
))
fig2.add_trace(go.Scatter(
    x=plot_test['Date'], y=plot_pred,
    mode='lines', name='Predicted',
    line=dict(color='#FF6B35', width=1.5, dash='dot')
))
fig2.update_layout(
    **PLOTLY_THEME, height=380,
    xaxis=dict(gridcolor='#1E2D40'),
    yaxis=dict(gridcolor='#1E2D40', title='AQI'),
    legend=dict(bgcolor='#111827', bordercolor='#1E2D40'),
    margin=dict(l=0, r=0, t=10, b=0)
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── Error Distribution ─────────────────────────────────────────────────────────
st.markdown("### Prediction Error Distribution")

errors = y_test.values - y_pred

fig3 = go.Figure(go.Histogram(
    x=errors, nbinsx=60,
    marker_color='#00D4AA', opacity=0.8
))
fig3.add_vline(x=0, line_dash='dash', line_color='#FF6B35',
               annotation_text='Zero Error', annotation_font_color='#FF6B35')
fig3.update_layout(
    **PLOTLY_THEME, height=300,
    xaxis=dict(gridcolor='#1E2D40', title='Prediction Error (Actual - Predicted)'),
    yaxis=dict(gridcolor='#1E2D40', title='Count'),
    margin=dict(l=0, r=0, t=10, b=0)
)
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ── Model Configuration ────────────────────────────────────────────────────────
st.markdown("### Model Configuration")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div style="background:#111827; border:1px solid #1E2D40; border-radius:12px; padding:24px;">
        <h4 style="color:#00D4AA; margin:0 0 16px 0;">XGBoost Parameters</h4>
        <p style="color:#8899AA; margin:4px 0;">n_estimators: <span style="color:#F0F4F8;">500</span></p>
        <p style="color:#8899AA; margin:4px 0;">learning_rate: <span style="color:#F0F4F8;">0.05</span></p>
        <p style="color:#8899AA; margin:4px 0;">max_depth: <span style="color:#F0F4F8;">6</span></p>
        <p style="color:#8899AA; margin:4px 0;">subsample: <span style="color:#F0F4F8;">0.8</span></p>
        <p style="color:#8899AA; margin:4px 0;">colsample_bytree: <span style="color:#F0F4F8;">0.8</span></p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background:#111827; border:1px solid #1E2D40; border-radius:12px; padding:24px;">
        <h4 style="color:#00D4AA; margin:0 0 16px 0;">Training Details</h4>
        <p style="color:#8899AA; margin:4px 0;">Train period: <span style="color:#F0F4F8;">2015–2018</span></p>
        <p style="color:#8899AA; margin:4px 0;">Test period: <span style="color:#F0F4F8;">2019–2020</span></p>
        <p style="color:#8899AA; margin:4px 0;">Train rows: <span style="color:#F0F4F8;">17,313</span></p>
        <p style="color:#8899AA; margin:4px 0;">Test rows: <span style="color:#F0F4F8;">12,036</span></p>
        <p style="color:#8899AA; margin:4px 0;">Features used: <span style="color:#F0F4F8;">23</span></p>
    </div>
    """, unsafe_allow_html=True)