# 🌫️ AQI Forecaster — India

A comprehensive Air Quality Index (AQI) forecasting and analysis dashboard for 26 major Indian cities using historical pollution data from 2015–2020.

## 🔗 Live Demo
Add your Streamlit Cloud URL after deployment
## 🔗 Live Demo
https://aqi-forecaster-y9czaqshoqdwbgapp9dxyre.streamlit.app

## 📊 Project Overview

This is a full-stack machine learning project that includes:
- Exploratory Data Analysis on 1,37,000+ records across 26 cities
- XGBoost Regression Model for next 7-day AQI forecasting
- Multi-page Streamlit Dashboard with dark theme UI

## 🧠 Model Performance

| Metric | Score |
|--------|-------|
| MAE | 15.74 |
| RMSE | 36.02 |
| MAPE | 12.06% |
| R2 Score | 0.9101 |

## 🗂️ Dataset

- Source: Central Pollution Control Board (CPCB) via Kaggle
- Cities: 26 major Indian cities
- Period: 2015–2020

## ⚙️ Feature Engineering

| Feature | Description |
|---------|-------------|
| AQI_lag1/3/7 | Previous 1, 3, 7 day AQI |
| AQI_roll7/30 | 7 and 30 day rolling average |
| Season | Winter / Spring / Monsoon / Autumn |
| PM_ratio | PM2.5 / PM10 ratio |
| Month, DayOfWeek | Temporal features |

## 🚀 How to Run Locally

pip install -r requirements.txt
cd app
streamlit run Home.py

## 📱 App Pages

- Home — Overview, most polluted cities, AQI scale reference
- City Analysis — Per-city trends, monthly/yearly patterns, pollutant breakdown
- Forecast — 7-day XGBoost predictions with health advisory
- Station Analysis — Station-level drilling, pollutant heatmap
- Model Insights — Feature importance, predicted vs actual, error distribution

## 🛠️ Tech Stack

- Language: Python 3.11
- ML: XGBoost, Scikit-learn
- Dashboard: Streamlit, Plotly
- Data: Pandas, NumPy

## 👤 Author

Pradyumna S Kushtagi
