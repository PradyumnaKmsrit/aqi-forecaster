import pandas as pd
import numpy as np
import os

# ── Load Data ──────────────────────────────────────────────────────────────────
print("Loading data...")
city_day = pd.read_csv('../data/raw/city_day.csv')
city_day['Date'] = pd.to_datetime(city_day['Date'])

# ── Sort by City and Date ──────────────────────────────────────────────────────
city_day = city_day.sort_values(['City', 'Date']).reset_index(drop=True)

# ── Handle Missing Values ──────────────────────────────────────────────────────
print("Handling missing values...")

pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']

for col in pollutants:
    city_day[col] = city_day.groupby('City')[col].transform(
        lambda x: x.ffill().bfill()
    )

city_day['AQI'] = city_day.groupby('City')['AQI'].transform(
    lambda x: x.ffill().bfill()
)

city_day = city_day.dropna(subset=['AQI'])

print("Missing values after cleaning:")
print(city_day.isnull().sum())

# ── Feature Engineering ────────────────────────────────────────────────────────
print("\nEngineering features...")

city_day['Year']      = city_day['Date'].dt.year
city_day['Month']     = city_day['Date'].dt.month
city_day['Day']       = city_day['Date'].dt.day
city_day['DayOfWeek'] = city_day['Date'].dt.dayofweek
city_day['Quarter']   = city_day['Date'].dt.quarter

def get_season(month):
    if month in [12, 1, 2]:
        return 0  # Winter
    elif month in [3, 4, 5]:
        return 1  # Spring
    elif month in [6, 7, 8, 9]:
        return 2  # Monsoon
    else:
        return 3  # Autumn

city_day['Season'] = city_day['Month'].apply(get_season)

city_day['AQI_lag1'] = city_day.groupby('City')['AQI'].shift(1)
city_day['AQI_lag3'] = city_day.groupby('City')['AQI'].shift(3)
city_day['AQI_lag7'] = city_day.groupby('City')['AQI'].shift(7)

city_day['AQI_roll7'] = city_day.groupby('City')['AQI'].transform(
    lambda x: x.rolling(window=7, min_periods=1).mean()
)
city_day['AQI_roll30'] = city_day.groupby('City')['AQI'].transform(
    lambda x: x.rolling(window=30, min_periods=1).mean()
)

city_day['PM_ratio'] = city_day['PM2.5'] / (city_day['PM10'] + 1e-5)
city_day['NO_ratio'] = city_day['NO2'] / (city_day['NOx'] + 1e-5)

city_day['City_encoded'] = city_day['City'].astype('category').cat.codes

city_day = city_day.dropna(subset=['AQI_lag1', 'AQI_lag3', 'AQI_lag7'])

# ── Save ───────────────────────────────────────────────────────────────────────
os.makedirs('../data/processed', exist_ok=True)
city_day.to_csv('../data/processed/city_day_processed.csv', index=False)

print("\nProcessed data shape:", city_day.shape)
print("Columns:", list(city_day.columns))
print("\nSample:")
print(city_day.head())
print("\nPreprocessing complete. Saved to data/processed/city_day_processed.csv")