import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load Data 
city_day = pd.read_csv('../data/raw/city_day.csv')
station_day = pd.read_csv('../data/raw/station_day.csv')
stations = pd.read_csv('../data/raw/stations.csv')

# Basic Info
print("=" * 60)
print("CITY DAY - SHAPE:", city_day.shape)
print(city_day.head())
print("\nColumn types:\n", city_day.dtypes)
print("\nMissing values:\n", city_day.isnull().sum())

print("=" * 60)
print("STATION DAY - SHAPE:", station_day.shape)
print(station_day.head())
print("\nMissing values:\n", station_day.isnull().sum())

print("=" * 60)
print("STATIONS - SHAPE:", stations.shape)
print(stations.head())

# Date Parsing
city_day['Date'] = pd.to_datetime(city_day['Date'])
station_day['Date'] = pd.to_datetime(station_day['Date'])

#AQI Distribution 
print("\nAQI Bucket Distribution:")
print(city_day['AQI_Bucket'].value_counts())

#Cities in dataset 
print("\nCities:", city_day['City'].unique())
print("Total cities:", city_day['City'].nunique())

# Date range
print("\nDate range:", city_day['Date'].min(), "to", city_day['Date'].max())

#Plot 1: AQI Bucket Distribution 
os.makedirs('../data/processed', exist_ok=True)

plt.figure(figsize=(10, 5))
order = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
city_day['AQI_Bucket'].value_counts().reindex(order).plot(kind='bar', color='steelblue')
plt.title('AQI Bucket Distribution Across All Cities')
plt.xlabel('AQI Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../data/processed/aqi_distribution.png')
plt.show()
print("Plot 1 saved.")

#Plot 2: Average AQI by City
plt.figure(figsize=(14, 6))
city_avg = city_day.groupby('City')['AQI'].mean().sort_values(ascending=False)
city_avg.plot(kind='bar', color='tomato')
plt.title('Average AQI by City')
plt.xlabel('City')
plt.ylabel('Average AQI')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../data/processed/avg_aqi_by_city.png')
plt.show()
print("Plot 2 saved.")

#Plot 3: AQI Trend Over Time (Delhi) 
delhi = city_day[city_day['City'] == 'Delhi'].copy()
plt.figure(figsize=(14, 5))
plt.plot(delhi['Date'], delhi['AQI'], color='darkorange', linewidth=0.8)
plt.title('AQI Trend Over Time - Delhi')
plt.xlabel('Date')
plt.ylabel('AQI')
plt.tight_layout()
plt.savefig('../data/processed/delhi_aqi_trend.png')
plt.show()
print("Plot 3 saved.")

#Plot 4: Monthly Average AQI (All Cities) 
city_day['Month'] = city_day['Date'].dt.month
monthly_avg = city_day.groupby('Month')['AQI'].mean()
plt.figure(figsize=(10, 5))
monthly_avg.plot(kind='bar', color='mediumseagreen')
plt.title('Monthly Average AQI (All Cities)')
plt.xlabel('Month')
plt.ylabel('Average AQI')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('../data/processed/monthly_avg_aqi.png')
plt.show()
print("Plot 4 saved.")

#Plot 5: Correlation Heatmap
plt.figure(figsize=(12, 8))
pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']
corr = city_day[pollutants].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Pollutant Correlation Heatmap')
plt.tight_layout()
plt.savefig('../data/processed/correlation_heatmap.png')
plt.show()
print("Plot 5 saved.")

print("\nEDA complete.")