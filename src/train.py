import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import joblib
import os
#Load Processed Data
print("Loading processed data...")
df = pd.read_csv('../data/processed/city_day_processed.csv')
df['Date'] = pd.to_datetime(df['Date'])
#Feature Selection
features = [
    'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3',
    'AQI_lag1', 'AQI_lag3', 'AQI_lag7',
    'AQI_roll7', 'AQI_roll30',
    'PM_ratio', 'NO_ratio',
    'Year', 'Month', 'Day', 'DayOfWeek', 'Quarter', 'Season',
    'City_encoded'
]
target = 'AQI'
#Temporal Train/Test Split 
train = df[df['Date'] < '2019-01-01']
test  = df[df['Date'] >= '2019-01-01']
print(f"Train size: {len(train)} rows")
print(f"Test size:  {len(test)} rows")
X_train = train[features]
y_train = train[target]
X_test  = test[features]
y_test  = test[target]
#Train XGBoost Model 
print("\nTraining XGBoost model...")
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100
)
#Evaluate
y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-5))) * 100
print(f"\nModel Performance:")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAPE : {mape:.2f}%")
#Feature Importance Plot
feat_imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(12, 6))
feat_imp.plot(kind='bar', color='steelblue')
plt.title('XGBoost Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('../data/processed/feature_importance.png')
plt.show()
print("Feature importance plot saved.")
#Predicted vs Actual Plot
plt.figure(figsize=(14, 5))
plt.plot(test['Date'].values[:200], y_test.values[:200], label='Actual', color='blue', linewidth=0.9)
plt.plot(test['Date'].values[:200], y_pred[:200], label='Predicted', color='red', linewidth=0.9, linestyle='--')
plt.title('Predicted vs Actual AQI (First 200 test days)')
plt.xlabel('Date')
plt.ylabel('AQI')
plt.legend()
plt.tight_layout()
plt.savefig('../data/processed/predicted_vs_actual.png')
plt.show()
print("Predicted vs actual plot saved.")
#Save Model
os.makedirs('../models', exist_ok=True)
joblib.dump(model, '../models/xgb_aqi_model.pkl')
joblib.dump(features, '../models/feature_names.pkl')
print("\nModel saved to models/xgb_aqi_model.pkl")
print("Training complete.")