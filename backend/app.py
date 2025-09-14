import time
import pandas as pd
import joblib

# ==========================
# 1. Load trained models
# ==========================
model_wind = joblib.load("C:\\Projects\\Hackathon\\backend\\wind_speed_1h_model.pkl")
model_thunder = joblib.load("C:\\Projects\\Hackathon\\backend\\thunderstorm_1h_model.pkl")

# ==========================
# 2. Load validation/test dataset
# ==========================
df = pd.read_csv("C:\\Projects\\Hackathon\\data\\test.csv")
# df=df[(df['year'] >= 2014)]

# Ensure datetime is parsed & sorted
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], errors='coerce')
df = df.sort_values('Formatted Date').reset_index(drop=True)
# 14-05-2014 19:00

# ==========================
# 3. Feature columns
# ==========================
numerical_features = [
    'Temperature (C)', 'Apparent Temperature (C)', 'Humidity',
    'Pressure (millibars)', 'Visibility (km)', 'Wind Bearing (degrees)',
    'Wind_Speed_prev_1h', 'Temperature_prev_1h', 'Humidity_prev_1h',
    'Pressure_prev_1h', 'Thunderstorm_prev_1h',
    'Wind_Speed_roll3h', 'Temperature_roll3h',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
]
categorical_features = [col for col in df.columns if col.startswith("Precip Type_")]
feature_columns = numerical_features + categorical_features

# ==========================
# 4. Simulate real-time predictions
# ==========================
for i in range(len(df)):
    X = df[feature_columns].iloc[i:i+1]

    wind_pred = model_wind.predict(X)[0]
    thunder_pred = model_thunder.predict(X)[0]

    # Get probability of thunderstorm = class 1
    thunder_prob = model_thunder.predict_proba(X)[0][1]

    timestamp = df['Formatted Date'].iloc[i]

    print(f"Time step {i}:")
    print(f"Timestamp: {timestamp}")
    print(f"Predicted Wind Speed (1h ahead): {wind_pred:.2f} km/h")
    print(f"Predicted Thunderstorm (1h ahead): {'Yes' if thunder_pred == 1 else 'No'}")
    print("-" * 40)
    time.sleep(1)  # simulate real-time (1 sec = 1 hour)

