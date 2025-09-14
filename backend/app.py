from flask import Flask, jsonify, send_from_directory
import pandas as pd
import joblib
import time
import threading
import numpy as np

app = Flask(__name__, static_folder='../frontend', static_url_path='')

# ==========================
# 1. Load trained models
# ==========================
model_wind = joblib.load("C:\\Projects\\Hackathon\\backend\\wind_speed_1h_model.pkl")
model_thunder = joblib.load("C:\\Projects\\Hackathon\\backend\\thunderstorm_1h_model.pkl")

# ==========================
# 2. Load test dataset
# ==========================
df = pd.read_csv("C:\\Projects\\Hackathon\\data\\test.csv")
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], errors='coerce')
df = df.sort_values('Formatted Date').reset_index(drop=True)

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
# 4. Shared state for predictions
# ==========================
current_index = 0
latest_prediction = {
    "timestamp": "--",
    "wind_speed": 0.0,
    "thunderstorm": False,
    "thunder_prob": 0.0
}

# ==========================
# 5. Background loop for simulation
# ==========================
def prediction_loop():
    global current_index, latest_prediction
    while True:
        X = df[feature_columns].iloc[current_index:current_index+1]

        wind_pred = float(model_wind.predict(X)[0])
        thunder_pred = int(model_thunder.predict(X)[0])
        thunder_prob = float(model_thunder.predict_proba(X)[0][1])

        latest_prediction = {
            "timestamp": str(df['Formatted Date'].iloc[current_index]),
            "wind_speed": wind_pred,
            "thunderstorm": bool(thunder_pred),
            "thunder_prob": thunder_prob
        }

        current_index += 1
        if current_index >= len(df):
            current_index = 0  # loop back to start

        time.sleep(1)  # simulate 1-hour timestep

# Start background thread
threading.Thread(target=prediction_loop, daemon=True).start()

# ==========================
# 6. Routes
# ==========================
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict')
def get_prediction():
    return jsonify(latest_prediction)

# ==========================
# 7. Run app
# ==========================
if __name__ == '__main__':
    app.run(debug=True)
