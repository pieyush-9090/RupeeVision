import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from flask import Flask, request, jsonify
from datetime import datetime, timedelta

app = Flask(__name__)

# Load Models and Scalers
models = {
    "US Dollar": tf.keras.models.load_model("models/lstm_usd.h5"),
    "Euro": tf.keras.models.load_model("models/lstm_euro.h5"),
    "Japanese Yen": tf.keras.models.load_model("models/lstm_yen.h5"),
    "Pound Sterling": tf.keras.models.load_model("models/lstm_pound.h5"),
}

scalers = {
    "US Dollar": joblib.load("models/scaler_US Dollar.pkl"),
    "Euro": joblib.load("models/scaler_Euro.pkl"),
    "Japanese Yen": joblib.load("models/scaler_Japanese Yen.pkl"),
    "Pound Sterling": joblib.load("models/scaler_Pound Sterling.pkl"),
}

# Function to make future predictions
def predict_exchange_rates(days_to_predict=7):
    last_known_date = datetime.today()  # Real-time date
    future_dates = [last_known_date + timedelta(days=i+1) for i in range(days_to_predict)]
    
    predictions = {}
    for currency, model in models.items():
        scaler = scalers[currency]
        
        # Load dataset to get last 30 days of data
        df = pd.read_csv("dataset/newdataset.csv")
        df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
        df = df.pivot(index="date", columns="currency_name", values="value").bfill()
        
        # Scale and get the last sequence
        scaled_data = scaler.transform(df[[currency]])
        last_sequence = scaled_data[-30:].reshape(1, 30, 1)
        
        # Generate predictions
        curr_sequence = last_sequence.copy()
        future_rates = []
        for _ in range(days_to_predict):
            next_day_scaled = model.predict(curr_sequence)
            next_day = scaler.inverse_transform(next_day_scaled)[0][0]
            future_rates.append(next_day)
            curr_sequence = np.append(curr_sequence[:, 1:, :], next_day_scaled.reshape(1, 1, 1), axis=1)
        
        predictions[currency] = future_rates
    
    return pd.DataFrame(predictions, index=future_dates).reset_index().rename(columns={"index": "Date"}).to_dict(orient="records")

@app.route("/predict", methods=["GET"])
def predict():
    days = request.args.get("days", default=7, type=int)
    results = predict_exchange_rates(days)
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
