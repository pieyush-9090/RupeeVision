from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import datetime
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load models and scalers
def load_models_and_scalers():
    models = {}
    scalers = {}
    currencies = ['US Dollar', 'Euro', 'Japanese Yen', 'Pound Sterling']
    
    for currency in currencies:
        try:
            models[currency] = load_model(f'lstm_{currency.lower().replace(" ", "_")}.h5')
            scalers[currency] = joblib.load(f'scaler_{currency}.pkl')
        except Exception as e:
            print(f"Error loading model or scaler for {currency}: {e}")
    return models, scalers

models, scalers = load_models_and_scalers()

# Load dataset
def load_dataset():
    df = pd.read_csv("newdataset.csv")
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df = df.pivot(index="date", columns="currency_name", values="value")
    df.bfill(inplace=True)
    return df

dataset = load_dataset()

# Prediction function
def predict_future_rates(days_to_predict=1):
    last_date = dataset.index[-1]
    future_dates = [last_date + datetime.timedelta(days=i+1) for i in range(days_to_predict)]
    future_predictions = {}
    
    for currency, model in models.items():
        scaled_data = scalers[currency].transform(dataset[[currency]])
        seq_length = 30
        last_sequence = scaled_data[-seq_length:].reshape(1, seq_length, 1)
        
        curr_sequence = last_sequence.copy()
        predictions = []
        
        for _ in range(days_to_predict):
            next_day_scaled = model.predict(curr_sequence)
            next_day = scalers[currency].inverse_transform(next_day_scaled)[0][0]
            predictions.append(next_day)
            curr_sequence = np.append(curr_sequence[:, 1:, :], next_day_scaled.reshape(1, 1, 1), axis=1)
        
        future_predictions[currency] = predictions
    
    results = pd.DataFrame(index=future_dates)
    for currency in future_predictions:
        results[currency] = future_predictions[currency]
    
    return results

# API Route for prediction
@app.route('/predict', methods=['GET'])
def predict():
    days = request.args.get('days', default=1, type=int)
    predictions = predict_future_rates(days_to_predict=days)
    return jsonify(predictions.to_dict())

# API Route for updating models and scalers
@app.route('/update', methods=['POST'])
def update_models():
    global dataset, models, scalers
    dataset = load_dataset()
    models, scalers = load_models_and_scalers()
    return jsonify({"status": "Models and scalers updated successfully"})

# API Route to check system status
@app.route('/status', methods=['GET'])
def status():
    status_report = {"models_loaded": list(models.keys()), "dataset_last_date": str(dataset.index[-1])}
    return jsonify(status_report)

if __name__ == '__main__':
    app.run(debug=True)
