from flask import Flask, request, jsonify
import joblib
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load models & scalers
currencies = ["US Dollar", "Euro", "Japanese Yen", "Pound Sterling"]
models = {c: tf.keras.models.load_model(f"lstm_{c.lower().replace(' ', '_')}.h5") for c in currencies}
scalers = {c: joblib.load(f"scaler_{c}.pkl") for c in currencies}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Receive input data
    days = data.get("days", 1)  # Default to 1-day prediction
    results = {}

    for currency in currencies:
        model = models[currency]
        scaler = scalers[currency]
        
        last_30_days = np.array(data["last_30_days"][currency]).reshape(-1, 1)
        last_30_days_scaled = scaler.transform(last_30_days)
        input_seq = np.expand_dims(last_30_days_scaled, axis=0)
        
        predictions = []
        for _ in range(days):
            pred_scaled = model.predict(input_seq)
            pred = scaler.inverse_transform(pred_scaled)[0, 0]
            predictions.append(pred)
            
            input_seq = np.roll(input_seq, -1, axis=1)
            input_seq[0, -1, 0] = pred_scaled[0, 0]

        results[currency] = predictions
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
