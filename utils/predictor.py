# utils/predictor.py

import os
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# 1. Dynamically get the absolute path of the current file (predictor.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Navigate up one level to the main project folder, then into the 'model' folder
model_dir = os.path.join(current_dir, '..', 'model')

# 3. Construct the exact, unbreakable paths to the files
model_path = os.path.join(model_dir, 'lstm_model.keras')
scaler_path = os.path.join(model_dir, 'scaler.pkl')

# 4. Load the files using the absolute paths
model = load_model(model_path, compile=False)
scaler = joblib.load(scaler_path)

def predict_prices(data):
    scaled = scaler.transform(data)
    X = []

    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])

    X = np.array(X)

    preds = model.predict(X)
    preds = scaler.inverse_transform(preds)

    return preds