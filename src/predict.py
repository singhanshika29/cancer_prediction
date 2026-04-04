import joblib
import numpy as np

# Load model & scaler
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

def predict(features):
    data = np.array(features).reshape(1, -1)
    data = scaler.transform(data)
    prediction = model.predict(data)[0]
    return "Malignant (Cancer)" if prediction == 1 else "Benign (No Cancer)"