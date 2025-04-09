import joblib
import numpy as np

# Load model and scaler
model = joblib.load("models/diabetes_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Example input: [Pregnancies, Glucose, BP, Skin, Insulin, BMI, DPF, Age]
sample = [[6, 148, 72, 35, 0, 33.6, 0.627, 50]]

# Scale the input
sample_scaled = scaler.transform(sample)

# Predict
result = model.predict(sample_scaled)
print("Prediction:", "Diabetic" if result[0] == 1 else "Not Diabetic")
