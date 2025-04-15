import joblib
import numpy as np
import pandas as pd

# Load model & scaler
model = joblib.load("models/diabetes_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Feature columns
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Sample input 
sample_data = pd.DataFrame([[6,148,72,35,0,33.6,0.627,50]], columns=columns)

# Scale and predict
scaled = scaler.transform(sample_data)
prediction = model.predict(scaled)

print("Prediction:", "Diabetic" if prediction[0] == 1 else "Not Diabetic")

