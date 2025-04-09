import streamlit as st
import joblib
import numpy as np

# Load model & scaler
model = joblib.load("models/diabetes_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Streamlit app title
st.title("🧠 Diabetes Prediction App")
st.write("Enter patient information below:")

# Input fields
preg = st.number_input("Pregnancies", 0)
glucose = st.number_input("Glucose", 0)
bp = st.number_input("Blood Pressure", 0)
skin = st.number_input("Skin Thickness", 0)
insulin = st.number_input("Insulin", 0)
bmi = st.number_input("BMI", 0.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0)
age = st.number_input("Age", 0)

# Predict button
if st.button("Predict"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)

    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    st.success(f"🔍 Prediction: {result}")
