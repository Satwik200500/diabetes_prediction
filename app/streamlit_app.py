import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("models/diabetes_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Streamlit app title and description
st.set_page_config(page_title="Diabetes Prediction App",  page_icon=None)
st.title(" **Diabetes Prediction App**")
st.write("### Predict whether a person has diabetes based on their health data.")
st.write("Please enter the following details:")

# Adding some explanations for each input parameter
st.markdown("""
### Parameters:
- **Pregnancies**: Number of times the person has been pregnant.
- **Glucose**: Glucose concentration in the blood (mg/dl).
- **Blood Pressure**: Blood pressure value (mm Hg).
- **Skin Thickness**: Skin fold thickness (mm).
- **Insulin**: Insulin level in the blood (µU/ml).
- **BMI**: Body Mass Index (kg/m²).
- **Diabetes Pedigree Function**: A function that scores the likelihood of diabetes based on family history.
- **Age**: Age of the person (years).
""")

# Create input fields 
preg = st.number_input("**Pregnancies**", min_value=0, max_value=20, step=1, help="Enter the number of times the person has been pregnant.")
glucose = st.number_input("**Glucose**", min_value=0, max_value=200, step=1, help="Enter the glucose concentration in mg/dl.")
bp = st.number_input("**Blood Pressure**", min_value=0, max_value=200, step=1, help="Enter the blood pressure in mm Hg.")
skin = st.number_input("**Skin Thickness**", min_value=0, max_value=100, step=1, help="Enter the skin fold thickness in mm.")
insulin = st.number_input("**Insulin**", min_value=0, max_value=500, step=1, help="Enter the insulin level in µU/ml.")
bmi = st.number_input("**BMI**", min_value=0.0, max_value=100.0, step=0.1, help="Enter the Body Mass Index (kg/m²).")
dpf = st.number_input("**Diabetes Pedigree Function**", min_value=0.0, max_value=2.5, step=0.01, help="Enter the diabetes pedigree function value.")
age = st.number_input("**Age**", min_value=1, max_value=120, step=1, help="Enter the age of the person.")

# Predict button
if st.button("**Predict**"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)

    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    
    # results display
    if prediction[0] == 1:
        st.markdown(f"<h2 style='color:red;'> Prediction: {result}</h2>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h2 style='color:green;'> Prediction: {result}</h2>", unsafe_allow_html=True)
    
    st.write("### Thank you for using the Diabetes Prediction App!")
    st.write("For a more accurate result, please consult with a medical professional.")

