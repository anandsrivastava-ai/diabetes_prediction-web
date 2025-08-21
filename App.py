import streamlit as st
import joblib
import numpy as np

# Load model and scaler
rf = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🩺 Diabetes Prediction App")

age = st.number_input("Age", 0, 120, 33)
bmi = st.number_input("BMI", 0.0, 100.0, 25.0)
glucose = st.number_input("Glucose Level", 0, 300, 120)

if st.button("Predict"):
    input_data = np.array([[glucose, bmi, age]])
    input_scaled = scaler.transform(input_data)

    prediction = rf.predict(input_scaled)[0]
    prediction_prob = rf.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Risk of diabetes (Probability: {prediction_prob:.2f})")
    else:
        st.success(f"✅ Not likely to have diabetes (Probability: {prediction_prob:.2f})")
