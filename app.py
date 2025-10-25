import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open('../model/diabetes_model.pkl', 'rb'))
_, _, _, _, scaler = pickle.load(open('../model/preprocessed.pkl', 'rb'))

st.title("Diabetes Prediction App")

# Input fields
pregnancies = st.number_input("Number of Pregnancies", min_value=0)
glucose = st.number_input("Glucose Level")
bp = st.number_input("Blood Pressure")
skin = st.number_input("Skin Thickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age")

if st.button("Predict"):
    input_data = [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
    input_scaled = scaler.transform(np.array(input_data).reshape(1, -1))
    prediction = model.predict(input_scaled)
    result = "Diabetes" if prediction[0] == 1 else "No Diabetes"
    st.success(f"Prediction: {result}")
