import streamlit as st
from src.predict import predict

st.title("Cancer Prediction App")

features = []

for i in range(30):
    val = st.number_input(f"Feature {i+1}")
    features.append(val)

if st.button("Predict"):
    result = predict(features)
    st.success(f"Prediction: {result}")