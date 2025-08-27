import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("google_stock_model.pkl")

st.title("📈 Google Stock Price Prediction")

open_price = st.number_input("Enter today's opening price", min_value=0.0)
high_price = st.number_input("Enter today's highest price", min_value=0.0)
low_price = st.number_input("Enter today's lowest price", min_value=0.0)

if st.button("Predict Closing Price"):
    features = np.array([[open_price, high_price, low_price]])  # only 3 features
    prediction = model.predict(features)
    st.success(f"Predicted Closing Price: ${prediction[0]:.2f}")
