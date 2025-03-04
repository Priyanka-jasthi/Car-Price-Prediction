import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Load model and feature list
try:
    model = joblib.load("car_price_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
except Exception as e:
    st.error(f"Model loading failed! Error: {e}")
    st.stop()

st.title("Car Price Prediction App")
st.write("Enter car attributes to predict the estimated car price:")

# User input (only important features)
wheelbase = st.number_input("Wheelbase", value=95.0)
carlength = st.number_input("Car Length", value=175.0)
carwidth = st.number_input("Car Width", value=65.0)
horsepower = st.number_input("Horsepower", value=100)

# Create DataFrame with user input
input_data = pd.DataFrame([[wheelbase, carlength, carwidth, horsepower]], 
                          columns=["wheelbase", "carlength", "carwidth", "horsepower"])

# Add missing columns and set them to 0
for col in feature_columns:
    if col not in input_data.columns:
        input_data[col] = 0  

# Reorder to match training data
input_data = input_data[feature_columns]

# Predict price
if st.button("Predict Price"):
    predicted_price = model.predict(input_data)[0]
    st.write(f"Estimated Car Price: **${predicted_price:,.2f}**")



