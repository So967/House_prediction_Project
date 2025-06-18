import streamlit as st

# Streamlit Page Config (for mobile responsiveness)
st.set_page_config(page_title="House Price Predictor", layout="centered")


import pandas as pd
import numpy as np
import joblib
import os

# Load the trained model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "house_price_model.pkl")
    return joblib.load(model_path)

model = load_model()

# UI Header
st.title("🏠 House Price Predictor")
st.markdown("Enter the details below to estimate the house price:")

# Using a form to avoid rerun on every input
with st.form("input_form"):
    area = st.number_input("📐 Area (in sq ft)", min_value=100, max_value=10000, value=1500)
    bedrooms = st.number_input("🛏️ Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("🛁 Bathrooms", min_value=1, max_value=10, value=2)
    
    submitted = st.form_submit_button("Predict Price")

# Predict only when the form is submitted
if submitted:
    try:
        # Check if all values are valid
        if None in (area, bedrooms, bathrooms):
            st.error("⚠️ Please fill out all fields.")
        else:
            input_data = pd.DataFrame([[area, bedrooms, bathrooms]], columns=['Area', 'Bedrooms', 'Bathrooms'])
            prediction = model.predict(input_data)[0]
            st.success(f"💰 Predicted Price: ₹{int(prediction):,}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")


