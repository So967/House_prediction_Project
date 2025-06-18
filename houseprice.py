import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# Set page configuration
st.set_page_config(page_title="House Price Predictor",
                   layout="wide",
                   page_icon="🏠")

# Load the trained model
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "house_price_model.pkl")
    return joblib.load(model_path)

model = load_model()

# Streamlit App UI
def main():
    st.title("🏠 House Price Predictor")
    st.markdown("Enter the details below to estimate the house price:")

    # Input fields in columns for responsiveness
    col1, col2, col3 = st.columns(3)

    with col1:
        area = st.text_input("📐 Area (in sq ft)", value="1500")

    with col2:
        bedrooms = st.text_input("🛏️ Bedrooms", value="3")

    with col3:
        bathrooms = st.text_input("🛁 Bathrooms", value="2")

    prediction_result = ''

    if st.button("Predict Price"):
        try:
            # Convert input to appropriate types
            area = float(area)
            bedrooms = int(bedrooms)
            bathrooms = int(bathrooms)

            input_data = pd.DataFrame([[area, bedrooms, bathrooms]],
                                      columns=['Area', 'Bedrooms', 'Bathrooms'])

            prediction = model.predict(input_data)[0]
            prediction_result = f"💰 Predicted Price: ₹{int(prediction):,}"

        except ValueError:
            prediction_result = "⚠️ Please enter valid numeric values."

        except Exception as e:
            prediction_result = f"❌ Prediction failed: {str(e)}"

    st.success(prediction_result)

# Run the app
if __name__ == '__main__':
    main()




