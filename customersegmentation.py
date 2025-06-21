import streamlit as st
import pandas as pd
import pickle

# Load model
with open("customer_segmentation_model.pkl", "rb") as f:
    model = pickle.load(f)

# Page setup
st.set_page_config(page_title="Customer Segmentation", layout="centered")
st.title("🧠 Customer Segmentation Predictor")

st.write("Enter customer information to predict the segment.")

# Input fields
age = st.slider("Age", 18, 80, 30)
income = st.number_input("Annual Income (k$)", min_value=0, max_value=200, value=50)
score = st.slider("Spending Score (1-100)", 1, 100, 50)
gender = st.selectbox("Gender", ["Male", "Female"])
years = st.slider("Years as Customer", 0, 30, 5)

# Convert gender to numeric if needed
gender_encoded = 1 if gender.lower() == 'male' else 0

# Create input dataframe
input_data = pd.DataFrame([{
    "Age": age,
    "Annual Income (k$)": income,
    "Spending Score (1-100)": score,
    "Gender": gender_encoded,
    "Years as Customer": years
}])

# Predict
if st.button("Predict Segment"):
    prediction = model.predict(input_data)[0]
    st.success(f"📊 Predicted Customer Segment: **{prediction}**")