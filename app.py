import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Insurance Predictor", layout="wide")

@st.cache_resource
def load_model():
    if os.path.exists('model.pkl'):
        return joblib.load('insurance_model.joblib')
    st.error("Model file 'insurance_model.joblib' not found!")
    st.stop()

model = load_model()

def age_group(age):
    if age < 25: return 'young'
    elif age < 45: return 'adult'
    elif age < 60: return 'middleaged'
    return 'senior'

def bmi(weight, height): 
    return weight / (height ** 2)

def lifestyle_risk(smoker, bmi_val):
    if smoker and bmi_val >= 30: return 'high'
    elif smoker or bmi_val >= 27: return 'medium'
    return 'low'

tier1_cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune']
tier2_cities = ['Jaipur', 'Chandigarh', 'Indore', 'Lucknow', 'Patna', 'Ranchi', 'Visakhapatnam', 
                'Coimbatore', 'Bhopal', 'Nagpur', 'Vadodara', 'Surat', 'Rajkot', 'Jodhpur', 'Raipur']

def city_tier(city):
    if city in tier1_cities: return 1
    elif city in tier2_cities: return 2
    return 3

st.title("Insurance Premium Predictor")
st.markdown("90% Accurate Random Forest Model")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 100, 30)
    weight = st.number_input("Weight (kg)", 30.0, 150.0, 6
