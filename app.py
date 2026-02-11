import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Insurance Predictor", layout="wide")

# FIXED: Check all possible model locations
@st.cache_resource
def load_model():
    possible_paths = ['model.pkl']
    
    for path in possible_paths:
        if os.path.exists(path):
            model = joblib.load(path)
            st.success(f"Model loaded from: {path}")
            return model
    
    st.error("❌ NO MODEL FILE FOUND!")
    st.info("""
    **Fix this:**
    1. In your notebook, run: `joblib.dump(pipeline, 'insurance_model.joblib')`
    2. Upload `insurance_model.joblib` to GitHub repo root
    3. Redeploy
    """)
    st.stop()

# Try to load model
try:
    model = load_model()
except:
    st.stop()

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
    weight = st.number_input("Weight (kg)", 30.0, 150.0, 65.0)
    height = st.number_input("Height (m)", 1.0, 2.5, 1.7, 0.01)
    smoker = st.checkbox("Smoker")

with col2:
    income = st.number_input("Income (LPA)", 0.0, 100.0, 5.0)
    occupation = st.selectbox("Occupation", 
        ['retired', 'freelancer', 'student', 'governmentjob', 
         'businessowner', 'unemployed', 'privatejob'])
    city = st.selectbox("City", tier1_cities + tier2_cities[:8])

# Calculate features
bmi_val = bmi(weight, height)
agegroup = age_group(age)
risk = lifestyle_risk(smoker, bmi_val)
citytier = city_tier(city)

col1, col2, col3, col4 = st.columns(4)
col1.metric("BMI", f"{bmi_val:.1f}")
col2.metric("Age Group", agegroup.title())
col3.metric("Risk", risk.title())
col4.metric("City Tier", citytier)

# ✅ FIXED: EXACT column names your model expects (snake_case)
input_df = pd.DataFrame({
    'bmi': [bmi_val],
    'age_group': [agegroup],      # ← FIXED: was 'agegroup'
    'lifestyle_risk': [risk],     # ← FIXED: was 'risk' 
    'city_tier': [citytier],      # ← FIXED: was 'citytier'
    'income_lpa': [income],       # ← FIXED: was 'income'
    'occupation': [occupation]
})

with st.expander("Debug: Input DataFrame"):
    st.dataframe(input_df)

if st.button("Predict Premium Category", type="primary"):
    try:
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        
        st.success(f"Predicted Category: {prediction}")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Highest Confidence", f"{max(probabilities):.1%}")
        with col2:
            st.bar_chart(dict(zip(model.classes_, probabilities)))
            
        st.info(f"All probabilities: {dict(zip(model.classes_, [f'{p:.1%}' for p in probabilities]))}")
        
    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
        st.code(str(e))

with st.sidebar:
    st.info("Model Features:\n• bmi\n• age_group\n• lifestyle_risk\n• city_tier\n• income_lpa\n• occupation")

