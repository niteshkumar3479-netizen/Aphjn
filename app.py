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
    weight = st.number_input("Weight (kg)", 30.0, 150.0, 65.0)
    height = st.number_input("Height (m)", 1.0, 2.5, 1.7, 0.01)
    smoker = st.checkbox("Smoker")

with col2:
    income = st.number_input("Income (LPA)", 0.0, 100.0, 5.0)
    occupation = st.selectbox("Occupation", 
        ['retired', 'freelancer', 'student', 'governmentjob', 
         'businessowner', 'unemployed', 'privatejob'])
    city = st.selectbox("City", tier1_cities + tier2_cities[:8])

bmi_val = bmi(weight, height)
agegroup = age_group(age)
risk = lifestyle_risk(smoker, bmi_val)
citytier = city_tier(city)

col1, col2, col3, col4 = st.columns(4)
col1.metric("BMI", f"{bmi_val:.1f}")
col2.metric("Age Group", agegroup.title())
col3.metric("Risk", risk.title())
col4.metric("City Tier", citytier)

input_df = pd.DataFrame({
    'bmi': [bmi_val],
    'agegroup': [agegroup],
    'lifestylerisk': [risk],
    'citytier': [citytier],
    'incomelpa': [income],
    'occupation': [occupation]
}, columns=['bmi', 'agegroup', 'lifestylerisk', 'citytier', 'incomelpa', 'occupation'])

with st.expander("Debug: Input DataFrame"):
    st.dataframe(input_df)

if st.button("Predict Premium Category", type="primary"):
    try:
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        
        st.success(f"Predicted: {prediction}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence", f"{max(probabilities):.1%}")
        with col2:
            st.bar_chart(dict(zip(model.classes_, probabilities)))
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

with st.sidebar:
    st.info("Features Used:\n- BMI\n- Age Group\n- Lifestyle Risk\n- City Tier\n- Income LPA\n- Occupation")
