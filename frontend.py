import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained pipeline
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# Define preprocessing functions exactly as in your notebook
@st.cache_data
def age_group(age):
    if age < 25:
        return 'young'
    elif age < 45:
        return 'adult'
    elif age < 60:
        return 'middleaged'
    else:
        return 'senior'

@st.cache_data
def bmi(weight, height):
    return weight / (height ** 2)

@st.cache_data
def lifestyle_risk(smoker, bmi_val):
    if smoker and bmi_val >= 30:
        return 'high'
    elif smoker or bmi_val >= 27:
        return 'medium'
    else:
        return 'low'

tier1_cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune']
tier2_cities = ['Jaipur', 'Chandigarh', 'Indore', 'Lucknow', 'Patna', 'Ranchi', 'Visakhapatnam', 
                'Coimbatore', 'Bhopal', 'Nagpur', 'Vadodara', 'Surat', 'Rajkot', 'Jodhpur', 
                'Raipur', 'Amritsar', 'Varanasi', 'Agra', 'Dehradun', 'Mysore', 'Jabalpur', 
                'Guwahati', 'Thiruvananthapuram', 'Ludhiana', 'Nashik', 'Allahabad', 'Udaipur', 
                'Aurangabad', 'Hubli', 'Belgaum', 'Salem', 'Vijayawada', 'Tiruchirappalli', 
                'Bhavnagar', 'Gwalior', 'Dhanbad', 'Bareilly', 'Aligarh', 'Gaya', 'Kozhikode', 
                'Warangal', 'Kolhapur', 'Bilaspur', 'Jalandhar', 'Noida', 'Guntur', 'Asansol', 'Siliguri']

@st.cache_data
def city_tier(city):
    if city in tier1_cities:
        return 1
    elif city in tier2_cities:
        return 2
    else:
        return 3

# Streamlit UI
st.title("🚑 Insurance Premium Category Predictor")
st.markdown("Enter details matching your training data (age, weight, height, income in LPA, smoker status, city, occupation).")

# Input fields based on original raw data
age = st.number_input("Age", min_value=0, max_value=100, value=30)
weight = st.number_input("Weight (kg)", min_value=0.0, value=60.0)
height = st.number_input("Height (m)", min_value=0.0, value=1.7)
income_lpa = st.number_input("Income (LPA)", min_value=0.0, value=5.0)
smoker = st.checkbox("Smoker")
city = st.selectbox("City", options=['Mumbai', 'Delhi', 'Bangalore', 'Indore', 'Jaipur', 'Lucknow'] + tier2_cities[:10])  # Partial list for UI
occupation = st.selectbox("Occupation", options=['retired', 'freelancer', 'student', 'governmentjob', 'businessowner', 'unemployed', 'privatejob'])

# Compute derived features
bmi_val = bmi(weight, height)
agegroup = age_group(age)
lifestylerisk = lifestyle_risk(smoker, bmi_val)
citytier = city_tier(city)

# Display derived features
with st.expander("View Derived Features"):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("BMI", f"{bmi_val:.2f}")
    col2.metric("Age Group", agegroup)
    col3.metric("Lifestyle Risk", lifestylerisk)
    col4.metric("City Tier", citytier)

# Prepare input DataFrame matching training format
input_data = pd.DataFrame({
    'bmi': [bmi_val],
    'agegroup': [agegroup],
    'lifestylerisk': [lifestylerisk],
    'citytier': [citytier],
    'incomelpa': [income_lpa],
    'occupation': [occupation]
})

if st.button("🔮 Predict Premium Category", type="primary"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    st.success(f"**Predicted Category: {prediction}**")
    
    # Show probabilities
    st.subheader("Prediction Probabilities")
    probs_df = pd.DataFrame({
        'Category': model.classes_,
        'Probability': probability
    }).round(3)
    st.dataframe(probs_df, use_container_width=True)
    
    # Example interpretation
    st.info("**Categories**: Low (affordable), Medium, High (expensive premium).")

# Sidebar with model info
with st.sidebar:
    st.header("Model Details")
    st.write("- **Algorithm**: RandomForestClassifier (Pipeline with OHE + passthrough)")
    st.write("- **Features**: bmi, agegroup, lifestylerisk, citytier, incomelpa, occupation")
    st.write("- **Target**: insurancepremiumcategory")
    st.write("- **Accuracy**: ~0.90 on test set [from notebook]")
