import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. CONFIGURATION & TITLE ---
st.set_page_config(page_title="Heart Risk Predictor", page_icon="🫀")

st.title("🫀 Heart Disease Risk Calculator")
st.markdown("""
This tool uses a **Random Forest** machine learning model to estimate the risk of heart disease.
*Adjust the sliders in the sidebar to simulate a patient.*
""")

# --- 2. LOAD THE MODEL ---
# We use @st.cache_resource so we don't reload the model every time we move a slider
@st.cache_resource
def load_model():
    try:
        return joblib.load('heart_disease_rf_model.pkl')
    except FileNotFoundError:
        st.error("Model file not found! Please run 'deploy_model.py' first to generate the .pkl file.")
        return None

model = load_model()

# --- 3. SIDEBAR: PATIENT VITALS ---
st.sidebar.header("Patient Vitals")

# We need to collect exactly the same features the model was trained on
# The order matters less for Random Forest in sklearn, but names must match!

def user_input_features():
    # Basic Info
    age = st.sidebar.slider("Age", 20, 80, 58)
    
    # Sex (Mapping text to numbers)
    sex_option = st.sidebar.selectbox("Sex", ("Male", "Female"))
    sex = 1 if sex_option == "Male" else 0
    
    # Chest Pain (The most important feature!)
    cp_option = st.sidebar.selectbox("Chest Pain Type", (
        "Typical Angina (0)", 
        "Atypical Angina (1)", 
        "Non-anginal Pain (2)", 
        "Asymptomatic (3)"
    ), index=2)
    cp = int(cp_option.split("(")[-1].strip(")"))

    # Vitals
    # UPDATED: Clarified this is SYSTOLIC and lowered the range to 80
    trestbps = st.sidebar.slider("Resting Systolic BP (Top Number)", 80, 200, 120)
    
    chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 210)
    thalach = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 130)
    
    # Labs / EKG
    fbs_option = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dl?", ("No", "Yes"))
    fbs = 1 if fbs_option == "Yes" else 0
    
    restecg = st.sidebar.selectbox("Resting ECG Results", (0, 1, 2), index=1)
    
    exang_option = st.sidebar.radio("Exercise Induced Angina?", ("No", "Yes"))
    exang = 1 if exang_option == "Yes" else 0
    
    oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.2, 1.5)
    slope = st.sidebar.selectbox("Slope of Peak Exercise ST", (0, 1, 2), index=1)
    
    ca = st.sidebar.slider("Number of Major Vessels (0-3)", 0, 3, 0)
    
    thal_option = st.sidebar.selectbox("Thalassemia", (
        "Normal (1)", 
        "Fixed Defect (2)", 
        "Reversible Defect (3)"
    ), index=1)
    thal = int(thal_option.split("(")[-1].strip(")"))

    # Create the Data Frame
    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    
    return pd.DataFrame(data, index=[0])

# Get input from user
input_df = user_input_features()

# Display input for verification
st.subheader("Patient Data Summary")
st.write(input_df)

# --- 4. PREDICTION ---
if st.button("Assess Risk"):
    if model:
        # Get Probability
        prediction_prob = model.predict_proba(input_df)[0][1]
        
        # Clinical Threshold (We optimized this to 0.3 earlier!)
        THRESHOLD = 0.3 
        is_high_risk = prediction_prob >= THRESHOLD
        
        # Display Results
        st.markdown("---")
        st.subheader("Assessment Result")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Risk Probability", value=f"{prediction_prob:.1%}")
            
        with col2:
            if is_high_risk:
                st.error("**HIGH RISK**")
                st.write("Recommendation: **Refer to Cardiology**")
            else:
                st.success("**LOW RISK**")
                st.write("Recommendation: **Standard Follow-up**")
                
        # Visual Bar
        st.progress(int(prediction_prob * 100))
        st.caption(f"Note: This tool uses a high-sensitivity threshold of {THRESHOLD*100}%.")