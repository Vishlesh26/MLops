import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page configuration for a premium look
st.set_page_config(
    page_title="Heart Disease Predictor",
    
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for glassmorphism and premium feel
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
    }
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {
        background-color: #262730;
        color: white;
        border-radius: 8px;
    }
    .prediction-card {
        padding: 30px;
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 20px;
    }
    h1, h2, h3 {
        color: #ff4b4b !important;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.title("Heart Disease Prediction System")
st.markdown("Enter medical details below to analyze heart disease risk.")

# Load Model
@st.cache_resource
def load_model():
    model = joblib.load("Model/xgb_tuned.pkl")
    # Force CPU for inference to avoid device mismatch warnings
    if hasattr(model, "set_params"):
        try:
            model.set_params(device="cpu")
        except:
            pass
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Input Form in Columns
with st.container():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👤 Patient Info")
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
        cp = st.selectbox("Chest Pain Type", options=[(1, 1), (2, 2), (3, 3), (4, 4)], format_func=lambda x: f"Type {x[0]}")[1]
        bp = st.number_input("Blood Pressure (BP)", min_value=50, max_value=250, value=120)

    with col2:
        st.subheader("🧪 Clinical Tests")
        chol = st.number_input("Cholesterol", min_value=50, max_value=600, value=200)
        fbs = st.selectbox("FBS over 120", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        ekg = st.selectbox("EKG Results", options=[0, 1, 2])
        max_hr = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=150)

    with col3:
        st.subheader("🏃 Exercise & ST")
        ex_angina = st.selectbox("Exercise Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        st_dep = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        slope = st.selectbox("Slope of ST", options=[1, 2, 3])
        vessels = st.selectbox("Number of Vessels (Fluoroscopy)", options=[0, 1, 2, 3])
        thallium = st.selectbox("Thallium", options=[3, 6, 7])

# Prediction Logic
if st.button("Predict Heart Disease Risk"):
    # Prepare Input Data
    # Feature Names based on training: ['id', 'Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina', 'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium']
    input_data = pd.DataFrame({
        'id': [0], # Placeholder ID
        'Age': [age],
        'Sex': [sex],
        'Chest pain type': [cp],
        'BP': [bp],
        'Cholesterol': [chol],
        'FBS over 120': [fbs],
        'EKG results': [ekg],
        'Max HR': [max_hr],
        'Exercise angina': [ex_angina],
        'ST depression': [st_dep],
        'Slope of ST': [slope],
        'Number of vessels fluro': [vessels],
        'Thallium': [thallium]
    })
    
    # Perform Prediction
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    # Display Result
    st.markdown("---")
    if prediction == 1:
        st.markdown(f"""
        <div class="prediction-card" style="border-left: 10px solid #ff4b4b;">
            <h2>High Risk Detected</h2>
            <p>The model predicts a <b>Presence</b> of Heart Disease.</p>
            <p>Confidence Level: <b>{probabilities[1]*100:.2f}%</b></p>
            <small>Please consult with a medical professional for a formal diagnosis.</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-card" style="border-left: 10px solid #28a745;">
            <h2 style="color: #28a745 !important;">✅ Low Risk Detected</h2>
            <p>The model predicts an <b>Absence</b> of Heart Disease.</p>
            <p>Confidence Level: <b>{probabilities[0]*100:.2f}%</b></p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
