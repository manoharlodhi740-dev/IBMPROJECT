import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Page configuration
st.set_page_config(page_title="CardioGuard - AI Cardiac Risk Predictor", layout="centered")

st.title("CardioGuard: Early Heart Disease Predictor")
st.markdown("**Aligned with UN SDG 3: Good Health and Well-being**")
st.write("Enter clinical parameters to predict early cardiovascular risk.")

# Train synthetic baseline model
@st.cache_resource
def get_trained_model():
    np.random.seed(42)
    X = np.random.randn(1000, 5)
    X[:, 0] = np.random.randint(25, 80, 1000)      # Age
    X[:, 1] = np.random.randint(90, 190, 1000)     # Resting BP
    X[:, 2] = np.random.randint(120, 350, 1000)    # Serum Cholesterol
    X[:, 3] = np.random.choice([0, 1], size=1000)  # Fasting Blood Sugar > 120 mg/dl
    X[:, 4] = np.random.randint(70, 210, 1000)     # Max Heart Rate
    
    # Target label: 1 (High Risk), 0 (Normal)
    y = ((X[:, 0] > 52) & (X[:, 1] > 135) | (X[:, 2] > 240)).astype(int)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf

model = get_trained_model()

# Form Inputs
st.subheader("Patient Clinical Data")
c1, c2 = st.columns(2)

with c1:
    age = st.slider("Age (Years)", 20, 85, 45)
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
    chol = st.slider("Serum Cholesterol (mg/dl)", 120, 400, 200)

with c2:
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [("No", 0), ("Yes", 1)])
    thalach = st.slider("Max Heart Rate Achieved", 70, 220, 150)

# Prediction Trigger
if st.button("Analyze Cardiac Risk", type="primary"):
    features = np.array([[age, trestbps, chol, fbs[1], thalach]])
    risk_prob = model.predict_proba(features)[0][1]
    
    st.divider()
    if risk_prob >= 0.5:
        st.error(f"High Risk Detected: {risk_prob*100:.1f}% estimated probability.")
        st.info("Recommendations: Schedule a formal ECG & lipid profile with a certified physician.")
    else:
        st.success(f"Low/Normal Risk: {(1 - risk_prob)*100:.1f}% safety confidence.")
        st.write("Maintain healthy lifestyle, regular exercise, and annual preventive check-ups.")