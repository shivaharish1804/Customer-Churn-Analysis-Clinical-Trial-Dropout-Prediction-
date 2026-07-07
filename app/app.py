"""
Clinical Trial Dropout Risk Dashboard
Run: streamlit run app/app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
import os

st.set_page_config(page_title="Trial Dropout Risk", page_icon="🏥", layout="wide")

st.title("🏥 Clinical Trial Dropout Risk Predictor")
st.markdown("Predict a patient's dropout risk and understand the key drivers.")

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/xgb_dropout_model.pkl")
ENCODERS_PATH = os.path.join(os.path.dirname(__file__), "../models/label_encoders.pkl")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "../models/feature_names.pkl")

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    return model, encoders, feature_names

try:
    model, encoders, feature_names = load_model()
    model_loaded = True
except Exception as e:
    st.warning(f"Model not found. Run the notebook first to train it. ({e})")
    model_loaded = False

# Sidebar — patient input
st.sidebar.header("Patient & Trial Details")

age = st.sidebar.slider("Age", 18, 85, 52)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
bmi = st.sidebar.slider("BMI", 16.0, 50.0, 27.5)
trial_phase = st.sidebar.selectbox("Trial Phase", ["Phase 1","Phase 2","Phase 3","Phase 4"])
trial_duration = st.sidebar.slider("Trial Duration (months)", 1, 60, 12)
therapeutic_area = st.sidebar.selectbox("Therapeutic Area",
    ["Oncology","Cardiology","Neurology","Infectious Disease","Metabolic"])
num_visits = st.sidebar.slider("Number of Visits Required", 1, 25, 8)
has_placebo = st.sidebar.checkbox("Has Placebo Arm", value=True)
procedures_per_visit = st.sidebar.slider("Procedures per Visit", 1, 8, 3)
distance_km = st.sidebar.slider("Distance to Site (km)", 1, 300, 30)
comorbidities = st.sidebar.slider("Number of Comorbidities", 0, 6, 2)
prior_trial = st.sidebar.checkbox("Prior Trial Participation")
employment = st.sidebar.selectbox("Employment Status",
    ["Employed","Unemployed","Retired","Student"])
lives_alone = st.sidebar.checkbox("Lives Alone")
disease_severity = st.sidebar.selectbox("Disease Severity", ["Mild","Moderate","Severe"])
adverse_event = st.sidebar.checkbox("Early Adverse Event")
burden_score = st.sidebar.slider("Patient Reported Burden Score (1-10)", 1, 10, 5)

if model_loaded:
    # Build input
    raw = {
        'age': age, 'sex': sex, 'bmi': bmi,
        'trial_phase': trial_phase, 'trial_duration_months': trial_duration,
        'therapeutic_area': therapeutic_area, 'num_visits_required': num_visits,
        'has_placebo_arm': int(has_placebo), 'num_procedures_per_visit': procedures_per_visit,
        'distance_to_site_km': distance_km, 'num_comorbidities': comorbidities,
        'prior_trial_participation': int(prior_trial), 'employment_status': employment,
        'lives_alone': int(lives_alone), 'disease_severity': disease_severity,
        'early_adverse_event': int(adverse_event),
        'patient_reported_burden_score': burden_score
    }

    df_input = pd.DataFrame([raw])
    cat_cols = ['sex','trial_phase','therapeutic_area','employment_status','disease_severity']
    for col in cat_cols:
        df_input[col] = encoders[col].transform(df_input[col])

    df_input = df_input[feature_names]
    dropout_prob = model.predict_proba(df_input)[:, 1][0]

    # Risk tier
    if dropout_prob < 0.33:
        tier, color, emoji = "Low Risk", "green", "🟢"
    elif dropout_prob < 0.66:
        tier, color, emoji = "Medium Risk", "orange", "🟡"
    else:
        tier, color, emoji = "High Risk", "red", "🔴"

    col1, col2, col3 = st.columns(3)
    col1.metric("Dropout Probability", f"{dropout_prob:.1%}")
    col2.metric("Risk Tier", f"{emoji} {tier}")
    col3.metric("Retention Probability", f"{1-dropout_prob:.1%}")

    st.markdown("---")

    # SHAP waterfall
    st.subheader("Why is this patient at risk?")
    explainer = shap.TreeExplainer(model)
    shap_exp = explainer(df_input)

    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(shap_exp[0], max_display=10, show=False)
    st.pyplot(plt.gcf())
    plt.close()

    # What-if distance
    st.subheader("What-If: Change Distance to Site")
    distances = np.arange(5, 151, 5)
    probs = []
    for d in distances:
        p = df_input.copy()
        p['distance_to_site_km'] = d
        probs.append(model.predict_proba(p)[:, 1][0])

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(distances, [p*100 for p in probs], color='#e74c3c', linewidth=2, marker='o', markersize=3)
    ax2.axvline(distance_km, color='gray', linestyle='--', label=f'Current: {distance_km}km')
    ax2.axhline(50, color='orange', linestyle='--', alpha=0.7, label='50% threshold')
    ax2.set_xlabel('Distance to Site (km)')
    ax2.set_ylabel('Dropout Risk (%)')
    ax2.set_title('Sensitivity to Travel Distance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    plt.close()
