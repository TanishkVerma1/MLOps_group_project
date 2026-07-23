import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📉", layout="wide")

# Download the model and its matching threshold from the Model Hub
model_path = hf_hub_download(repo_id="TanishkV18/churn-model", filename="best_churn_model.joblib")
threshold_path = hf_hub_download(repo_id="TanishkV18/churn-model", filename="threshold.txt")

# Load the model and threshold
model = joblib.load(model_path)
with open(threshold_path) as f:
    classification_threshold = float(f.read().strip())

with st.sidebar:
    st.header("About")
    st.write("Predicts whether a telecom customer is likely to churn, using an XGBoost model trained on historical account data.")
    st.metric("Model", "XGBoost")
    st.metric("Decision threshold", f"{classification_threshold:.2f}")

# Streamlit UI for Customer Churn Prediction
st.title("📉 Telco Customer Churn Prediction")
st.caption("Internal retention tool — enter a customer's account and service details to estimate churn risk.")
st.divider()

# Collect user input
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen?", ["Yes", "No"])
    Partner = st.selectbox("Has Partner?", ["Yes", "No"])
    Dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)

with col2:
    st.subheader("Services")
    PhoneService = st.selectbox("Phone Service?", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines?", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security?", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup?", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection?", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support?", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV?", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies?", ["Yes", "No", "No internet service"])

with col3:
    st.subheader("Billing")
    Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing?", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
    TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, value=840.0)

# Build the input row to match model training
input_data = pd.DataFrame([{
    'gender': gender,
    'SeniorCitizen': 1 if SeniorCitizen == "Yes" else 0,
    'Partner': Partner,
    'Dependents': Dependents,
    'tenure': tenure,
    'PhoneService': PhoneService,
    'MultipleLines': MultipleLines,
    'InternetService': InternetService,
    'OnlineSecurity': OnlineSecurity,
    'OnlineBackup': OnlineBackup,
    'DeviceProtection': DeviceProtection,
    'TechSupport': TechSupport,
    'StreamingTV': StreamingTV,
    'StreamingMovies': StreamingMovies,
    'Contract': Contract,
    'PaperlessBilling': PaperlessBilling,
    'PaymentMethod': PaymentMethod,
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges,
}])



with st.expander("Review entered details"):
    st.dataframe(input_data.T.rename(columns={0: "Value"}))
    
# Predict button
st.divider()
predict_col, _ = st.columns([1, 3])
with predict_col:
    run = st.button("🔍 Predict Churn Risk", use_container_width=True, type="primary")

if run:
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = int(prediction_proba >= classification_threshold)

    if prediction_proba >= 0.7:
        risk_tier, color = "High Risk", "red"
    elif prediction_proba >= classification_threshold:
        risk_tier, color = "Medium Risk", "orange"
    else:
        risk_tier, color = "Low Risk", "green"

    result_col1, result_col2 = st.columns(2)
    with result_col1:
        st.metric("Churn Probability", f"{prediction_proba:.1%}")
        st.progress(min(prediction_proba, 1.0))
    with result_col2:
        st.markdown(f"### :{color}[{risk_tier}]")
        verdict = "likely to churn" if prediction == 1 else "not likely to churn"
        st.write(f"This customer is **{verdict}**.")
