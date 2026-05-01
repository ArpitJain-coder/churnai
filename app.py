import streamlit as st
import pandas as pd
import pickle
import joblib

# Page Configuration
st.set_page_config(page_title="Customer Churn Prediction", layout="wide", page_icon="📊")

# Load models
@st.cache_resource
def load_models(version="1.1"):
    model = pickle.load(open('churn_model.pkl', 'rb'))
    scaler = joblib.load('scaler (1).pkl')
    return model, scaler

try:
    model, scaler = load_models(version="1.1")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# All 30 features expected by the model in the correct order
FEATURE_NAMES = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male',
    'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check'
]

st.title("📊 Customer Churn Prediction App")
st.markdown("""
This application predicts whether a telecom customer is likely to churn or not based on their demographics, 
services subscribed, and account details.
""")

st.sidebar.header("📝 Enter Customer Information")

# 1. Demographics
st.sidebar.subheader("Demographics")
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner (Married/Living Together)", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])

# 2. Services Subscribed
st.sidebar.subheader("Services Subscribed")
phone_service = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# Dynamic options based on internet service
internet_options = ["No", "Yes"] if internet_service != "No" else ["No internet service"]
online_security = st.sidebar.selectbox("Online Security", internet_options)
online_backup = st.sidebar.selectbox("Online Backup", internet_options)
device_protection = st.sidebar.selectbox("Device Protection", internet_options)
tech_support = st.sidebar.selectbox("Tech Support", internet_options)
streaming_tv = st.sidebar.selectbox("Streaming TV", internet_options)
streaming_movies = st.sidebar.selectbox("Streaming Movies", internet_options)

# 3. Account Information
st.sidebar.subheader("Account Information")
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"
])

st.sidebar.subheader("Charges & Tenure")
tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
total_charges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, value=50.0)

# Dictionary with the raw input features mapped to dummy variables
input_data = {
    'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'gender_Male': 1 if gender == "Male" else 0,
    'Partner_Yes': 1 if partner == "Yes" else 0,
    'Dependents_Yes': 1 if dependents == "Yes" else 0,
    'PhoneService_Yes': 1 if phone_service == "Yes" else 0,
    'MultipleLines_No phone service': 1 if multiple_lines == "No phone service" else 0,
    'MultipleLines_Yes': 1 if multiple_lines == "Yes" else 0,
    'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
    'InternetService_No': 1 if internet_service == "No" else 0,
    'OnlineSecurity_No internet service': 1 if online_security == "No internet service" else 0,
    'OnlineSecurity_Yes': 1 if online_security == "Yes" else 0,
    'OnlineBackup_No internet service': 1 if online_backup == "No internet service" else 0,
    'OnlineBackup_Yes': 1 if online_backup == "Yes" else 0,
    'DeviceProtection_No internet service': 1 if device_protection == "No internet service" else 0,
    'DeviceProtection_Yes': 1 if device_protection == "Yes" else 0,
    'TechSupport_No internet service': 1 if tech_support == "No internet service" else 0,
    'TechSupport_Yes': 1 if tech_support == "Yes" else 0,
    'StreamingTV_No internet service': 1 if streaming_tv == "No internet service" else 0,
    'StreamingTV_Yes': 1 if streaming_tv == "Yes" else 0,
    'StreamingMovies_No internet service': 1 if streaming_movies == "No internet service" else 0,
    'StreamingMovies_Yes': 1 if streaming_movies == "Yes" else 0,
    'Contract_One year': 1 if contract == "One year" else 0,
    'Contract_Two year': 1 if contract == "Two year" else 0,
    'PaperlessBilling_Yes': 1 if paperless_billing == "Yes" else 0,
    'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
    'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
    'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0
}

# Ensure dataframe has exactly the right columns in the exact order
input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)

st.subheader("Data Summary Before Scaling")
st.dataframe(input_df)

# Scale numeric features
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
input_df_scaled = input_df.copy()
input_df_scaled[numeric_cols] = scaler.transform(input_df[numeric_cols])

st.write("---")
# Predict button
if st.button("🔮 Predict Churn", use_container_width=True):
    with st.spinner("Analyzing customer data..."):
        prediction = model.predict(input_df_scaled)[0]
        probability = model.predict_proba(input_df_scaled)[0][1]
        
        st.markdown("### Prediction Result")
        
        # Display results with nice formatting
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error("🚨 The customer is likely to CHURN.")
            else:
                st.success("✅ The customer is likely to STAY.")
                
        with col2:
            st.metric(label="Churn Probability", value=f"{probability * 100:.1f}%")
