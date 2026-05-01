from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import joblib
import os
from fastapi.middleware.cors import CORSMiddleware
import xgboost as xgb

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models from parent directory (Vercel sets cwd differently sometimes)
base_dir = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(base_dir, 'churn_model.pkl')
scaler_path = os.path.join(base_dir, 'scaler (1).pkl')

try:
    model = pickle.load(open(model_path, 'rb'))
    scaler = joblib.load(scaler_path)
except Exception:
    # Try current dir just in case
    model = pickle.load(open('churn_model.pkl', 'rb'))
    scaler = joblib.load('scaler (1).pkl')

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    tenure: int
    MonthlyCharges: float
    TotalCharges: float

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

@app.post("/api/predict")
async def predict(data: CustomerData):
    try:
        input_data = {
            'SeniorCitizen': 1 if data.SeniorCitizen == "Yes" else 0,
            'tenure': data.tenure,
            'MonthlyCharges': data.MonthlyCharges,
            'TotalCharges': data.TotalCharges,
            'gender_Male': 1 if data.gender == "Male" else 0,
            'Partner_Yes': 1 if data.Partner == "Yes" else 0,
            'Dependents_Yes': 1 if data.Dependents == "Yes" else 0,
            'PhoneService_Yes': 1 if data.PhoneService == "Yes" else 0,
            'MultipleLines_No phone service': 1 if data.MultipleLines == "No phone service" else 0,
            'MultipleLines_Yes': 1 if data.MultipleLines == "Yes" else 0,
            'InternetService_Fiber optic': 1 if data.InternetService == "Fiber optic" else 0,
            'InternetService_No': 1 if data.InternetService == "No" else 0,
            'OnlineSecurity_No internet service': 1 if data.OnlineSecurity == "No internet service" else 0,
            'OnlineSecurity_Yes': 1 if data.OnlineSecurity == "Yes" else 0,
            'OnlineBackup_No internet service': 1 if data.OnlineBackup == "No internet service" else 0,
            'OnlineBackup_Yes': 1 if data.OnlineBackup == "Yes" else 0,
            'DeviceProtection_No internet service': 1 if data.DeviceProtection == "No internet service" else 0,
            'DeviceProtection_Yes': 1 if data.DeviceProtection == "Yes" else 0,
            'TechSupport_No internet service': 1 if data.TechSupport == "No internet service" else 0,
            'TechSupport_Yes': 1 if data.TechSupport == "Yes" else 0,
            'StreamingTV_No internet service': 1 if data.StreamingTV == "No internet service" else 0,
            'StreamingTV_Yes': 1 if data.StreamingTV == "Yes" else 0,
            'StreamingMovies_No internet service': 1 if data.StreamingMovies == "No internet service" else 0,
            'StreamingMovies_Yes': 1 if data.StreamingMovies == "Yes" else 0,
            'Contract_One year': 1 if data.Contract == "One year" else 0,
            'Contract_Two year': 1 if data.Contract == "Two year" else 0,
            'PaperlessBilling_Yes': 1 if data.PaperlessBilling == "Yes" else 0,
            'PaymentMethod_Credit card (automatic)': 1 if data.PaymentMethod == "Credit card (automatic)" else 0,
            'PaymentMethod_Electronic check': 1 if data.PaymentMethod == "Electronic check" else 0,
            'PaymentMethod_Mailed check': 1 if data.PaymentMethod == "Mailed check" else 0
        }

        df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
        
        numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        df[numeric_cols] = scaler.transform(df[numeric_cols])

        prediction = int(model.predict(df)[0])
        probability = float(model.predict_proba(df)[0][1])

        return {
            "prediction": prediction,
            "probability": probability,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
