from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import pickle
import joblib
import os
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
import datetime
import sqlite3

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

# Initialize Supabase Client
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Failed to initialize Supabase: {e}")

# Fallback to Local SQLite if Supabase is not available
LOCAL_DB_PATH = os.path.join(base_dir, 'local_history.db')
if not supabase:
    try:
        conn = sqlite3.connect(LOCAL_DB_PATH)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS churn_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gender TEXT,
                tenure INTEGER,
                monthly_charges REAL,
                contract TEXT,
                prediction_result TEXT,
                churn_probability REAL,
                created_at TEXT
            )
        ''')
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Failed to initialize SQLite: {e}")

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

@app.get("/")
async def serve_frontend():
    import os
    base_dir = os.path.dirname(os.path.dirname(__file__))
    html_path = os.path.join(base_dir, 'index.html')
    if not os.path.exists(html_path):
        html_path = "index.html"
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

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
        
        result_payload = {
            "prediction": prediction,
            "probability": probability,
            "status": "success"
        }

        # Log to Supabase or SQLite
        db_payload = {
            "gender": data.gender,
            "tenure": data.tenure,
            "monthly_charges": data.MonthlyCharges,
            "contract": data.Contract,
            "prediction_result": "Churn" if prediction == 1 else "Stay",
            "churn_probability": probability,
            "created_at": datetime.datetime.utcnow().isoformat()
        }

        if supabase:
            try:
                supabase.table("churn_predictions").insert(db_payload).execute()
            except Exception as e:
                print(f"Supabase logging failed: {e}")
        else:
            try:
                conn = sqlite3.connect(LOCAL_DB_PATH)
                c = conn.cursor()
                c.execute('''
                    INSERT INTO churn_predictions (gender, tenure, monthly_charges, contract, prediction_result, churn_probability, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (db_payload["gender"], db_payload["tenure"], db_payload["monthly_charges"], db_payload["contract"], db_payload["prediction_result"], db_payload["churn_probability"], db_payload["created_at"]))
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"SQLite logging failed: {e}")

        return result_payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
async def get_history():
    if supabase:
        try:
            response = supabase.table("churn_predictions").select("*").order("created_at", desc=True).limit(5).execute()
            return {"data": response.data, "message": "success", "source": "supabase"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        try:
            conn = sqlite3.connect(LOCAL_DB_PATH)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute('SELECT * FROM churn_predictions ORDER BY created_at DESC LIMIT 5')
            rows = c.fetchall()
            conn.close()
            data = [dict(ix) for ix in rows]
            return {"data": data, "message": "success", "source": "sqlite"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/features")
async def get_features():
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_impacts = [{"feature": f.replace('_', ' ').title(), "impact": float(imp)} for f, imp in zip(FEATURE_NAMES, importances)]
            feature_impacts.sort(key=lambda x: x["impact"], reverse=True)
            return {"data": feature_impacts[:8], "message": "success"}
        return {"data": [], "message": "Model has no feature importances"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    try:
        if supabase:
            res = supabase.table("churn_predictions").select("*").execute()
            data = res.data
        else:
            conn = sqlite3.connect(LOCAL_DB_PATH)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute('SELECT * FROM churn_predictions')
            rows = c.fetchall()
            conn.close()
            data = [dict(ix) for ix in rows]

        if not data:
            return {
                "churn_rate": "0.0%",
                "active_customers": "0",
                "avg_ltv": "$0",
                "history": [],
                "source": "supabase" if supabase else "sqlite"
            }
        
        total = len(data)
        churns = sum(1 for d in data if d.get('prediction_result') == 'Churn')
        churn_rate = (churns / total) * 100
        
        total_ltv = sum(float(d.get('monthly_charges', 0)) * int(d.get('tenure', 0)) for d in data)
        avg_ltv = total_ltv / total
        
        # Calculate monthly history for charts
        history_data = []
        for d in data[-20:]: # last 20 for chart
             history_data.append({
                 "date": d.get('created_at', '')[:10],
                 "prob": d.get('churn_probability', 0),
                 "result": d.get('prediction_result', '')
             })

        return {
            "churn_rate": f"{churn_rate:.1f}%",
            "active_customers": f"{total}",
            "avg_ltv": f"${avg_ltv:,.0f}",
            "history": history_data,
            "source": "supabase" if supabase else "sqlite"
        }
    except Exception as e:
        print("Stats error:", e)
        return {
            "churn_rate": "0.0%",
            "active_customers": "0",
            "avg_ltv": "$0",
            "history": [],
            "source": "error"
        }
