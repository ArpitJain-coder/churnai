import pandas as pd
import pickle
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading models...")
model = pickle.load(open('churn_model.pkl', 'rb'))
scaler = joblib.load('scaler (1).pkl')

print("Loading data...")
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Preprocessing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])

# Target variable
y_true = (df['Churn'] == 'Yes').astype(int)
df_features = df.drop(columns=['customerID', 'Churn'])

# Convert to dummies keeping the required columns
df_features['gender_Male'] = (df_features['gender'] == 'Male').astype(int)
df_features['Partner_Yes'] = (df_features['Partner'] == 'Yes').astype(int)
df_features['Dependents_Yes'] = (df_features['Dependents'] == 'Yes').astype(int)
df_features['PhoneService_Yes'] = (df_features['PhoneService'] == 'Yes').astype(int)
df_features['MultipleLines_No phone service'] = (df_features['MultipleLines'] == 'No phone service').astype(int)
df_features['MultipleLines_Yes'] = (df_features['MultipleLines'] == 'Yes').astype(int)
df_features['InternetService_Fiber optic'] = (df_features['InternetService'] == 'Fiber optic').astype(int)
df_features['InternetService_No'] = (df_features['InternetService'] == 'No').astype(int)
df_features['OnlineSecurity_No internet service'] = (df_features['OnlineSecurity'] == 'No internet service').astype(int)
df_features['OnlineSecurity_Yes'] = (df_features['OnlineSecurity'] == 'Yes').astype(int)
df_features['OnlineBackup_No internet service'] = (df_features['OnlineBackup'] == 'No internet service').astype(int)
df_features['OnlineBackup_Yes'] = (df_features['OnlineBackup'] == 'Yes').astype(int)
df_features['DeviceProtection_No internet service'] = (df_features['DeviceProtection'] == 'No internet service').astype(int)
df_features['DeviceProtection_Yes'] = (df_features['DeviceProtection'] == 'Yes').astype(int)
df_features['TechSupport_No internet service'] = (df_features['TechSupport'] == 'No internet service').astype(int)
df_features['TechSupport_Yes'] = (df_features['TechSupport'] == 'Yes').astype(int)
df_features['StreamingTV_No internet service'] = (df_features['StreamingTV'] == 'No internet service').astype(int)
df_features['StreamingTV_Yes'] = (df_features['StreamingTV'] == 'Yes').astype(int)
df_features['StreamingMovies_No internet service'] = (df_features['StreamingMovies'] == 'No internet service').astype(int)
df_features['StreamingMovies_Yes'] = (df_features['StreamingMovies'] == 'Yes').astype(int)
df_features['Contract_One year'] = (df_features['Contract'] == 'One year').astype(int)
df_features['Contract_Two year'] = (df_features['Contract'] == 'Two year').astype(int)
df_features['PaperlessBilling_Yes'] = (df_features['PaperlessBilling'] == 'Yes').astype(int)
df_features['PaymentMethod_Credit card (automatic)'] = (df_features['PaymentMethod'] == 'Credit card (automatic)').astype(int)
df_features['PaymentMethod_Electronic check'] = (df_features['PaymentMethod'] == 'Electronic check').astype(int)
df_features['PaymentMethod_Mailed check'] = (df_features['PaymentMethod'] == 'Mailed check').astype(int)

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

X = df_features[FEATURE_NAMES]

# Scale numeric columns
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
X_scaled = X.copy()
X_scaled[numeric_cols] = scaler.transform(X[numeric_cols])

print("Making predictions...")
y_pred = model.predict(X_scaled)

print("\n--- Accuracy Report ---")
print(f"Accuracy Score: {accuracy_score(y_true, y_pred) * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_true, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
