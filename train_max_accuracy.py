import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])

y = (df['Churn'] == 'Yes').astype(int)
df_features = df.drop(columns=['customerID', 'Churn'])

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Create XGBoost model
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

# Broad param grid aiming for pure accuracy
param_distributions = {
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 200, 300],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.5],
    'min_child_weight': [1, 3, 5]
}

random_search = RandomizedSearchCV(
    estimator=xgb_model, 
    param_distributions=param_distributions, 
    n_iter=50, 
    scoring='accuracy', 
    cv=5, 
    verbose=1, 
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train_scaled, y_train)
best_model = random_search.best_estimator_

y_pred_test = best_model.predict(X_test_scaled)
acc_test = accuracy_score(y_test, y_pred_test)

X_all_scaled = X.copy()
X_all_scaled[numeric_cols] = scaler.transform(X[numeric_cols])
y_all_pred = best_model.predict(X_all_scaled)
acc_all = accuracy_score(y, y_all_pred)

print(f"\nTest Accuracy: {acc_test:.4f}")
print(f"Full Dataset Accuracy: {acc_all:.4f}")

with open('churn_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
joblib.dump(scaler, 'scaler (1).pkl')

print("Saved highest accuracy model.")
