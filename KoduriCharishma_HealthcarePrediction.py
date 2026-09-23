"""
train_model.py
Trains three ML models on the healthcare dataset and saves them to /models.
Run this once before launching the Streamlit app.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_absolute_error, r2_score
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. Load & clean data
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "healthcare_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Normalise column names
df.columns = df.columns.str.strip()

# Title-case name fields so they're consistent
df["Name"] = df["Name"].str.title()

# Parse dates → stay-length feature
df["Date of Admission"] = pd.to_datetime(df["Date of Admission"], errors="coerce")
df["Discharge Date"]    = pd.to_datetime(df["Discharge Date"],    errors="coerce")
df["Stay Length"]       = (df["Discharge Date"] - df["Date of Admission"]).dt.days.fillna(0).astype(int)

# Drop columns not useful for modelling
df.drop(columns=["Name", "Doctor", "Hospital", "Date of Admission",
                  "Discharge Date", "Room Number"], inplace=True)

print(f"Dataset shape after cleaning: {df.shape}")

# ─────────────────────────────────────────────
# 2. Encode categoricals
# ─────────────────────────────────────────────
categorical_cols = ["Gender", "Blood Type", "Medical Condition",
                    "Insurance Provider", "Admission Type", "Medication",
                    "Test Results"]

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Save encoders + feature metadata
with open(os.path.join(MODEL_DIR, "encoders.pkl"), "wb") as f:
    pickle.dump(encoders, f)

print("Encoders saved.")

# ─────────────────────────────────────────────
# 3. Feature sets
# ─────────────────────────────────────────────
BASE_FEATURES = ["Age", "Gender", "Blood Type", "Medical Condition",
                 "Insurance Provider", "Admission Type", "Medication",
                 "Stay Length"]

# ── Model A: Test Result Classifier ──────────
X_a = df[BASE_FEATURES]
y_a = df["Test Results"]
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
    X_a, y_a, test_size=0.2, random_state=42, stratify=y_a
)
clf_test = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf_test.fit(X_train_a, y_train_a)
acc_a = accuracy_score(y_test_a, clf_test.predict(X_test_a))
print(f"\n[Test Result Classifier]  Accuracy: {acc_a:.4f}")
print(classification_report(y_test_a, clf_test.predict(X_test_a),
      target_names=encoders["Test Results"].classes_))
with open(os.path.join(MODEL_DIR, "test_result_model.pkl"), "wb") as f:
    pickle.dump(clf_test, f)

# ── Model B: Billing Amount Regressor ────────
BILLING_FEATURES = ["Age", "Gender", "Blood Type", "Medical Condition",
                    "Insurance Provider", "Admission Type", "Medication",
                    "Stay Length"]
X_b = df[BILLING_FEATURES]
y_b = df["Billing Amount"]
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_b, y_b, test_size=0.2, random_state=42
)
reg_billing = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                         max_depth=5, random_state=42)
reg_billing.fit(X_train_b, y_train_b)
mae_b = mean_absolute_error(y_test_b, reg_billing.predict(X_test_b))
r2_b  = r2_score(y_test_b, reg_billing.predict(X_test_b))
print(f"\n[Billing Regressor]  MAE: {mae_b:.2f}  |  R²: {r2_b:.4f}")
with open(os.path.join(MODEL_DIR, "billing_model.pkl"), "wb") as f:
    pickle.dump(reg_billing, f)

# ── Model C: Admission Type Classifier ───────
ADM_FEATURES = ["Age", "Gender", "Blood Type", "Medical Condition",
                "Insurance Provider", "Medication", "Stay Length",
                "Billing Amount", "Test Results"]
X_c = df[ADM_FEATURES]
y_c = df["Admission Type"]
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_c, y_c, test_size=0.2, random_state=42, stratify=y_c
)
clf_adm = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf_adm.fit(X_train_c, y_train_c)
acc_c = accuracy_score(y_test_c, clf_adm.predict(X_test_c))
print(f"\n[Admission Type Classifier]  Accuracy: {acc_c:.4f}")
print(classification_report(y_test_c, clf_adm.predict(X_test_c),
      target_names=encoders["Admission Type"].classes_))
with open(os.path.join(MODEL_DIR, "admission_model.pkl"), "wb") as f:
    pickle.dump(clf_adm, f)

# Save feature lists so app.py can stay in sync
meta = {
    "base_features":    BASE_FEATURES,
    "billing_features": BILLING_FEATURES,
    "adm_features":     ADM_FEATURES,
    "model_accuracy": {
        "test_result": round(acc_a, 4),
        "admission":   round(acc_c, 4),
    },
    "model_metrics": {
        "billing_mae": round(mae_b, 2),
        "billing_r2":  round(r2_b,  4),
    }
}
with open(os.path.join(MODEL_DIR, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print("\nAll models saved to /models. Training complete.")
