# ============================================================
# train_model.py
# Trains a Linear Regression model and saves it as a .pkl file
# Run once:  python train_model.py
# ============================================================

import json
import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------- 1. Find and load the dataset ----------
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model")
os.makedirs(MODEL_DIR, exist_ok=True)          # creates the model folder

candidates = [
    "data/cleaned_job_salary_dataset.csv",
    "cleaned_job_salary_dataset.csv",
    "data/job_salary_prediction_dataset.csv",
    "job_salary_prediction_dataset.csv",
]
csv_path = None
for name in candidates:
    p = os.path.join(BASE, name)
    if os.path.exists(p):
        csv_path = p
        break

if csv_path is None:
    print("ERROR: dataset CSV not found.")
    print("Project folder:", BASE)
    print("Files here    :", os.listdir(BASE))
    print("Put the CSV in this folder (or in a 'data' folder) and run again.")
    sys.exit(1)

print("Using dataset:", csv_path)
df = pd.read_csv(csv_path)
print("Dataset shape:", df.shape)

# ---------- 2. Features and target ----------
categorical_features = [
    "job_title", "education_level", "industry",
    "company_size", "location", "remote_work",
]
numerical_features = ["experience_years", "skills_count", "certifications"]

X = df[categorical_features + numerical_features]
y = df["salary"]

# ---------- 3. Train-test split (80/20) ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ---------- 4. Preprocessing + model in ONE pipeline ----------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numerical_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ]
)

# ---------- 5. Train ----------
model.fit(X_train, y_train)

# ---------- 6. Evaluate ----------
pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("\nMODEL PERFORMANCE (Linear Regression)")
print(f"MAE  : {mae:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R2   : {r2:.4f}")

# ---------- 7. Save the model ----------
joblib.dump(model, os.path.join(MODEL_DIR, "salary_model.pkl"))

# Save dropdown options + metrics for the website
options = {col: sorted(df[col].unique().tolist()) for col in categorical_features}

top_paying_titles = (
    df.groupby("job_title")["salary"].mean().sort_values(ascending=False).head(5)
)
top_paying_locations = (
    df.groupby("location")["salary"].mean().sort_values(ascending=False).head(5)
)
avg_salary_by_education = (
    df.groupby("education_level")["salary"].mean().sort_values(ascending=False)
)

stats = {
    "records": int(len(df)),
    "features": int(len(categorical_features) + len(numerical_features)),
    "avg_salary": float(round(df["salary"].mean(), 2)),
    "min_salary": float(round(df["salary"].min(), 2)),
    "max_salary": float(round(df["salary"].max(), 2)),
    "avg_experience": float(round(df["experience_years"].mean(), 1)),
    "top_paying_titles": [
        {"label": k, "value": float(round(v, 0))} for k, v in top_paying_titles.items()
    ],
    "top_paying_locations": [
        {"label": k, "value": float(round(v, 0))} for k, v in top_paying_locations.items()
    ],
    "avg_salary_by_education": [
        {"label": k, "value": float(round(v, 0))} for k, v in avg_salary_by_education.items()
    ],
}

meta = {
    "options": options,
    "metrics": {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 4)},
    "stats": stats,
}
with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("\nSaved to:", MODEL_DIR)