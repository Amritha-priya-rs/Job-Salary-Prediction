# ============================================================
# app.py  -  Flask backend
# Run:  python app.py   then open http://127.0.0.1:5000
# ============================================================

import json
import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE, "model", "salary_model.pkl"))
with open(os.path.join(BASE, "model", "meta.json")) as f:
    meta = json.load(f)

OPTIONS = meta["options"]
METRICS = meta["metrics"]
STATS = meta["stats"]


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    form_data = {}
    active_tab = "home"

    if request.method == "POST":
        active_tab = "predict"
        form_data = request.form.to_dict()
        try:
            row = pd.DataFrame([{
                "job_title": form_data["job_title"],
                "education_level": form_data["education_level"],
                "industry": form_data["industry"],
                "company_size": form_data["company_size"],
                "location": form_data["location"],
                "remote_work": form_data["remote_work"],
                "experience_years": float(form_data["experience_years"]),
                "skills_count": float(form_data["skills_count"]),
                "certifications": float(form_data["certifications"]),
            }])
            result = model.predict(row)[0]
            prediction = f"{max(result, 0):,.0f}"
        except (KeyError, ValueError):
            error = "Please fill in all fields with valid values."

    return render_template(
        "index.html",
        options=OPTIONS,
        metrics=METRICS,
        stats=STATS,
        prediction=prediction,
        error=error,
        form=form_data,
        active_tab=active_tab,
    )


if __name__ == "__main__":
    app.run(debug=True)