"""
Flask application entry point for the Student Score Prediction web app.
Serves a form-based UI where users can input student demographic and
academic details to receive a predicted math score.
"""

import os
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from score_predictor.pipeline.inference import InferenceEngine, StudentRecord

app = Flask(__name__, template_folder="views")


@app.route("/")
def landing_page():
    """Render the welcome / landing page."""
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def handle_prediction():
    """Accept student details via form and return the predicted math score."""
    if request.method == "GET":
        return render_template("home.html")

    # Collect form inputs into a StudentRecord dataclass-like object
    record = StudentRecord(
        gender=request.form.get("gender"),
        race_ethnicity=request.form.get("ethnicity"),
        parental_level_of_education=request.form.get("parental_level_of_education"),
        lunch=request.form.get("lunch"),
        test_preparation_course=request.form.get("test_preparation_course"),
        reading_score=float(request.form.get("reading_score")),
        writing_score=float(request.form.get("writing_score")),
    )

    input_df = record.to_dataframe()
    engine = InferenceEngine()
    prediction = engine.run(input_df)

    return render_template("home.html", results=round(prediction[0], 2))


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)
