"""
Inference pipeline — loads the persisted preprocessor and model,
and exposes a simple predict interface for the Flask app.
"""

import os
import sys

import pandas as pd

from score_predictor.exception import PipelineException
from score_predictor.utils import restore_object


class InferenceEngine:
    """Loads artifacts and runs predictions on new student records."""

    def __init__(
        self,
        model_path: str = os.path.join("output", "trained_model.pkl"),
        preprocessor_path: str = os.path.join("output", "preprocessor.pkl"),
    ):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path

    def run(self, features_df: pd.DataFrame):
        """
        Transform input features and return model predictions.

        Parameters
        ----------
        features_df : pd.DataFrame
            Single-row (or multi-row) DataFrame with the same columns
            the preprocessor was trained on.

        Returns
        -------
        np.ndarray
            Array of predicted math scores.
        """
        try:
            model = restore_object(self.model_path)
            preprocessor = restore_object(self.preprocessor_path)

            scaled_features = preprocessor.transform(features_df)
            return model.predict(scaled_features)

        except Exception as exc:
            raise PipelineException(exc, sys.exc_info())


class StudentRecord:
    """
    Convenience wrapper that converts individual form fields
    into a single-row DataFrame suitable for the inference pipeline.
    """

    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: float,
        writing_score: float,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def to_dataframe(self) -> pd.DataFrame:
        """Convert this record into a one-row DataFrame."""
        try:
            return pd.DataFrame({
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            })
        except Exception as exc:
            raise PipelineException(exc, sys.exc_info())
