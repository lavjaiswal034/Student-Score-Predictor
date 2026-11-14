"""
Feature engineering component — builds sklearn preprocessing pipelines
for numeric and categorical columns and applies them to train/test data.
"""

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from score_predictor.logger import logger
from score_predictor.exception import PipelineException
from score_predictor.utils import persist_object


# ── Column definitions ──────────────────────────────────────────────
NUMERIC_FEATURES = ["writing_score", "reading_score"]
CATEGORICAL_FEATURES = [
    "gender",
    "race_ethnicity",
    "parental_level_of_education",
    "lunch",
    "test_preparation_course",
]
TARGET_COLUMN = "math_score"


@dataclass
class FeatureEngineerConfig:
    """Config holding the path where the fitted preprocessor is saved."""
    preprocessor_path: str = os.path.join("output", "preprocessor.pkl")


class FeatureEngineer:
    """Builds and applies column-wise preprocessing pipelines."""

    def __init__(self):
        self.config = FeatureEngineerConfig()

    def _build_preprocessor(self) -> ColumnTransformer:
        """Construct a ColumnTransformer with separate numeric and categorical pipelines."""
        numeric_pipe = Pipeline(
            steps=[
                ("fill_missing", SimpleImputer(strategy="median")),
                ("standardize", StandardScaler()),
            ]
        )

        categorical_pipe = Pipeline(
            steps=[
                ("fill_missing", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
                ("standardize", StandardScaler(with_mean=False)),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipe, NUMERIC_FEATURES),
                ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
            ]
        )

    def run(self, train_csv: str, test_csv: str):
        """
        Read train/test CSVs, fit the preprocessor on training data,
        transform both splits, and return numpy arrays.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, str]
            (train_array, test_array, preprocessor_path)
        """
        try:
            train_df = pd.read_csv(train_csv)
            test_df = pd.read_csv(test_csv)
            logger.info("Loaded train (%d rows) and test (%d rows) for transformation",
                        len(train_df), len(test_df))

            preprocessor = self._build_preprocessor()

            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN]
            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN]

            logger.info("Fitting preprocessor on training data")
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Combine features and target back into single arrays
            train_array = np.c_[X_train_transformed, np.array(y_train)]
            test_array = np.c_[X_test_transformed, np.array(y_test)]

            persist_object(self.config.preprocessor_path, preprocessor)
            logger.info("Preprocessor saved to %s", self.config.preprocessor_path)

            return train_array, test_array, self.config.preprocessor_path

        except Exception as exc:
            raise PipelineException(exc, sys.exc_info())
