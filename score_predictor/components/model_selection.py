"""
Model selection component — trains several regression algorithms using
GridSearchCV, picks the best one, and persists it to disk.
"""

import os
import sys
from dataclasses import dataclass

import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from score_predictor.logger import logger
from score_predictor.exception import PipelineException
from score_predictor.utils import persist_object, benchmark_models


# ── Candidate models and their search grids ─────────────────────────
CANDIDATES = {
    "RandomForest": RandomForestRegressor(),
    "DecisionTree": DecisionTreeRegressor(),
    "GradientBoosting": GradientBoostingRegressor(),
    "LinearRegression": LinearRegression(),
    "XGBoost": XGBRegressor(),
    "CatBoost": CatBoostRegressor(verbose=False),
    "AdaBoost": AdaBoostRegressor(),
}

SEARCH_GRIDS = {
    "DecisionTree": {
        "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
    },
    "RandomForest": {
        "n_estimators": [8, 16, 32, 64, 128, 256],
    },
    "GradientBoosting": {
        "learning_rate": [0.1, 0.01, 0.05, 0.001],
        "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
        "n_estimators": [8, 16, 32, 64, 128, 256],
    },
    "LinearRegression": {},
    "XGBoost": {
        "learning_rate": [0.1, 0.01, 0.05, 0.001],
        "n_estimators": [8, 16, 32, 64, 128, 256],
    },
    "CatBoost": {
        "depth": [6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "iterations": [30, 50, 100],
    },
    "AdaBoost": {
        "learning_rate": [0.1, 0.01, 0.5, 0.001],
        "n_estimators": [8, 16, 32, 64, 128, 256],
    },
}

MIN_ACCEPTABLE_R2 = 0.6


@dataclass
class ModelSelectorConfig:
    """Where the final trained model is saved."""
    model_path: str = os.path.join("output", "trained_model.pkl")


class ModelSelector:
    """Evaluates multiple regressors and persists the champion model."""

    def __init__(self):
        self.config = ModelSelectorConfig()

    def run(self, train_array: np.ndarray, test_array: np.ndarray) -> float:
        """
        Train all candidate models, select the best by R², and save it.

        Returns
        -------
        float
            R² score of the best model on the test set.

        Raises
        ------
        PipelineException
            If no model exceeds the minimum acceptable R² threshold.
        """
        try:
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            logger.info("Benchmarking %d candidate models", len(CANDIDATES))
            scores = benchmark_models(
                X_train, y_train, X_test, y_test,
                candidates=CANDIDATES,
                hyperparams=SEARCH_GRIDS,
            )

            # Identify the winner
            best_name = max(scores, key=scores.get)
            best_score = scores[best_name]
            logger.info("Best model: %s with R² = %.4f", best_name, best_score)

            if best_score < MIN_ACCEPTABLE_R2:
                raise PipelineException(
                    f"No model met the R² threshold ({MIN_ACCEPTABLE_R2}). "
                    f"Best was {best_name} at {best_score:.4f}."
                )

            best_model = CANDIDATES[best_name]
            persist_object(self.config.model_path, best_model)
            logger.info("Champion model saved to %s", self.config.model_path)

            # Return the score for reporting
            return best_score

        except Exception as exc:
            raise PipelineException(exc, sys.exc_info())
