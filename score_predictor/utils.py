"""
Utility helpers for serialization, model evaluation, and file I/O.
"""

import os
import sys
import pickle
from typing import Dict, Any

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from score_predictor.exception import PipelineException


def persist_object(filepath: str, obj: Any) -> None:
    """Serialize a Python object to disk using pickle."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)
    except Exception as exc:
        raise PipelineException(exc, sys.exc_info())


def restore_object(filepath: str) -> Any:
    """Deserialize a pickled object from disk."""
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        raise PipelineException(exc, sys.exc_info())


def benchmark_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    candidates: Dict[str, Any],
    hyperparams: Dict[str, Dict],
) -> Dict[str, float]:
    """
    Run GridSearchCV for each candidate model and return a dict
    mapping model name -> R² score on the test set.
    """
    try:
        scores = {}
        for name, estimator in candidates.items():
            search = GridSearchCV(
                estimator,
                hyperparams.get(name, {}),
                cv=3,
                scoring="r2",
            )
            search.fit(X_train, y_train)

            # Refit with the best hyperparameters found
            estimator.set_params(**search.best_params_)
            estimator.fit(X_train, y_train)

            y_pred = estimator.predict(X_test)
            scores[name] = r2_score(y_test, y_pred)

        return scores

    except Exception as exc:
        raise PipelineException(exc, sys.exc_info())
