"""
Data ingestion component — reads the raw CSV, performs a stratified-random
train/test split, and persists the splits to the output directory.
"""

import os
import sys
from dataclasses import dataclass, field

import pandas as pd
from sklearn.model_selection import train_test_split

from score_predictor.logger import logger
from score_predictor.exception import PipelineException
from score_predictor.components.feature_engineering import FeatureEngineer, FeatureEngineerConfig
from score_predictor.components.model_selection import ModelSelector, ModelSelectorConfig


@dataclass
class IngestionPaths:
    """Holds the file paths where raw / split data will be stored."""
    raw_data: str = os.path.join("output", "raw_dataset.csv")
    train_split: str = os.path.join("output", "train_split.csv")
    test_split: str = os.path.join("output", "test_split.csv")


class DataIngestion:
    """Reads the source CSV, splits it 80/20, and saves results."""

    def __init__(self):
        self.paths = IngestionPaths()

    def run(self, source_csv: str = os.path.join("data", "student_scores.csv")) -> tuple:
        """
        Execute the ingestion step.

        Parameters
        ----------
        source_csv : str
            Path to the raw dataset.

        Returns
        -------
        tuple[str, str]
            Paths to the training and testing CSV files.
        """
        logger.info("Starting data ingestion from %s", source_csv)
        try:
            df = pd.read_csv(source_csv)
            logger.info("Loaded %d rows and %d columns", *df.shape)

            os.makedirs(os.path.dirname(self.paths.raw_data), exist_ok=True)
            df.to_csv(self.paths.raw_data, index=False)

            train_df, test_df = train_test_split(
                df, test_size=0.2, random_state=42
            )
            train_df.to_csv(self.paths.train_split, index=False)
            test_df.to_csv(self.paths.test_split, index=False)

            logger.info(
                "Split complete — train: %d rows, test: %d rows",
                len(train_df), len(test_df),
            )
            return self.paths.train_split, self.paths.test_split

        except Exception as exc:
            raise PipelineException(exc, sys.exc_info())


# ── Quick-run entry point ────────────────────────────────────────────
if __name__ == "__main__":
    ingestion = DataIngestion()
    train_path, test_path = ingestion.run()

    engineer = FeatureEngineer()
    train_arr, test_arr, _ = engineer.run(train_path, test_path)

    selector = ModelSelector()
    r2 = selector.run(train_arr, test_arr)
    print(f"Best model R² score: {r2:.4f}")
