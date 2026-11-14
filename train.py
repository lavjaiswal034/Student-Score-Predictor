"""
One-command training script.
Usage: python train.py
"""

from score_predictor.components.data_ingestion import DataIngestion
from score_predictor.components.feature_engineering import FeatureEngineer
from score_predictor.components.model_selection import ModelSelector


def main():
    print("=" * 50)
    print("  Student Score Predictor — Training Pipeline")
    print("=" * 50)

    # Step 1: Ingest and split data
    print("\n[1/3] Ingesting data...")
    ingestion = DataIngestion()
    train_path, test_path = ingestion.run()
    print(f"      Train → {train_path}")
    print(f"      Test  → {test_path}")

    # Step 2: Feature engineering
    print("\n[2/3] Transforming features...")
    engineer = FeatureEngineer()
    train_arr, test_arr, preprocessor_path = engineer.run(train_path, test_path)
    print(f"      Preprocessor saved → {preprocessor_path}")

    # Step 3: Model selection
    print("\n[3/3] Training & evaluating models...")
    selector = ModelSelector()
    best_r2 = selector.run(train_arr, test_arr)
    print(f"      Best model R² score: {best_r2:.4f}")

    print("\n✓ Pipeline complete. Artifacts saved in output/")


if __name__ == "__main__":
    main()
