from pathlib import Path

import pandas as pd

from src.model import train_model


DATA_PATH = Path("data/training_data.csv")


def main():
    print("Loading training data...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "data/training_data.csv does not exist. "
            "Run `python -m src.data_pipeline` first."
        )

    data = pd.read_csv(DATA_PATH)

    print(f"Loaded {len(data):,} lap records.")
    print("Training machine-learning model...")

    model, mae, rmse, r2 = train_model(data)

    print()
    print("Training complete.")
    print()
    print(f"MAE:  {mae:.3f} seconds")
    print(f"RMSE: {rmse:.3f} seconds")
    print(f"R²:   {r2:.3f}")
    print()
    print("Model saved to: models/lap_time_model.joblib")


if __name__ == "__main__":
    main()