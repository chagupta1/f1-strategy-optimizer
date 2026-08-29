from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


MODEL_PATH = Path("models/lap_time_model.joblib")

FEATURE_COLUMNS = [
    "Driver",
    "LapNumber",
    "Compound",
    "TyreLife",
    "Stint",
    "TrackStatus",
    "IsPersonalBest",
    "Year",
    "GrandPrix",
]

CATEGORICAL_FEATURES = [
    "Driver",
    "Compound",
    "GrandPrix",
]

NUMERIC_FEATURES = [
    "LapNumber",
    "TyreLife",
    "Stint",
    "TrackStatus",
    "IsPersonalBest",
    "Year",
]


def build_model():
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def train_model(data: pd.DataFrame):
    data = data.copy()

    data = data.dropna(subset=["LapTimeSeconds"])

    X = data[FEATURE_COLUMNS]
    y = data["LapTimeSeconds"]

    model = build_model()
    model.fit(X, y)

    predictions = model.predict(X)

    mae = mean_absolute_error(y, predictions)
    rmse = mean_squared_error(y, predictions) ** 0.5
    r2 = r2_score(y, predictions)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return model, mae, rmse, r2


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python train.py` first."
        )

    return joblib.load(MODEL_PATH)