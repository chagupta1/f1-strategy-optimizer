import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
    "LapNumber",
    "TyreLife",
    "Stint",
    "AirTemp",
    "TrackTemp",
    "Humidity",
    "WindSpeed",
    "Rainfall",
]


CATEGORICAL_FEATURES = [
    "Driver",
    "Team",
    "GrandPrix",
    "Compound",
    "TrackStatus",
]


FEATURE_COLUMNS = (
    NUMERIC_FEATURES +
    CATEGORICAL_FEATURES
)


def add_engineered_features(df):
    data = df.copy()

    defaults = {
        "TyreLife": 1,
        "Stint": 1,
        "LapNumber": 1,
        "AirTemp": 25,
        "TrackTemp": 35,
        "Humidity": 50,
        "WindSpeed": 2,
        "Rainfall": 0,
    }

    for column, default in defaults.items():
        if column not in data.columns:
            data[column] = default

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        ).fillna(default)

    data["TyreAgeSquared"] = (
        data["TyreLife"] ** 2
    )

    data["TrackTempDelta"] = (
        data["TrackTemp"] -
        data["AirTemp"]
    )

    data["HeatStress"] = (
        data["TrackTemp"] *
        data["Humidity"] / 100
    )

    data["WetTrack"] = (
        data["Rainfall"] > 0
    ).astype(int)

    data["StintProgress"] = (
        data["TyreLife"] /
        data["TyreLife"].clip(lower=1)
    )

    data["Compound"] = (
        data["Compound"]
        .fillna("UNKNOWN")
        .astype(str)
    )

    for column in CATEGORICAL_FEATURES:
        if column not in data.columns:
            data[column] = "UNKNOWN"

        data[column] = (
            data[column]
            .fillna("UNKNOWN")
            .astype(str)
        )

    return data


def prepare_prediction_row(
    driver,
    team,
    grand_prix,
    compound,
    track_status,
    lap_number,
    tyre_life,
    stint,
    air_temp,
    track_temp,
    humidity,
    wind_speed,
    rainfall,
):
    row = pd.DataFrame(
        [
            {
                "Driver": driver,
                "Team": team,
                "GrandPrix": grand_prix,
                "Compound": compound,
                "TrackStatus": track_status,
                "LapNumber": lap_number,
                "TyreLife": tyre_life,
                "Stint": stint,
                "AirTemp": air_temp,
                "TrackTemp": track_temp,
                "Humidity": humidity,
                "WindSpeed": wind_speed,
                "Rainfall": rainfall,
            }
        ]
    )

    return add_engineered_features(row)