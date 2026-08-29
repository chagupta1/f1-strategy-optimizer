from pathlib import Path

import fastf1
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "fastf1_cache"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

fastf1.Cache.enable_cache(str(CACHE_DIR))


RACES = [
    (2023, "Bahrain"),
    (2023, "Saudi Arabian"),
    (2023, "Australian"),
    (2023, "Azerbaijan"),
    (2023, "Miami"),
    (2023, "Monaco"),
    (2023, "Spanish"),
    (2023, "Canadian"),
    (2023, "Austrian"),
    (2023, "British"),
    (2023, "Hungarian"),
    (2023, "Belgian"),
    (2023, "Dutch"),
    (2023, "Italian"),
    (2023, "Singapore"),
    (2023, "Japanese"),
    (2023, "Qatar"),
    (2023, "United States"),
    (2023, "Mexico"),
    (2023, "São Paulo"),
    (2023, "Las Vegas"),
    (2023, "Abu Dhabi"),
]


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_weather(session):
    """
    Extract representative session weather.

    FastF1 exposes weather measurements through
    session.weather_data. We aggregate them into
    race-level features suitable for ML.
    """

    try:
        weather = session.weather_data

        if weather is None or weather.empty:
            return {
                "AirTemp": 25.0,
                "TrackTemp": 35.0,
                "Humidity": 50.0,
                "WindSpeed": 2.0,
                "Rainfall": 0.0,
            }

        result = {
            "AirTemp": safe_float(
                weather["AirTemp"].mean()
                if "AirTemp" in weather.columns
                else 25.0,
                25.0,
            ),
            "TrackTemp": safe_float(
                weather["TrackTemp"].mean()
                if "TrackTemp" in weather.columns
                else 35.0,
                35.0,
            ),
            "Humidity": safe_float(
                weather["Humidity"].mean()
                if "Humidity" in weather.columns
                else 50.0,
                50.0,
            ),
            "WindSpeed": safe_float(
                weather["WindSpeed"].mean()
                if "WindSpeed" in weather.columns
                else 2.0,
                2.0,
            ),
            "Rainfall": safe_float(
                weather["Rainfall"].mean()
                if "Rainfall" in weather.columns
                else 0.0,
                0.0,
            ),
        }

        return result

    except Exception:
        return {
            "AirTemp": 25.0,
            "TrackTemp": 35.0,
            "Humidity": 50.0,
            "WindSpeed": 2.0,
            "Rainfall": 0.0,
        }


def build_dataset(
    races=None,
    output_path=DATA_DIR / "training_data.csv",
):
    """
    Build an ML dataset from historical F1 race sessions.

    Each row represents a valid race lap and includes:
    - driver
    - team
    - circuit
    - tire compound
    - tire age
    - stint
    - track status
    - weather
    - race context
    - lap time
    """

    races = races or RACES

    rows = []

    for year, grand_prix in races:

        print(f"Loading {year} {grand_prix}...")

        try:
            session = fastf1.get_session(
                year,
                grand_prix,
                "R",
            )

            session.load(
                telemetry=False,
                weather=True,
                messages=False,
            )

            laps = session.laps

            if laps is None or laps.empty:
                continue

            weather = extract_weather(session)

            for _, lap in laps.iterrows():

                lap_time = lap.get("LapTime")

                if pd.isna(lap_time):
                    continue

                try:
                    lap_seconds = lap_time.total_seconds()
                except AttributeError:
                    continue

                if lap_seconds <= 0 or lap_seconds > 200:
                    continue

                compound = lap.get(
                    "Compound",
                    "UNKNOWN",
                )

                if pd.isna(compound):
                    compound = "UNKNOWN"

                driver = lap.get(
                    "Driver",
                    "UNKNOWN",
                )

                if pd.isna(driver):
                    driver = "UNKNOWN"

                team = lap.get(
                    "Team",
                    "UNKNOWN",
                )

                if pd.isna(team):
                    team = "UNKNOWN"

                rows.append(
                    {
                        "Driver": driver,
                        "Team": team,
                        "Year": year,
                        "GrandPrix": grand_prix,
                        "LapNumber": safe_float(
                            lap.get("LapNumber")
                        ),
                        "LapTimeSeconds": lap_seconds,
                        "Compound": str(compound),
                        "TyreLife": safe_float(
                            lap.get("TyreLife"),
                            1,
                        ),
                        "Stint": safe_float(
                            lap.get("Stint"),
                            1,
                        ),
                        "TrackStatus": str(
                            lap.get(
                                "TrackStatus",
                                "1",
                            )
                        ),
                        "IsPersonalBest": bool(
                            lap.get(
                                "IsPersonalBest",
                                False,
                            )
                        ),
                        "AirTemp": weather["AirTemp"],
                        "TrackTemp": weather["TrackTemp"],
                        "Humidity": weather["Humidity"],
                        "WindSpeed": weather["WindSpeed"],
                        "Rainfall": weather["Rainfall"],
                    }
                )

        except Exception as exc:
            print(
                f"Skipping {year} {grand_prix}: {exc}"
            )

    dataframe = pd.DataFrame(rows)

    if dataframe.empty:
        raise RuntimeError(
            "No training data was collected."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved {len(dataframe):,} laps to "
        f"{output_path}"
    )

    return dataframe


if __name__ == "__main__":
    build_dataset()