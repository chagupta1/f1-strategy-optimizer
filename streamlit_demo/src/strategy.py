from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features import (
    prepare_prediction_row,
)


@dataclass
class StrategyResult:
    name: str
    compounds: list
    predicted_time: float
    average_lap: float
    pit_stops: int
    confidence: float


COMPOUNDS = [
    "SOFT",
    "MEDIUM",
    "HARD",
]


PIT_STOP_LOSS = 22.0


def generate_strategies():
    return [
        ["SOFT", "MEDIUM"],
        ["SOFT", "HARD"],
        ["MEDIUM", "HARD"],
        ["MEDIUM", "SOFT"],
        ["HARD", "MEDIUM"],
        ["SOFT", "MEDIUM", "SOFT"],
        ["MEDIUM", "HARD", "MEDIUM"],
        ["SOFT", "HARD", "SOFT"],
    ]


def distribute_stints(total_laps, compounds):
    count = len(compounds)

    base = total_laps // count
    remainder = total_laps % count

    lengths = []

    for i in range(count):
        length = base

        if i < remainder:
            length += 1

        lengths.append(length)

    return lengths


def simulate_strategy(
    model,
    driver,
    team,
    grand_prix,
    compounds,
    total_laps,
    air_temp,
    track_temp,
    humidity,
    wind_speed,
    rainfall,
):
    stint_lengths = distribute_stints(
        total_laps,
        compounds,
    )

    total_time = 0.0
    lap_predictions = []

    global_lap = 1

    for stint_number, (
        compound,
        stint_length,
    ) in enumerate(
        zip(compounds, stint_lengths),
        start=1,
    ):

        for tyre_life in range(
            1,
            stint_length + 1,
        ):

            row = prepare_prediction_row(
                driver=driver,
                team=team,
                grand_prix=grand_prix,
                compound=compound,
                track_status="1",
                lap_number=global_lap,
                tyre_life=tyre_life,
                stint=stint_number,
                air_temp=air_temp,
                track_temp=track_temp,
                humidity=humidity,
                wind_speed=wind_speed,
                rainfall=rainfall,
            )

            prediction = float(
                model.predict(row)[0]
            )

            # Add a small degradation component so
            # the simulator explicitly accounts for
            # increasing tire age.
            degradation = (
                0.035 *
                max(0, tyre_life - 1) ** 1.25
            )

            if compound == "SOFT":
                degradation *= 1.15

            elif compound == "HARD":
                degradation *= 0.75

            weather_penalty = 0.0

            if rainfall > 0:
                if compound != "INTERMEDIATE":
                    weather_penalty += 3.0

            if track_temp > 45:
                weather_penalty += (
                    0.025 *
                    (track_temp - 45)
                )

            adjusted_prediction = (
                prediction +
                degradation +
                weather_penalty
            )

            total_time += adjusted_prediction

            lap_predictions.append(
                {
                    "Lap": global_lap,
                    "Compound": compound,
                    "TyreLife": tyre_life,
                    "PredictedLapTime": adjusted_prediction,
                    "Stint": stint_number,
                }
            )

            global_lap += 1

    pit_stops = len(compounds) - 1

    total_time += (
        pit_stops *
        PIT_STOP_LOSS
    )

    average_lap = (
        total_time / total_laps
    )

    # Confidence is intentionally conservative:
    # more stops and extreme weather produce
    # greater uncertainty.
    confidence = 90.0

    confidence -= pit_stops * 4

    if rainfall > 0:
        confidence -= 12

    if track_temp > 50:
        confidence -= 5

    confidence = max(
        50.0,
        min(95.0, confidence),
    )

    return {
        "Strategy": " → ".join(compounds),
        "Compounds": compounds,
        "PredictedRaceTime": total_time,
        "AverageLap": average_lap,
        "PitStops": pit_stops,
        "Confidence": confidence,
        "LapData": lap_predictions,
    }


def optimize_strategies(
    model,
    driver,
    team,
    grand_prix,
    total_laps,
    air_temp,
    track_temp,
    humidity,
    wind_speed,
    rainfall,
):
    results = []

    for compounds in generate_strategies():

        result = simulate_strategy(
            model=model,
            driver=driver,
            team=team,
            grand_prix=grand_prix,
            compounds=compounds,
            total_laps=total_laps,
            air_temp=air_temp,
            track_temp=track_temp,
            humidity=humidity,
            wind_speed=wind_speed,
            rainfall=rainfall,
        )

        results.append(result)

    results.sort(
        key=lambda result:
        result["PredictedRaceTime"]
    )

    return results


def format_race_time(seconds):
    minutes = int(seconds // 60)
    remaining = seconds - minutes * 60

    return (
        f"{minutes}:{remaining:05.2f}"
    )