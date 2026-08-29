# F1 Strategy Optimizer

Machine-learning race strategy analysis using historical Formula 1 race, tire, and weather data.

## Overview

F1 Strategy Optimizer is an interactive machine-learning application that predicts lap performance and evaluates alternative race strategies under changing tire, weather, and race conditions.

The project combines historical Formula 1 data with feature engineering, supervised machine learning, and strategy simulation to answer:

> Given a driver, circuit, tire selection, race length, and weather conditions, which strategy is expected to produce the fastest race?

## Features

### ML Lap-Time Prediction

The system predicts expected lap time using features including:

* Driver
* Team
* Circuit
* Tire compound
* Tire age
* Stint number
* Lap number
* Track status
* Air temperature
* Track temperature
* Humidity
* Wind speed
* Rainfall

Engineered features include tire-age effects, temperature differentials, heat stress, and wet-track indicators.

### Strategy Optimizer

The Strategy Optimizer evaluates multiple tire strategies and estimates:

* Predicted total race time
* Average lap time
* Number of pit stops
* Tire degradation
* Weather effects
* Strategy confidence

Strategies are ranked automatically, with the fastest predicted strategy highlighted as the recommendation.

### Weather Analysis

Users can explore how changing track temperatures and environmental conditions affect predicted lap performance.

### Tire Degradation

The application models how predicted lap performance changes as tire age increases and compares Soft, Medium, and Hard compounds.

### Model Comparison

Multiple regression models are evaluated:

* Ridge Regression
* Random Forest
* Gradient Boosting

The application reports:

* Mean Absolute Error
* Root Mean Squared Error
* R²
* Actual vs. predicted lap performance

## Architecture


Historical F1 Data
       │
       ▼
FastF1 Data Pipeline
       │
       ▼
Feature Engineering
       │
       ├── Driver / Team
       ├── Tire
       ├── Circuit
       ├── Race Context
       └── Weather
       │
       ▼
ML Model Training
       │
       ▼
Lap-Time Prediction
       │
       ▼
Strategy Simulation
       │
       ├── Tire Degradation
       ├── Pit-Stop Cost
       ├── Weather Effects
       └── Race Distance
       │
       ▼
Strategy Ranking
       │
       ▼
Interactive Streamlit Application
```

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* FastF1
* Plotly
* Streamlit
* Joblib

## Project Structure

```text
f1-strategy-optimizer/
├── app.py
├── train.py
├── README.md
├── requirements.txt
├── data/
│   ├── training_data.csv
│   └── fastf1_cache/
├── models/
│   ├── lap_time_model.joblib
│   └── model_metadata.joblib
└── src/
    ├── __init__.py
    ├── data_pipeline.py
    ├── features.py
    ├── model.py
    └── strategy.py


## Strategy Simulation

For each candidate strategy, the simulator divides the race into tire stints and predicts lap performance across the entire stint.

The candidate strategies are then ranked by predicted total race time.

## Important Modeling Note

The strategy optimizer is a predictive decision-support system rather than a perfect representation of an F1 team's race simulator.

Real-world strategy decisions depend on additional information such as:

* Traffic
* Tire availability
* Safety-car probability
* Fuel load
* Overtaking difficulty
* Track evolution
* Competitor strategy
* Pit-lane conditions
* Driver behavior
* Real-time weather forecasts

The project intentionally simplifies some of these factors while demonstrating how historical data and machine learning can be combined with scenario simulation.


The project demonstrates an end-to-end data/ML workflow:

1. Collecting real-world data
2. Cleaning and transforming data
3. Engineering predictive features
4. Training multiple machine-learning models
5. Evaluating model performance
6. Building a simulation layer
7. Turning predictions into an optimization problem
8. Creating an interactive decision-support interface


