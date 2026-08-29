#  F1 Strategy Optimizer

An ML-powered Formula 1 strategy optimizer that uses historical lap data to predict lap times and evaluate tyre strategy under different race conditions.

## Live Demo

https://f1-strategy-optimizer-demo.streamlit.app/

> **Demo note:** The live deployment uses a lightweight preloaded demo dataset from the Monaco Grand Prix rather than the full historical dataset. The full project contains significantly more data and is intended for local use.

## What It Does

* Predicts expected lap times using a Random Forest regression model
* Evaluates tyre compounds and tyre life
* Accounts for driver, track, race, and stint characteristics
* Applies interpretable weather and tyre-condition adjustments
* Simulates different strategy scenarios
* Provides visualizations and strategy recommendations through an interactive Streamlit interface

## Machine Learning

The model uses a `RandomForestRegressor` with preprocessing for categorical and numerical features.

### Features

* Driver
* Lap number
* Tyre compound
* Tyre life
* Stint
* Track status
* Personal-best indicator
* Race year
* Grand Prix

### Model

* Random Forest Regression
* 300 estimators
* Maximum depth: 20
* Minimum samples per leaf: 2
* One-hot encoding for categorical variables
* Median imputation for numerical features

## Data

The full project uses historical Formula 1 lap-level data processed through the project's data pipeline.

For the deployed demo, a smaller pre-loaded Monaco dataset is included so the application can run within Streamlit's deployment constraints.

## Project Structure

```text
f1-strategy-optimizer/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── src/
│   ├── data_pipeline.py
│   ├── features.py
│   ├── model.py
│   ├── simulation.py
│   └── strategy.py
├── models/
│   ├── feature_importance.csv
│   └── metrics.csv
├── data/
│   └── demo/
├── scripts/
│   └── create_demo_data.py
└── streamlit_demo/
    ├── app.py
    ├── demo_laps.csv
    ├── lap_time_model.joblib
    └── src/
```

F1 race strategy involves continuously balancing tyre degradation, pace, weather, track conditions, and pit-stop decisions. This project explores how machine learning can be combined with interpretable strategy logic to help evaluate those decisions.

## Tech Stack

**Python · Pandas · NumPy · Scikit-learn · Streamlit · Plotly · FastF1**
