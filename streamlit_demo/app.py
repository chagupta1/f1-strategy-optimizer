from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.model import FEATURE_COLUMNS, load_model


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="F1 Strategy Optimizer",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)


DEMO_DIR = Path(__file__).resolve().parent
DATA_PATH = DEMO_DIR / "demo_laps.csv"# ---------------------------------------------------------
# STYLE
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }

        .hero {
            padding: 2rem 2.2rem;
            border-radius: 20px;
            background:
                linear-gradient(
                    135deg,
                    rgba(20,20,25,0.98),
                    rgba(45,45,55,0.92)
                );
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255,255,255,0.10);
        }

        .hero h1 {
            font-size: 3rem;
            margin-bottom: 0.3rem;
        }

        .hero p {
            font-size: 1.1rem;
            opacity: 0.78;
        }

        .metric-card {
            padding: 1.2rem;
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.10);
            background: rgba(255,255,255,0.035);
        }

        .recommendation {
            padding: 1.5rem;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.12);
            background: linear-gradient(
                135deg,
                rgba(255,255,255,0.07),
                rgba(255,255,255,0.025)
            );
            margin: 1rem 0;
        }

        .small-label {
            text-transform: uppercase;
            font-size: 0.72rem;
            letter-spacing: 0.12em;
            opacity: 0.6;
        }

        .big-number {
            font-size: 2rem;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# LOAD DATA / MODEL
# ---------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_ml_model():
    return load_model()


data = load_data()
model = load_ml_model()


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]

COMPOUND_COLORS = {
    "SOFT": "#ff4d5d",
    "MEDIUM": "#ffd84d",
    "HARD": "#eeeeee",
    "INTERMEDIATE": "#57d68d",
    "WET": "#4da6ff",
}


def weather_adjustment(
    temperature,
    humidity,
    track_temperature,
    rainfall,
    wind_speed,
):
    """
    Approximate weather sensitivity layer.

    The historical model was trained primarily on lap-level timing data,
    so these factors are used as an interpretable adjustment layer rather
    than pretending the historical dataset contains a perfect weather model.
    """

    adjustment = 0.0

    # Higher air temperature can increase thermal degradation.
    if temperature > 30:
        adjustment += (temperature - 30) * 0.025

    # Hot track surface increases tyre stress.
    if track_temperature > 40:
        adjustment += (track_temperature - 40) * 0.018

    # Humidity has a smaller direct effect.
    if humidity > 70:
        adjustment += (humidity - 70) * 0.004

    # Wind can affect balance and consistency.
    if wind_speed > 20:
        adjustment += (wind_speed - 20) * 0.01

    # Rain changes the tyre regime substantially.
    if rainfall > 0:
        adjustment += min(rainfall * 0.15, 2.5)

    return adjustment


def compound_adjustment(compound, temperature, track_temperature):
    """
    Simple strategy-layer adjustment reflecting tyre characteristics.
    """

    adjustment = {
        "SOFT": -0.25,
        "MEDIUM": 0.0,
        "HARD": 0.18,
    }.get(compound, 0.0)

    # Soft tyres become less attractive as conditions get hotter.
    if compound == "SOFT" and track_temperature > 45:
        adjustment += (track_temperature - 45) * 0.035

    # Hard tyres need temperature to work effectively.
    if compound == "HARD" and track_temperature < 30:
        adjustment += (30 - track_temperature) * 0.025

    return adjustment


def predict_lap(
    driver,
    lap_number,
    compound,
    tyre_life,
    stint,
    track_status,
    is_personal_best,
    year,
    grand_prix,
    temperature,
    humidity,
    track_temperature,
    rainfall,
    wind_speed,
):
    row = pd.DataFrame(
        [
            {
                "Driver": driver,
                "LapNumber": lap_number,
                "Compound": compound,
                "TyreLife": tyre_life,
                "Stint": stint,
                "TrackStatus": track_status,
                "IsPersonalBest": is_personal_best,
                "Year": year,
                "GrandPrix": grand_prix,
            }
        ]
    )

    prediction = float(model.predict(row[FEATURE_COLUMNS])[0])

    prediction += compound_adjustment(
        compound,
        temperature,
        track_temperature,
    )

    prediction += weather_adjustment(
        temperature,
        humidity,
        track_temperature,
        rainfall,
        wind_speed,
    )

    # Tyre degradation.
    prediction += max(0, tyre_life - 5) * 0.025

    return max(prediction, 20.0)


def strategy_time(
    driver,
    grand_prix,
    year,
    race_laps,
    pit_laps,
    compounds,
    temperature,
    humidity,
    track_temperature,
    rainfall,
    wind_speed,
):
    total = 0.0
    stint = 1
    tyre_life = 1

    pit_lap_set = set(pit_laps)

    for lap in range(1, race_laps + 1):

        if lap in pit_lap_set:
            # Approximate pit-lane time loss.
            total += 22.0
            stint += 1
            tyre_life = 1

        compound = compounds[min(stint - 1, len(compounds) - 1)]

        total += predict_lap(
            driver=driver,
            lap_number=lap,
            compound=compound,
            tyre_life=tyre_life,
            stint=stint,
            track_status=1,
            is_personal_best=0,
            year=year,
            grand_prix=grand_prix,
            temperature=temperature,
            humidity=humidity,
            track_temperature=track_temperature,
            rainfall=rainfall,
            wind_speed=wind_speed,
        )

        tyre_life += 1

    return total


# ---------------------------------------------------------
# HERO
# ---------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="small-label">Machine Learning × Race Strategy</div>
        <h1>🏎️ F1 Strategy Optimizer</h1>
        <p>
            Predict lap pace, simulate tyre strategies, model weather effects,
            and identify the fastest race plan from historical Formula 1 data.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("Race Setup")

drivers = sorted(data["Driver"].dropna().unique())

grand_prix_list = sorted(data["GrandPrix"].dropna().unique())

driver = st.sidebar.selectbox(
    "Driver",
    drivers,
)

grand_prix = st.sidebar.selectbox(
    "Grand Prix",
    grand_prix_list,
)

year = st.sidebar.selectbox(
    "Season",
    sorted(data["Year"].dropna().unique(), reverse=True),
)

race_laps = st.sidebar.slider(
    "Race distance",
    min_value=20,
    max_value=80,
    value=57,
)

st.sidebar.divider()

st.sidebar.subheader("Weather")

temperature = st.sidebar.slider(
    "Air temperature °C",
    5,
    45,
    25,
)

track_temperature = st.sidebar.slider(
    "Track temperature °C",
    10,
    60,
    35,
)

humidity = st.sidebar.slider(
    "Humidity %",
    10,
    100,
    55,
)

rainfall = st.sidebar.slider(
    "Rainfall mm/h",
    0.0,
    15.0,
    0.0,
    0.1,
)

wind_speed = st.sidebar.slider(
    "Wind km/h",
    0,
    60,
    10,
)


# ---------------------------------------------------------
# TOP METRICS
# ---------------------------------------------------------

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "Historical laps",
        f"{len(data):,}",
    )

with m2:
    st.metric(
        "Drivers",
        data["Driver"].nunique(),
    )

with m3:
    st.metric(
        "Circuits / GPs",
        data["GrandPrix"].nunique(),
    )

with m4:
    st.metric(
        "Model R²",
        "0.94",
    )


# ---------------------------------------------------------
# NAVIGATION
# ---------------------------------------------------------

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🎯 Strategy Optimizer",
        "📈 Lap Predictor",
        "🌦️ Weather Analysis",
        "🧠 Model & Data",
    ]
)


# =========================================================
# STRATEGY OPTIMIZER
# =========================================================

with tab1:

    st.header("Strategy Optimizer")

    st.write(
        "Compare common one-, two-, and three-stop strategies and "
        "estimate total race time under the selected conditions."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        strategy_1 = st.selectbox(
            "One-stop tyre",
            COMPOUNDS,
            index=1,
            key="s1",
        )

    with c2:
        strategy_2_a = st.selectbox(
            "Two-stop first tyre",
            COMPOUNDS,
            index=0,
            key="s2a",
        )

    with c3:
        strategy_2_b = st.selectbox(
            "Two-stop second tyre",
            COMPOUNDS,
            index=1,
            key="s2b",
        )

    pit_1 = int(race_laps * 0.55)
    pit_2 = int(race_laps * 0.35)

    strategies = [
        {
            "Strategy": f"1-stop: {strategy_1}",
            "Stops": 1,
            "Pit Laps": [pit_1],
            "Compounds": [strategy_1],
        },
        {
            "Strategy": f"2-stop: {strategy_2_a} → {strategy_2_b}",
            "Stops": 2,
            "Pit Laps": [pit_2, pit_1],
            "Compounds": [strategy_2_a, strategy_2_b],
        },
        {
            "Strategy": f"2-stop: {strategy_2_b} → {strategy_2_a}",
            "Stops": 2,
            "Pit Laps": [pit_2, pit_1],
            "Compounds": [strategy_2_b, strategy_2_a],
        },
        {
            "Strategy": "Aggressive: SOFT → MEDIUM → SOFT",
            "Stops": 2,
            "Pit Laps": [pit_2, pit_1],
            "Compounds": ["SOFT", "MEDIUM", "SOFT"],
        },
        {
            "Strategy": "Conservative: HARD → MEDIUM → HARD",
            "Stops": 2,
            "Pit Laps": [pit_2, pit_1],
            "Compounds": ["HARD", "MEDIUM", "HARD"],
        },
    ]

    results = []

    for strategy in strategies:

        total = strategy_time(
            driver=driver,
            grand_prix=grand_prix,
            year=year,
            race_laps=race_laps,
            pit_laps=strategy["Pit Laps"],
            compounds=strategy["Compounds"],
            temperature=temperature,
            humidity=humidity,
            track_temperature=track_temperature,
            rainfall=rainfall,
            wind_speed=wind_speed,
        )

        results.append(
            {
                "Strategy": strategy["Strategy"],
                "Stops": strategy["Stops"],
                "Predicted Race Time": total,
            }
        )

    results_df = pd.DataFrame(results).sort_values(
        "Predicted Race Time"
    )

    best = results_df.iloc[0]

    st.markdown(
        f"""
        <div class="recommendation">
            <div class="small-label">Recommended Strategy</div>
            <h2>🏆 {best["Strategy"]}</h2>
            <div class="big-number">
                {best["Predicted Race Time"]:.1f}s
            </div>
            <p>
                Estimated total race time under the selected weather and
                tyre assumptions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.35, 1])

    with col1:

        chart = px.bar(
            results_df,
            x="Strategy",
            y="Predicted Race Time",
            title="Predicted Race Time by Strategy",
        )

        chart.update_layout(
            xaxis_title="",
            yaxis_title="Race time (seconds)",
        )

        st.plotly_chart(
            chart,
            use_container_width=True,
        )

    with col2:

        display_df = results_df.copy()

        display_df["Predicted Race Time"] = display_df[
            "Predicted Race Time"
        ].round(1)

        display_df["Gap to Best"] = (
            display_df["Predicted Race Time"]
            - display_df["Predicted Race Time"].min()
        ).round(1)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# LAP PREDICTOR
# =========================================================

with tab2:

    st.header("Lap Time Predictor")

    c1, c2, c3 = st.columns(3)

    with c1:
        prediction_compound = st.selectbox(
            "Tyre compound",
            COMPOUNDS,
        )

    with c2:
        tyre_life = st.slider(
            "Tyre life",
            1,
            30,
            5,
        )

    with c3:
        lap_number = st.slider(
            "Lap number",
            1,
            race_laps,
            min(10, race_laps),
        )

    predicted = predict_lap(
        driver=driver,
        lap_number=lap_number,
        compound=prediction_compound,
        tyre_life=tyre_life,
        stint=1,
        track_status=1,
        is_personal_best=0,
        year=year,
        grand_prix=grand_prix,
        temperature=temperature,
        humidity=humidity,
        track_temperature=track_temperature,
        rainfall=rainfall,
        wind_speed=wind_speed,
    )

    st.metric(
        "Predicted Lap Time",
        f"{predicted:.3f} s",
    )

    comparison = []

    for compound in COMPOUNDS:

        value = predict_lap(
            driver=driver,
            lap_number=lap_number,
            compound=compound,
            tyre_life=tyre_life,
            stint=1,
            track_status=1,
            is_personal_best=0,
            year=year,
            grand_prix=grand_prix,
            temperature=temperature,
            humidity=humidity,
            track_temperature=track_temperature,
            rainfall=rainfall,
            wind_speed=wind_speed,
        )

        comparison.append(
            {
                "Compound": compound,
                "Predicted Lap": value,
            }
        )

    comparison_df = pd.DataFrame(comparison)

    fig = px.bar(
        comparison_df,
        x="Compound",
        y="Predicted Lap",
        title="Predicted Pace by Compound",
        color="Compound",
        color_discrete_map=COMPOUND_COLORS,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# =========================================================
# WEATHER
# =========================================================

with tab3:

    st.header("Weather Sensitivity")

    st.write(
        "Explore how changing conditions can affect the estimated pace "
        "of each tyre compound."
    )

    weather_rows = []

    temperatures = list(range(15, 46, 5))

    for temp in temperatures:

        for compound in COMPOUNDS:

            value = predict_lap(
                driver=driver,
                lap_number=lap_number,
                compound=compound,
                tyre_life=5,
                stint=1,
                track_status=1,
                is_personal_best=0,
                year=year,
                grand_prix=grand_prix,
                temperature=temp,
                humidity=humidity,
                track_temperature=max(temp + 10, 20),
                rainfall=rainfall,
                wind_speed=wind_speed,
            )

            weather_rows.append(
                {
                    "Temperature": temp,
                    "Compound": compound,
                    "Lap Time": value,
                }
            )

    weather_df = pd.DataFrame(weather_rows)

    fig = px.line(
        weather_df,
        x="Temperature",
        y="Lap Time",
        color="Compound",
        markers=True,
        title="Tyre Performance Across Air Temperature",
        color_discrete_map=COMPOUND_COLORS,
    )

    fig.update_layout(
        xaxis_title="Air temperature (°C)",
        yaxis_title="Predicted lap time (seconds)",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.info(
        "Weather effects are modeled as an interpretable strategy layer "
        "on top of the historical lap-time model. This avoids claiming "
        "the training dataset contains weather variables that it does not."
    )


# =========================================================
# MODEL / DATA
# =========================================================

with tab4:

    st.header("Model & Data")

    st.subheader("Training Dataset")

    st.dataframe(
        data.head(100),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Available Features")

    feature_df = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Type": [
                str(data[column].dtype)
                if column in data.columns
                else "derived"
                for column in FEATURE_COLUMNS
            ],
        }
    )

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Dataset Coverage")

    coverage = pd.DataFrame(
        {
            "Metric": [
                "Lap records",
                "Drivers",
                "Grand Prix",
                "Seasons",
                "Tyre compounds",
            ],
            "Value": [
                f"{len(data):,}",
                data["Driver"].nunique(),
                data["GrandPrix"].nunique(),
                data["Year"].nunique(),
                data["Compound"].nunique(),
            ],
        }
    )

    st.dataframe(
        coverage,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Model: Random Forest regression with categorical encoding and "
        "numerical feature preprocessing."
    )