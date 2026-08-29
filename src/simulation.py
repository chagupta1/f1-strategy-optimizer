import numpy as np
import pandas as pd


def simulate_strategy(
    strategy_time,
    simulations=10000,
    uncertainty=1.5,
    random_state=42,
):
    """
    Monte Carlo simulation of strategy outcomes.

    Adds random race-time uncertainty to estimate
    the distribution of possible outcomes.
    """

    rng = np.random.default_rng(
        random_state
    )

    simulated_times = rng.normal(
        loc=strategy_time,
        scale=uncertainty,
        size=simulations,
    )

    return simulated_times


def compare_strategy_simulations(
    strategies,
    simulations=10000,
):
    """
    Run Monte Carlo simulations for every candidate
    strategy.
    """

    results = []

    for _, row in strategies.iterrows():

        simulated = simulate_strategy(
            row["EstimatedTime"],
            simulations=simulations,
        )

        results.append(
            {
                "Strategy": row["Strategy"],
                "MeanTime": simulated.mean(),
                "BestCase": np.percentile(
                    simulated,
                    10,
                ),
                "WorstCase": np.percentile(
                    simulated,
                    90,
                ),
                "StdDev": simulated.std(),
            }
        )

    return pd.DataFrame(results)


def calculate_win_probability(
    strategy_times,
):
    """
    Estimate relative probability of a strategy
    producing the fastest simulated race time.
    """

    if not strategy_times:
        return {}

    strategy_names = list(
        strategy_times.keys()
    )

    arrays = []

    for strategy in strategy_names:

        arrays.append(
            simulate_strategy(
                strategy_times[strategy]
            )
        )

    simulations = np.array(arrays)

    winners = np.argmin(
        simulations,
        axis=0,
    )

    probabilities = {}

    for index, strategy in enumerate(
        strategy_names
    ):

        probabilities[strategy] = (
            winners == index
        ).mean()
    return probabilities