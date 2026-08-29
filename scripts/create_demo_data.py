from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

source = DATA_DIR / "training_data.csv"
demo_dir = DATA_DIR / "demo"

demo_dir.mkdir(parents=True, exist_ok=True)

print("Loading full training dataset...")
df = pd.read_csv(source)

print(f"Full dataset: {len(df):,} laps")

# Pick one real race from the dataset for the lightweight deployed demo.
# Prefer a race with enough drivers/laps to make the demo useful.
preferred_races = [
    "Monaco",
    "British",
    "Belgian",
    "Italian",
    "Abu Dhabi",
]

selected_race = None

for race in preferred_races:
    matches = df[df["GrandPrix"].astype(str).str.contains(race, case=False, na=False)]
    if len(matches) >= 500:
        selected_race = race
        demo = matches.copy()
        break

if selected_race is None:
    # Fallback: use the largest Grand Prix in the dataset.
    counts = df["GrandPrix"].value_counts()
    selected_race = counts.index[0]
    demo = df[df["GrandPrix"] == selected_race].copy()

print(f"Selected demo race: {selected_race}")
print(f"Demo rows before sampling: {len(demo):,}")

# Keep the demo small enough for deployment while retaining
# representative laps across drivers, compounds, and stints.
if len(demo) > 1500:
    demo = (
        demo.groupby(
            ["Driver", "Compound"],
            group_keys=False
        )
        .apply(
            lambda x: x.sample(
                min(len(x), 40),
                random_state=42
            )
        )
        .reset_index(drop=True)
    )

demo_path = demo_dir / "demo_laps.csv"
demo.to_csv(demo_path, index=False)

# Also create a tiny summary file for the app.
summary = pd.DataFrame(
    {
        "Metric": [
            "Source laps",
            "Demo laps",
            "Demo race",
            "Drivers",
            "Tyre compounds",
            "Years",
        ],
        "Value": [
            len(df),
            len(demo),
            selected_race,
            demo["Driver"].nunique(),
            demo["Compound"].nunique(),
            demo["Year"].nunique(),
        ],
    }
)

summary_path = demo_dir / "demo_summary.csv"
summary.to_csv(summary_path, index=False)

print()
print("Demo data created:")
print(f"  {demo_path}")
print(f"  {summary_path}")
print()
print(f"Demo size: {len(demo):,} laps")
print(f"Drivers: {demo['Driver'].nunique()}")
print(f"Compounds: {sorted(demo['Compound'].dropna().unique())}")