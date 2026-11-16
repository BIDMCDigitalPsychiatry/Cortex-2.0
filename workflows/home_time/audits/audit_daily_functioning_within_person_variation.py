from pathlib import Path
import pandas as pd
import numpy as np

PATH = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/daily_functioning/"
    "daily_functioning_home_overlap.csv"
)

df = pd.read_csv(PATH, low_memory=False)

df["functioning_score"] = pd.to_numeric(
    df["functioning_score"],
    errors="coerce",
)

df["home_time_pct"] = pd.to_numeric(
    df["home_time_pct"],
    errors="coerce",
)

m = df.dropna(
    subset=[
        "participant_id",
        "functioning_score",
        "home_time_pct",
    ]
).copy()

by_person = (
    m.groupby("participant_id")
    .agg(
        n_days=("functioning_score", "size"),
        functioning_mean=("functioning_score", "mean"),
        functioning_sd=("functioning_score", "std"),
        functioning_min=("functioning_score", "min"),
        functioning_max=("functioning_score", "max"),
        functioning_unique=("functioning_score", "nunique"),
        home_mean=("home_time_pct", "mean"),
        home_sd=("home_time_pct", "std"),
        home_min=("home_time_pct", "min"),
        home_max=("home_time_pct", "max"),
    )
    .reset_index()
)

by_person["functioning_range"] = (
    by_person["functioning_max"]
    - by_person["functioning_min"]
)

by_person["home_range"] = (
    by_person["home_max"]
    - by_person["home_min"]
)

print("=" * 100)
print("WITHIN-PERSON VARIATION AUDIT")
print("=" * 100)

print("\nMatched observations:", len(m))
print("Participants:", m["participant_id"].nunique())

print("\nDays per participant:")
print(
    by_person["n_days"]
    .describe(
        percentiles=[0.25, 0.5, 0.75, 0.9]
    )
    .to_string()
)

print("\nParticipants with functioning variation:")
print(
    ">=2 unique functioning scores:",
    int((by_person["functioning_unique"] >= 2).sum()),
)
print(
    ">=3 unique functioning scores:",
    int((by_person["functioning_unique"] >= 3).sum()),
)
print(
    "No functioning variation:",
    int((by_person["functioning_unique"] == 1).sum()),
)

print("\nFunctioning within-person SD:")
print(
    by_person["functioning_sd"]
    .describe(
        percentiles=[0.25, 0.5, 0.75, 0.9]
    )
    .to_string()
)

print("\nFunctioning within-person range:")
print(
    by_person["functioning_range"]
    .describe(
        percentiles=[0.25, 0.5, 0.75, 0.9]
    )
    .to_string()
)

print("\nHome-time within-person SD:")
print(
    by_person["home_sd"]
    .describe(
        percentiles=[0.25, 0.5, 0.75, 0.9]
    )
    .to_string()
)

print("\nHome-time within-person range:")
print(
    by_person["home_range"]
    .describe(
        percentiles=[0.25, 0.5, 0.75, 0.9]
    )
    .to_string()
)

print("\nFunctioning score distribution among matched days:")
print(
    m["functioning_score"]
    .value_counts()
    .sort_index()
    .to_string()
)

OUT = PATH.parent / "daily_functioning_within_person_variation.csv"
by_person.to_csv(OUT, index=False)

print("\nSaved:")
print(" ", OUT)
