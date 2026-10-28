from pathlib import Path
import pandas as pd
import numpy as np

DC = Path("/shared/Digital Clinic")

GPS_ROOT = (
    DC / "raphe_outputs/digital_clinic/"
    "feature_reliability/gps_observability"
)

OUT = (
    DC / "cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical"
)
OUT.mkdir(parents=True, exist_ok=True)

MAP = (
    DC / "raphe_outputs/digital_clinic/inventory/"
    "raphe_participant_map.csv"
)

SURVEY = DC / "active/survey_tabular_long.csv"

# ------------------------------------------------------------
# GPS reference days
# ------------------------------------------------------------

cal = pd.read_parquet(
    GPS_ROOT / "gps_validation_reference_location_features.parquet"
)
cal["cohort"] = "calibration"

hold = pd.read_parquet(
    GPS_ROOT / "gps_reliability_holdout_reference_location_features.parquet"
)
hold["cohort"] = "holdout"

gps = pd.concat([cal, hold], ignore_index=True)

gps["date"] = pd.to_datetime(
    gps["date"]
).dt.date

# ------------------------------------------------------------
# Participant mapping
# ------------------------------------------------------------

mapping = pd.read_csv(MAP)

# ------------------------------------------------------------
# Survey data
# ------------------------------------------------------------

survey = pd.read_csv(
    SURVEY,
    low_memory=False,
)

survey = survey[
    survey["activity_name"].isin(
        [
            "Weekly Depression Survey",
            "Weekly Anxiety Survey",
        ]
    )
].copy()

survey["datetime"] = pd.to_datetime(
    survey["datetime"],
    utc=True,
    errors="coerce",
)

survey = survey.merge(
    mapping,
    left_on="participant_id",
    right_on="source_participant_id",
    how="inner",
)

survey = survey.rename(
    columns={
        "participant_id_y": "clinical_participant_id"
    }
)

# ------------------------------------------------------------
# Score definitions
# ------------------------------------------------------------

def numeric_value(x):
    try:
        return float(x)
    except Exception:
        return np.nan


survey["numeric_value"] = survey["value"].apply(
    numeric_value
)

# Depression:
# use the first 9 PHQ-style symptom items.
# Exclude functional-difficulty item.
dep = survey[
    survey["activity_name"].eq(
        "Weekly Depression Survey"
    )
].copy()

dep = dep[
    dep["numeric_value"].between(0, 3)
]

dep_scores = (
    dep.groupby(
        [
            "clinical_participant_id",
            "timestamp",
        ],
        as_index=False,
    )
    .agg(
        assessment_datetime=("datetime", "min"),
        depression_score=("numeric_value", "sum"),
        depression_items=("numeric_value", "count"),
    )
)

# Require all 9 scored items
dep_scores.loc[
    dep_scores["depression_items"] != 9,
    "depression_score",
] = np.nan

dep_scores["assessment_date"] = (
    dep_scores["assessment_datetime"].dt.date
)

# Anxiety:
# use first 7 GAD-style symptom items.
# Exclude functional-difficulty item.
anx = survey[
    survey["activity_name"].eq(
        "Weekly Anxiety Survey"
    )
].copy()

anx = anx[
    anx["numeric_value"].between(0, 3)
]

anx_scores = (
    anx.groupby(
        [
            "clinical_participant_id",
            "timestamp",
        ],
        as_index=False,
    )
    .agg(
        assessment_datetime=("datetime", "min"),
        anxiety_score=("numeric_value", "sum"),
        anxiety_items=("numeric_value", "count"),
    )
)

# Require all 7 scored items
anx_scores.loc[
    anx_scores["anxiety_items"] != 7,
    "anxiety_score",
] = np.nan

anx_scores["assessment_date"] = (
    anx_scores["assessment_datetime"].dt.date
)

# ------------------------------------------------------------
# Attach nearest assessment within ±7 days
# ------------------------------------------------------------

def nearest_score(
    gps_row,
    scores,
    score_col,
):
    x = scores[
        scores["clinical_participant_id"].eq(
            gps_row["participant_id"]
        )
    ].copy()

    x = x[x[score_col].notna()]

    if x.empty:
        return {
            score_col: np.nan,
            f"{score_col}_assessment_date": pd.NaT,
            f"{score_col}_distance_days": np.nan,
        }

    x["distance_days"] = x[
        "assessment_date"
    ].apply(
        lambda d: abs(
            (d - gps_row["date"]).days
        )
    )

    x = x.sort_values(
        [
            "distance_days",
            "assessment_datetime",
        ]
    )

    r = x.iloc[0]

    if r["distance_days"] > 7:
        return {
            score_col: np.nan,
            f"{score_col}_assessment_date": pd.NaT,
            f"{score_col}_distance_days": r["distance_days"],
        }

    return {
        score_col: r[score_col],
        f"{score_col}_assessment_date": r["assessment_date"],
        f"{score_col}_distance_days": r["distance_days"],
    }


rows = []

for _, g in gps.iterrows():

    d = nearest_score(
        g,
        dep_scores,
        "depression_score",
    )

    a = nearest_score(
        g,
        anx_scores,
        "anxiety_score",
    )

    row = g.to_dict()
    row.update(d)
    row.update(a)

    rows.append(row)

out = pd.DataFrame(rows)

out.to_csv(
    OUT / "gps_reference_days_with_clinical_scores.csv",
    index=False,
)

# ------------------------------------------------------------
# Compact audit
# ------------------------------------------------------------

print("=" * 80)
print("GPS + CLINICAL ALIGNMENT")
print("=" * 80)

print("Rows:", len(out))
print(
    "Participants:",
    out["participant_id"].nunique(),
)

for score in [
    "depression_score",
    "anxiety_score",
]:
    print("\n", score)
    print("-" * 50)

    x = out[out[score].notna()]

    print(
        "Aligned GPS days:",
        len(x),
        "/",
        len(out),
    )

    print(
        "Participants:",
        x["participant_id"].nunique(),
    )

    print(
        "Score mean:",
        round(x[score].mean(), 3),
    )

    print(
        "Score SD:",
        round(x[score].std(), 3),
    )

    print(
        "Score range:",
        (
            x[score].min(),
            x[score].max(),
        ),
    )

    print("\nBy cohort:")

    print(
        x.groupby("cohort")
        .agg(
            rows=(score, "size"),
            participants=(
                "participant_id",
                "nunique",
            ),
            mean=(score, "mean"),
            sd=(score, "std"),
        )
        .to_string()
    )

print(
    "\nSaved:",
    OUT / "gps_reference_days_with_clinical_scores.csv",
)
