from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/"
    "digital_clinic/feature_reliability/gps_clinical"
)

INPUT = ROOT / "gps_clinical_weekly_features_with_home.parquet"

OUTDIR = ROOT / "full_cohort_phq_analysis"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT = OUTDIR / "phq_severity_gps_summary.csv"


df = pd.read_parquet(INPUT).copy()

df["depression_score"] = pd.to_numeric(
    df["depression_score"],
    errors="coerce",
)

df["time_at_home_pct"] = pd.to_numeric(
    df["time_at_home_pct"],
    errors="coerce",
)

df["home_hours_per_day"] = (
    df["time_at_home_pct"] / 100.0 * 24.0
)

df["movement_km_per_day"] = (
    pd.to_numeric(
        df["displacement_distance_mean_m"],
        errors="coerce",
    )
    / 1000.0
)

df["mobility_diversity"] = pd.to_numeric(
    df["location_entropy_mean"],
    errors="coerce",
)


# Standard PHQ-9 severity categories
bins = [-0.001, 4, 9, 14, 19, 27]

labels = [
    "Minimal (0–4)",
    "Mild (5–9)",
    "Moderate (10–14)",
    "Moderately severe (15–19)",
    "Severe (20–27)",
]

df["phq_severity"] = pd.cut(
    df["depression_score"],
    bins=bins,
    labels=labels,
    ordered=True,
)


features = {
    "Movement distance (km/day)": "movement_km_per_day",
    "Time at home (h/day)": "home_hours_per_day",
    "Mobility diversity": "mobility_diversity",
}


print("=" * 100)
print("PHQ-9 SEVERITY × GPS — FULL COHORT")
print("=" * 100)

clinical = df.dropna(
    subset=["depression_score"]
).copy()

print("PHQ windows:", len(clinical))
print(
    "Participants with PHQ:",
    clinical["participant_id"].nunique(),
)

print()
print("=" * 100)
print("PHQ-9 SEVERITY DISTRIBUTION")
print("=" * 100)

for group in labels:

    x = clinical[
        clinical["phq_severity"].eq(group)
    ]

    print(
        f"{group:<30}"
        f" windows={len(x):>4}"
        f" | participants={x['participant_id'].nunique():>3}"
    )


rows = []

for feature_label, col in features.items():

    print()
    print("=" * 100)
    print(feature_label.upper())
    print("=" * 100)

    for group in labels:

        x = clinical.loc[
            clinical["phq_severity"].eq(group),
            ["participant_id", col],
        ].dropna()

        values = x[col]

        if len(values) == 0:
            continue

        mean = values.mean()
        median = values.median()
        sd = values.std(ddof=1)
        sem = sd / np.sqrt(len(values))

        q25 = values.quantile(0.25)
        q75 = values.quantile(0.75)

        n_windows = len(values)
        n_participants = x["participant_id"].nunique()

        rows.append(
            {
                "feature": feature_label,
                "phq_severity": group,
                "n_windows": n_windows,
                "n_participants": n_participants,
                "mean": mean,
                "median": median,
                "sd": sd,
                "sem": sem,
                "q25": q25,
                "q75": q75,
            }
        )

        print(
            f"{group:<30}"
            f" n={n_windows:>4}"
            f" | participants={n_participants:>3}"
            f" | mean={mean:>8.3f}"
            f" | median={median:>8.3f}"
            f" | IQR=[{q25:.3f}, {q75:.3f}]"
        )


summary = pd.DataFrame(rows)

summary.to_csv(
    OUT,
    index=False,
)


# -------------------------------------------------------------------------
# Compact trend table for visual inspection
# -------------------------------------------------------------------------

print()
print("=" * 100)
print("DOSE-RESPONSE SUMMARY")
print("=" * 100)

for feature_label in features:

    x = summary[
        summary["feature"].eq(feature_label)
    ]

    print()
    print(feature_label)

    for _, row in x.iterrows():

        print(
            f"  {row['phq_severity']:<30}"
            f"{row['mean']:.3f}"
        )


print()
print("=" * 100)
print("SAVED")
print("=" * 100)
print(OUT)
