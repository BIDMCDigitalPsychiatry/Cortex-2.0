from pathlib import Path

import pandas as pd
import statsmodels.api as sm


ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/"
    "digital_clinic/feature_reliability/gps_clinical"
)

INPUT_PATH = ROOT / "gps_clinical_weekly_features_with_home.parquet"

OUTPUT_DIR = ROOT / "within_between_phq_home_time"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "within_between_phq_home_time_clustered_stats.csv"
DATA_PATH = OUTPUT_DIR / "within_between_phq_home_time_clustered_analysis.csv"


# =============================================================================
# LOAD
# =============================================================================

df = pd.read_parquet(
    INPUT_PATH
).copy()

df["depression_score"] = pd.to_numeric(
    df["depression_score"],
    errors="coerce",
)

df["time_at_home_pct"] = pd.to_numeric(
    df["time_at_home_pct"],
    errors="coerce",
)

df["equivalent_home_hours"] = (
    df["time_at_home_pct"]
    / 100.0
    * 24.0
)

analysis = (
    df[
        [
            "participant_id",
            "depression_score",
            "equivalent_home_hours",
        ]
    ]
    .dropna()
    .copy()
)

analysis = analysis[
    analysis["depression_score"].between(
        0,
        27,
        inclusive="both",
    )
].copy()


# =============================================================================
# WITHIN / BETWEEN DECOMPOSITION
# =============================================================================

analysis["phq_between"] = (
    analysis.groupby(
        "participant_id"
    )[
        "depression_score"
    ]
    .transform("mean")
)

analysis["phq_within"] = (
    analysis["depression_score"]
    - analysis["phq_between"]
)

analysis.to_csv(
    DATA_PATH,
    index=False,
)


print("=" * 100)
print("WITHIN- VS BETWEEN-PERSON PHQ-9 → TIME AT HOME")
print("CLUSTER-ROBUST OLS")
print("=" * 100)

print(
    "Weekly observations:",
    len(analysis),
)

print(
    "Participants:",
    analysis[
        "participant_id"
    ].nunique(),
)


# =============================================================================
# MODEL
# =============================================================================

X = analysis[
    [
        "phq_between",
        "phq_within",
    ]
].copy()

X = sm.add_constant(
    X
)

y = analysis[
    "equivalent_home_hours"
]

model = sm.OLS(
    y,
    X,
)

result = model.fit(
    cov_type="cluster",
    cov_kwds={
        "groups":
            analysis[
                "participant_id"
            ]
    },
)


# =============================================================================
# RESULTS
# =============================================================================

rows = []

for term, label in [
    (
        "phq_between",
        "Between-person PHQ effect",
    ),
    (
        "phq_within",
        "Within-person PHQ effect",
    ),
]:

    beta = float(
        result.params[
            term
        ]
    )

    ci = (
        result
        .conf_int()
        .loc[
            term
        ]
    )

    p = float(
        result.pvalues[
            term
        ]
    )

    rows.append(
        {
            "effect":
                label,

            "beta_hours_per_1_phq":
                beta,

            "beta_hours_per_5_phq":
                beta * 5.0,

            "ci_low_hours_per_5_phq":
                float(
                    ci.iloc[0]
                ) * 5.0,

            "ci_high_hours_per_5_phq":
                float(
                    ci.iloc[1]
                ) * 5.0,

            "clustered_p_value":
                p,
        }
    )


stats = pd.DataFrame(
    rows
)

stats.to_csv(
    CSV_PATH,
    index=False,
)


print()
print("=" * 100)
print("RESULT")
print("=" * 100)


for _, row in stats.iterrows():

    print()

    print(
        row[
            "effect"
        ]
    )

    print(
        "Slope per 5 PHQ-9 points:",
        f"{row['beta_hours_per_5_phq']:+.2f} h/day",
    )

    print(
        "95% CI:",
        (
            f"["
            f"{row['ci_low_hours_per_5_phq']:+.2f}, "
            f"{row['ci_high_hours_per_5_phq']:+.2f}"
            f"]"
        ),
    )

    print(
        "Cluster-adjusted p:",
        f"{row['clustered_p_value']:.6f}",
    )


print()
print("=" * 100)
print("CORRELATION CHECK")
print("=" * 100)

print(
    "corr(phq_between, phq_within):",
    analysis[
        [
            "phq_between",
            "phq_within",
        ]
    ]
    .corr()
    .iloc[
        0,
        1
    ],
)


print()
print("Saved:")
print(CSV_PATH)
print(DATA_PATH)
