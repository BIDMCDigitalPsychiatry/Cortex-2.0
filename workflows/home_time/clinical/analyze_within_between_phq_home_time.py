from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/"
    "digital_clinic/feature_reliability/gps_clinical"
)

INPUT_PATH = ROOT / "gps_clinical_weekly_features_with_home.parquet"

OUTPUT_DIR = ROOT / "within_between_phq_home_time"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CSV_PATH = OUTPUT_DIR / "within_between_phq_home_time_stats.csv"
DATA_PATH = OUTPUT_DIR / "within_between_phq_home_time_analysis.csv"


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
# PERSON-MEAN DECOMPOSITION
# =============================================================================

participant_mean_phq = (
    analysis.groupby(
        "participant_id"
    )[
        "depression_score"
    ]
    .transform("mean")
)

analysis[
    "phq_between"
] = participant_mean_phq

analysis[
    "phq_within"
] = (
    analysis[
        "depression_score"
    ]
    - participant_mean_phq
)


analysis.to_csv(
    DATA_PATH,
    index=False,
)


print("=" * 100)
print("WITHIN- VS BETWEEN-PERSON PHQ-9 → TIME AT HOME")
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
# RANDOM-INTERCEPT MODEL
# =============================================================================

model = smf.mixedlm(
    (
        "equivalent_home_hours ~ "
        "phq_between + phq_within"
    ),
    data=analysis,
    groups=analysis[
        "participant_id"
    ],
)

result = model.fit(
    reml=False,
    method="lbfgs",
    maxiter=2000,
    disp=False,
)


# =============================================================================
# EXTRACT RESULTS
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

    beta_per_point = float(
        result.params[
            term
        ]
    )

    beta_per_5 = (
        beta_per_point
        * 5.0
    )

    ci = (
        result
        .conf_int()
        .loc[
            term
        ]
    )

    ci_low_per_5 = (
        float(
            ci.iloc[0]
        )
        * 5.0
    )

    ci_high_per_5 = (
        float(
            ci.iloc[1]
        )
        * 5.0
    )

    p_value = float(
        result.pvalues[
            term
        ]
    )

    rows.append(
        {
            "effect":
                label,

            "beta_hours_per_1_phq":
                beta_per_point,

            "beta_hours_per_5_phq":
                beta_per_5,

            "ci_low_hours_per_5_phq":
                ci_low_per_5,

            "ci_high_hours_per_5_phq":
                ci_high_per_5,

            "p_value":
                p_value,
        }
    )


stats = pd.DataFrame(
    rows
)

stats.to_csv(
    CSV_PATH,
    index=False,
)


# =============================================================================
# PRINT
# =============================================================================

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
        "p:",
        f"{row['p_value']:.6f}",
    )


print()
print("=" * 100)
print("MODEL STATUS")
print("=" * 100)

print(
    "Converged:",
    result.converged,
)

print(
    "Random intercept variance:",
    float(
        result.cov_re.iloc[
            0,
            0
        ]
    ),
)

print()
print("Saved:")
print(CSV_PATH)
print(DATA_PATH)
