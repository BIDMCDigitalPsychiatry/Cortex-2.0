from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/"
    "digital_clinic/feature_reliability/gps_clinical"
)

INPUT_PATH = ROOT / "gps_clinical_weekly_features_with_home.parquet"

OUTPUT_DIR = ROOT / "full_cohort_phq_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BINARY_CSV = OUTPUT_DIR / "phq_binary_results.csv"
CONTINUOUS_CSV = OUTPUT_DIR / "phq_continuous_results.csv"


df = pd.read_parquet(INPUT_PATH).copy()

df["depression_score"] = pd.to_numeric(
    df["depression_score"],
    errors="coerce",
)

df["time_at_home_pct"] = pd.to_numeric(
    df["time_at_home_pct"],
    errors="coerce",
)

df["equivalent_home_hours"] = (
    df["time_at_home_pct"] / 100.0 * 24.0
)

df["mobility_diversity"] = pd.to_numeric(
    df["location_entropy_mean"],
    errors="coerce",
)

df["movement_distance_km"] = (
    pd.to_numeric(
        df["displacement_distance_mean_m"],
        errors="coerce",
    )
    / 1000.0
)


features = {
    "Equivalent time at home (h/day)":
        "equivalent_home_hours",

    "Mobility diversity":
        "mobility_diversity",

    "Daily movement distance (km/day)":
        "movement_distance_km",
}


print("=" * 100)
print("FULL-COHORT PHQ-9 × GPS ANALYSIS")
print("=" * 100)

print(
    "Total weekly rows:",
    len(df),
)

print(
    "Total participants:",
    df["participant_id"].nunique(),
)

print(
    "Rows with depression score:",
    df["depression_score"].notna().sum(),
)

print(
    "Participants with depression score:",
    df.loc[
        df["depression_score"].notna(),
        "participant_id",
    ].nunique(),
)


# =============================================================================
# BINARY: PHQ 0–4 VS >=5
# =============================================================================

binary_rows = []

for label, col in features.items():

    x = df[
        [
            "participant_id",
            "depression_score",
            col,
        ]
    ].dropna().copy()

    x["elevated"] = (
        x["depression_score"] >= 5
    ).astype(int)

    lower = x.loc[
        x["elevated"].eq(0),
        col,
    ]

    elevated = x.loc[
        x["elevated"].eq(1),
        col,
    ]

    X = sm.add_constant(
        x[["elevated"]]
    )

    result = sm.OLS(
        x[col],
        X,
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups":
                x["participant_id"]
        },
    )

    beta = float(
        result.params["elevated"]
    )

    ci = (
        result.conf_int()
        .loc["elevated"]
    )

    p = float(
        result.pvalues["elevated"]
    )

    binary_rows.append(
        {
            "feature": label,
            "n_windows": len(x),
            "n_participants":
                x["participant_id"].nunique(),
            "lower_n": len(lower),
            "elevated_n": len(elevated),
            "lower_mean": lower.mean(),
            "elevated_mean": elevated.mean(),
            "difference_elevated_minus_lower":
                beta,
            "ci_low":
                float(ci.iloc[0]),
            "ci_high":
                float(ci.iloc[1]),
            "clustered_p":
                p,
        }
    )


binary = pd.DataFrame(binary_rows)
binary.to_csv(BINARY_CSV, index=False)


print()
print("=" * 100)
print("BINARY PHQ-9: 0–4 VS >=5")
print("=" * 100)

for _, r in binary.iterrows():

    print()
    print(r["feature"])
    print("-" * 60)

    print(
        "Windows / participants:",
        int(r["n_windows"]),
        "/",
        int(r["n_participants"]),
    )

    print(
        "Lower mean:",
        f"{r['lower_mean']:.3f}",
    )

    print(
        "Elevated mean:",
        f"{r['elevated_mean']:.3f}",
    )

    print(
        "Difference:",
        f"{r['difference_elevated_minus_lower']:+.3f}",
    )

    print(
        "95% CI:",
        f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]",
    )

    print(
        "Cluster-adjusted p:",
        f"{r['clustered_p']:.6f}",
    )


# =============================================================================
# CONTINUOUS PHQ
# =============================================================================

continuous_rows = []

for label, col in features.items():

    x = df[
        [
            "participant_id",
            "depression_score",
            col,
        ]
    ].dropna().copy()

    X = sm.add_constant(
        x[["depression_score"]]
    )

    result = sm.OLS(
        x[col],
        X,
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups":
                x["participant_id"]
        },
    )

    beta = float(
        result.params[
            "depression_score"
        ]
    )

    ci = (
        result.conf_int()
        .loc["depression_score"]
    )

    p = float(
        result.pvalues[
            "depression_score"
        ]
    )

    rho = float(
        x[
            [
                "depression_score",
                col,
            ]
        ]
        .corr(method="spearman")
        .iloc[0, 1]
    )

    continuous_rows.append(
        {
            "feature": label,
            "n_windows": len(x),
            "n_participants":
                x["participant_id"].nunique(),
            "slope_per_1_phq":
                beta,
            "slope_per_5_phq":
                beta * 5.0,
            "ci_low_per_5":
                float(ci.iloc[0]) * 5.0,
            "ci_high_per_5":
                float(ci.iloc[1]) * 5.0,
            "clustered_p":
                p,
            "spearman_rho":
                rho,
        }
    )


continuous = pd.DataFrame(
    continuous_rows
)

continuous.to_csv(
    CONTINUOUS_CSV,
    index=False,
)


print()
print("=" * 100)
print("CONTINUOUS PHQ-9")
print("=" * 100)

for _, r in continuous.iterrows():

    print()
    print(r["feature"])
    print("-" * 60)

    print(
        "Windows / participants:",
        int(r["n_windows"]),
        "/",
        int(r["n_participants"]),
    )

    print(
        "Slope per 5 PHQ-9 points:",
        f"{r['slope_per_5_phq']:+.3f}",
    )

    print(
        "95% CI:",
        f"[{r['ci_low_per_5']:+.3f}, "
        f"{r['ci_high_per_5']:+.3f}]",
    )

    print(
        "Cluster-adjusted p:",
        f"{r['clustered_p']:.6f}",
    )

    print(
        "Spearman rho:",
        f"{r['spearman_rho']:+.3f}",
    )


print()
print("=" * 100)
print("SAVED")
print("=" * 100)

print(BINARY_CSV)
print(CONTINUOUS_CSV)
