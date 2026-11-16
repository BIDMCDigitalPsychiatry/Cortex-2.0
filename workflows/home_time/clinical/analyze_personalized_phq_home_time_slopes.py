from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/"
    "digital_clinic/feature_reliability/gps_clinical"
)

INPUT_PATH = ROOT / "gps_clinical_weekly_features_with_home.parquet"

OUTPUT_DIR = ROOT / "personalized_phq_home_time"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PNG_PATH = OUTPUT_DIR / "personalized_phq_home_time_slopes.png"
PDF_PATH = OUTPUT_DIR / "personalized_phq_home_time_slopes.pdf"
CSV_PATH = OUTPUT_DIR / "personalized_phq_home_time_slopes.csv"


# =============================================================================
# SETTINGS
# =============================================================================

MIN_OBSERVATIONS = 5
MIN_UNIQUE_PHQ = 2

POSITIVE_COLOR = "#C77C45"
NEGATIVE_COLOR = "#4F88A8"
NEUTRAL_COLOR = "#7A7A7A"
CI_COLOR = "#A9A9A9"


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


print("=" * 100)
print("PERSONALIZED PHQ-9 → TIME AT HOME")
print("=" * 100)

print(
    "Available weekly observations:",
    len(analysis),
)

print(
    "Available participants:",
    analysis[
        "participant_id"
    ].nunique(),
)


# =============================================================================
# PARTICIPANT-SPECIFIC MODELS
# =============================================================================

rows = []


for participant_id, x in analysis.groupby(
    "participant_id",
    sort=True,
):

    x = x.sort_values(
        "depression_score"
    ).copy()

    n_obs = len(x)

    n_unique_phq = (
        x[
            "depression_score"
        ].nunique()
    )

    if n_obs < MIN_OBSERVATIONS:
        continue

    if n_unique_phq < MIN_UNIQUE_PHQ:
        continue


    X = sm.add_constant(
        x[
            [
                "depression_score",
            ]
        ]
    )

    y = x[
        "equivalent_home_hours"
    ]


    result = sm.OLS(
        y,
        X,
    ).fit(
        cov_type="HC3"
    )

    slope_per_point = float(
        result.params[
            "depression_score"
        ]
    )

    slope_per_5 = (
        slope_per_point
        * 5.0
    )

    ci = (
        result
        .conf_int()
        .loc[
            "depression_score"
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
            "depression_score"
        ]
    )


    spearman_rho = float(
        x[
            [
                "depression_score",
                "equivalent_home_hours",
            ]
        ]
        .corr(
            method="spearman"
        )
        .iloc[
            0,
            1
        ]
    )


    rows.append(
        {
            "participant_id":
                participant_id,

            "n_observations":
                n_obs,

            "n_unique_phq_scores":
                n_unique_phq,

            "phq_min":
                float(
                    x[
                        "depression_score"
                    ].min()
                ),

            "phq_max":
                float(
                    x[
                        "depression_score"
                    ].max()
                ),

            "mean_home_hours":
                float(
                    x[
                        "equivalent_home_hours"
                    ].mean()
                ),

            "slope_hours_per_1_phq":
                slope_per_point,

            "slope_hours_per_5_phq":
                slope_per_5,

            "ci_low_hours_per_5_phq":
                ci_low_per_5,

            "ci_high_hours_per_5_phq":
                ci_high_per_5,

            "p_value":
                p_value,

            "spearman_rho":
                spearman_rho,
        }
    )

results = pd.DataFrame(
    rows
)


if results.empty:
    raise RuntimeError(
        "No participants met the minimum eligibility criteria."
    )


results["direction"] = np.select(
    [
        results[
            "ci_low_hours_per_5_phq"
        ] > 0,

        results[
            "ci_high_hours_per_5_phq"
        ] < 0,
    ],
    [
        "Positive CI excludes 0",
        "Negative CI excludes 0",
    ],
    default="CI includes 0",
)


results = results.sort_values(
    "slope_hours_per_5_phq"
).reset_index(
    drop=True
)


results.to_csv(
    CSV_PATH,
    index=False,
)


# =============================================================================
# SUMMARY
# =============================================================================

n_positive = int(
    (
        results[
            "slope_hours_per_5_phq"
        ] > 0
    ).sum()
)

n_negative = int(
    (
        results[
            "slope_hours_per_5_phq"
        ] < 0
    ).sum()
)

n_positive_ci = int(
    (
        results[
            "ci_low_hours_per_5_phq"
        ] > 0
    ).sum()
)

n_negative_ci = int(
    (
        results[
            "ci_high_hours_per_5_phq"
        ] < 0
    ).sum()
)


print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)

print(
    "Participants analyzed:",
    len(results),
)

print(
    "Positive individual slopes:",
    n_positive,
)

print(
    "Negative individual slopes:",
    n_negative,
)

print(
    "Positive slopes with 95% CI > 0:",
    n_positive_ci,
)

print(
    "Negative slopes with 95% CI < 0:",
    n_negative_ci,
)

print(
    "Median slope per 5 PHQ points:",
    f"{results['slope_hours_per_5_phq'].median():+.2f} h/day",
)

print(
    "Mean slope per 5 PHQ points:",
    f"{results['slope_hours_per_5_phq'].mean():+.2f} h/day",
)

print(
    "Slope range:",
    (
        f"{results['slope_hours_per_5_phq'].min():+.2f} "
        "to "
        f"{results['slope_hours_per_5_phq'].max():+.2f} h/day"
    ),
)


print()
print("=" * 100)
print("PARTICIPANT RESULTS")
print("=" * 100)

print(
    results[
        [
            "participant_id",
            "n_observations",
            "phq_min",
            "phq_max",
            "slope_hours_per_5_phq",
            "ci_low_hours_per_5_phq",
            "ci_high_hours_per_5_phq",
            "p_value",
            "spearman_rho",
        ]
    ]
    .to_string(
        index=False
    )
)


# =============================================================================
# FOREST FIGURE
# =============================================================================

plt.rcParams.update(
    {
        "font.family":
            "DejaVu Sans",

        "font.size":
            8.5,

        "axes.titlesize":
            10,

        "axes.labelsize":
            9,

        "xtick.labelsize":
            8,

        "ytick.labelsize":
            8,
    }
)


fig_height = max(
    5.5,
    0.30 * len(results) + 1.8,
)


fig, ax = plt.subplots(
    figsize=(
        7.2,
        fig_height,
    )
)


y_pos = np.arange(
    len(results)
)


for i, row in results.iterrows():

    slope = row[
        "slope_hours_per_5_phq"
    ]

    ci_low = row[
        "ci_low_hours_per_5_phq"
    ]

    ci_high = row[
        "ci_high_hours_per_5_phq"
    ]


    if (
        ci_low > 0
    ):
        color = POSITIVE_COLOR

    elif (
        ci_high < 0
    ):
        color = NEGATIVE_COLOR

    else:
        color = NEUTRAL_COLOR


    ax.plot(
        [
            ci_low,
            ci_high,
        ],
        [
            i,
            i,
        ],
        color=CI_COLOR,
        linewidth=1.4,
        zorder=1,
    )


    ax.scatter(
        slope,
        i,
        s=34,
        color=color,
        edgecolor="white",
        linewidth=0.5,
        zorder=2,
    )


# Zero-reference line
ax.axvline(
    0,
    color="#333333",
    linewidth=1,
    linestyle="--",
    alpha=0.8,
)


# Population estimate from prior analysis
POPULATION_SLOPE = 1.173
POPULATION_CI_LOW = 0.09
POPULATION_CI_HIGH = 2.26


ax.axvspan(
    POPULATION_CI_LOW,
    POPULATION_CI_HIGH,
    color="#D8E4EA",
    alpha=0.35,
    zorder=0,
)


ax.axvline(
    POPULATION_SLOPE,
    color="#173650",
    linewidth=1.3,
    alpha=0.9,
)


ax.set_yticks(
    y_pos
)

ax.set_yticklabels(
    results[
        "participant_id"
    ]
)


ax.set_xlabel(
    "Change in equivalent time at home per 5-point PHQ-9 increase (hours/day)"
)


ax.set_title(
    "Individual associations between depressive symptoms and time at home",
    loc="left",
    fontweight="bold",
    pad=10,
)


ax.grid(
    axis="x",
    linewidth=0.5,
    alpha=0.12,
)


ax.spines[
    "top"
].set_visible(
    False
)

ax.spines[
    "right"
].set_visible(
    False
)


ax.invert_yaxis()


fig.text(
    0.99,
    0.012,
    (
        "Points are participant-specific slopes; bars are HC3 95% CIs. "
        "Blue shading shows the population-level 95% CI."
    ),
    ha="right",
    va="bottom",
    fontsize=7.5,
    color="#666666",
)


plt.tight_layout(
    rect=[
        0.02,
        0.04,
        1,
        1,
    ]
)


plt.savefig(
    PNG_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.savefig(
    PDF_PATH,
    bbox_inches="tight",
)

plt.close()


print()
print("=" * 100)
print("SAVED")
print("=" * 100)

print(PNG_PATH)
print(PDF_PATH)
print(CSV_PATH)
