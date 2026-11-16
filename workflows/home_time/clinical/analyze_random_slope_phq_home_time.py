from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/"
    "digital_clinic/feature_reliability/gps_clinical"
)

INPUT_PATH = ROOT / "gps_clinical_weekly_features_with_home.parquet"

OUTPUT_DIR = ROOT / "random_slope_phq_home_time"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PNG_PATH = OUTPUT_DIR / "random_slope_phq_home_time.png"
PDF_PATH = OUTPUT_DIR / "random_slope_phq_home_time.pdf"
CSV_PATH = OUTPUT_DIR / "random_slope_phq_home_time.csv"


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


# Center PHQ around overall mean.
analysis["phq_centered"] = (
    analysis["depression_score"]
    - analysis["depression_score"].mean()
)


print("=" * 100)
print("RANDOM-SLOPE PHQ-9 → TIME AT HOME")
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

print(
    "Mean PHQ-9:",
    f"{analysis['depression_score'].mean():.2f}",
)


# =============================================================================
# MIXED MODEL
# =============================================================================
#
# Population:
#
# home_time ~ PHQ
#
# Participant:
#
# random intercept + random PHQ slope
#
# =============================================================================

model = smf.mixedlm(
    "equivalent_home_hours ~ phq_centered",
    data=analysis,
    groups=analysis["participant_id"],
    re_formula="~phq_centered",
)


result = model.fit(
    reml=False,
    method="lbfgs",
    maxiter=2000,
    disp=False,
)


population_slope_per_point = float(
    result.params[
        "phq_centered"
    ]
)

population_slope_per_5 = (
    population_slope_per_point
    * 5.0
)


population_ci = (
    result
    .conf_int()
    .loc[
        "phq_centered"
    ]
)


population_ci_low_5 = (
    float(
        population_ci.iloc[0]
    )
    * 5.0
)

population_ci_high_5 = (
    float(
        population_ci.iloc[1]
    )
    * 5.0
)

population_p = float(
    result.pvalues[
        "phq_centered"
    ]
)


# =============================================================================
# PARTICIPANT-SPECIFIC SHRUNK SLOPES
# =============================================================================

rows = []


for participant_id, group in analysis.groupby(
    "participant_id",
    sort=True,
):

    random_effect = result.random_effects[
        participant_id
    ]


    # statsmodels naming can vary slightly; identify slope component.
    slope_random = np.nan

    for name, value in random_effect.items():

        if "phq_centered" in str(name):

            slope_random = float(
                value
            )

            break


    if not np.isfinite(
        slope_random
    ):

        raise RuntimeError(
            f"Could not identify random slope for {participant_id}: "
            f"{random_effect}"
        )


    personal_slope_per_point = (
        population_slope_per_point
        + slope_random
    )

    personal_slope_per_5 = (
        personal_slope_per_point
        * 5.0
    )


    rows.append(
        {
            "participant_id":
                participant_id,

            "n_observations":
                len(group),

            "phq_min":
                float(
                    group[
                        "depression_score"
                    ].min()
                ),

            "phq_max":
                float(
                    group[
                        "depression_score"
                    ].max()
                ),

            "phq_range":
                float(
                    group[
                        "depression_score"
                    ].max()
                    - group[
                        "depression_score"
                    ].min()
                ),

            "random_slope_deviation_per_5":
                slope_random
                * 5.0,

            "personal_slope_hours_per_5_phq":
                personal_slope_per_5,
        }
    )


personal = pd.DataFrame(
    rows
)


personal = personal.sort_values(
    "personal_slope_hours_per_5_phq"
).reset_index(
    drop=True
)


personal.to_csv(
    CSV_PATH,
    index=False,
)


# =============================================================================
# SUMMARY
# =============================================================================

positive = int(
    (
        personal[
            "personal_slope_hours_per_5_phq"
        ] > 0
    ).sum()
)

negative = int(
    (
        personal[
            "personal_slope_hours_per_5_phq"
        ] < 0
    ).sum()
)


print()
print("=" * 100)
print("POPULATION EFFECT")
print("=" * 100)

print(
    "Slope per 5 PHQ-9 points:",
    f"{population_slope_per_5:+.2f} h/day",
)

print(
    "95% CI:",
    (
        f"[{population_ci_low_5:+.2f}, "
        f"{population_ci_high_5:+.2f}]"
    ),
)

print(
    "p:",
    f"{population_p:.6f}",
)


print()
print("=" * 100)
print("PERSONALIZED SHRUNK SLOPES")
print("=" * 100)

print(
    "Positive participant slopes:",
    positive,
)

print(
    "Negative participant slopes:",
    negative,
)

print(
    "Median personalized slope:",
    (
        f"{personal['personal_slope_hours_per_5_phq'].median():+.2f} "
        "h/day"
    ),
)

print(
    "Range:",
    (
        f"{personal['personal_slope_hours_per_5_phq'].min():+.2f} "
        "to "
        f"{personal['personal_slope_hours_per_5_phq'].max():+.2f} "
        "h/day"
    ),
)


print()
print(
    personal[
        [
            "participant_id",
            "n_observations",
            "phq_min",
            "phq_max",
            "phq_range",
            "personal_slope_hours_per_5_phq",
        ]
    ]
    .to_string(
        index=False
    )
)


# =============================================================================
# FIGURE
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
    6,
    0.27 * len(personal) + 1.8,
)


fig, ax = plt.subplots(
    figsize=(
        7.0,
        fig_height,
    )
)


y = np.arange(
    len(personal)
)


colors = np.where(
    personal[
        "personal_slope_hours_per_5_phq"
    ] >= 0,
    "#C77C45",
    "#4F88A8",
)


ax.scatter(
    personal[
        "personal_slope_hours_per_5_phq"
    ],
    y,
    s=36,
    c=colors,
    edgecolor="white",
    linewidth=0.5,
    zorder=3,
)


# Zero line
ax.axvline(
    0,
    color="#555555",
    linestyle="--",
    linewidth=1,
)


# Population estimate
ax.axvline(
    population_slope_per_5,
    color="#173650",
    linewidth=1.5,
)


ax.axvspan(
    population_ci_low_5,
    population_ci_high_5,
    color="#DCE6EB",
    alpha=0.45,
    zorder=0,
)


ax.set_yticks(
    y
)

ax.set_yticklabels(
    personal[
        "participant_id"
    ]
)


ax.invert_yaxis()


ax.set_xlabel(
    "Participant-specific change in time at home\n"
    "per 5-point PHQ-9 increase (hours/day)"
)


ax.set_title(
    "Heterogeneity in depression-related time at home",
    loc="left",
    fontweight="bold",
)


ax.grid(
    axis="x",
    alpha=0.12,
    linewidth=0.5,
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


fig.text(
    0.99,
    0.012,
    (
        "Participant slopes are random-slope estimates with partial pooling; "
        "blue band shows population 95% CI."
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
