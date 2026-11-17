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

OUTPUT_DIR = ROOT / "continuous_phq_other_mobility"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PNG_PATH = OUTPUT_DIR / "continuous_phq_other_mobility.png"
PDF_PATH = OUTPUT_DIR / "continuous_phq_other_mobility.pdf"
CSV_PATH = OUTPUT_DIR / "continuous_phq_other_mobility_stats.csv"


# =============================================================================
# SETTINGS
# =============================================================================

POINT_COLOR = "#9EC9DE"
LINE_COLOR = "#173650"
BAND_COLOR = "#BFCBD4"
RANDOM_SEED = 20261102


# =============================================================================
# HELPERS
# =============================================================================

def clustered_continuous_model(
    data,
    outcome_col,
):
    x = data[
        [
            "participant_id",
            "depression_score",
            outcome_col,
        ]
    ].dropna().copy()

    X = sm.add_constant(
        x[
            [
                "depression_score",
            ]
        ]
    )

    y = x[
        outcome_col
    ]

    result = sm.OLS(
        y,
        X,
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups":
                x[
                    "participant_id"
                ]
        },
    )

    slope = float(
        result.params[
            "depression_score"
        ]
    )

    p_value = float(
        result.pvalues[
            "depression_score"
        ]
    )

    ci = (
        result
        .conf_int()
        .loc[
            "depression_score"
        ]
    )

    ci_low = float(
        ci.iloc[0]
    )

    ci_high = float(
        ci.iloc[1]
    )

    spearman_rho = float(
        x[
            [
                "depression_score",
                outcome_col,
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

    return (
        x,
        result,
        slope,
        ci_low,
        ci_high,
        p_value,
        spearman_rho,
    )


def prediction_grid(
    x,
    result,
):
    x_min = float(
        x[
            "depression_score"
        ].min()
    )

    x_max = float(
        x[
            "depression_score"
        ].max()
    )

    x_grid = np.linspace(
        x_min,
        x_max,
        250,
    )

    pred_X = np.column_stack(
        [
            np.ones(
                len(x_grid)
            ),
            x_grid,
        ]
    )

    params = (
        result.params
        .to_numpy()
    )

    cov = (
        result
        .cov_params()
        .to_numpy()
    )

    pred_mean = (
        pred_X
        @ params
    )

    pred_var = np.einsum(
        "ij,jk,ik->i",
        pred_X,
        cov,
        pred_X,
    )

    pred_se = np.sqrt(
        np.maximum(
            pred_var,
            0,
        )
    )

    pred_low = (
        pred_mean
        - 1.96
        * pred_se
    )

    pred_high = (
        pred_mean
        + 1.96
        * pred_se
    )

    return (
        x_grid,
        pred_mean,
        pred_low,
        pred_high,
    )


def format_p(p):
    if p < 0.001:
        return "p < 0.001"

    return f"p = {p:.3f}"


# =============================================================================
# LOAD
# =============================================================================

df = pd.read_parquet(
    INPUT_PATH
).copy()


df[
    "depression_score"
] = pd.to_numeric(
    df[
        "depression_score"
    ],
    errors="coerce",
)


df[
    "mobility_diversity"
] = pd.to_numeric(
    df[
        "location_entropy_mean"
    ],
    errors="coerce",
)


df[
    "movement_distance_km"
] = (
    pd.to_numeric(
        df[
            "displacement_distance_mean_m"
        ],
        errors="coerce",
    )
    / 1000.0
)


df = df[
    df[
        "depression_score"
    ].between(
        0,
        27,
        inclusive="both",
    )
].copy()


# =============================================================================
# ANALYSES
# =============================================================================

results = []


# -----------------------------------------------------------------------------
# Mobility diversity
# -----------------------------------------------------------------------------

(
    diversity_data,
    diversity_result,
    diversity_slope,
    diversity_ci_low,
    diversity_ci_high,
    diversity_p,
    diversity_rho,
) = clustered_continuous_model(
    df,
    "mobility_diversity",
)


results.append(
    {
        "outcome":
            "Mobility diversity",

        "n_weekly_observations":
            len(
                diversity_data
            ),

        "n_participants":
            diversity_data[
                "participant_id"
            ].nunique(),

        "slope_per_1_phq":
            diversity_slope,

        "slope_per_5_phq":
            diversity_slope
            * 5.0,

        "ci_low_per_5_phq":
            diversity_ci_low
            * 5.0,

        "ci_high_per_5_phq":
            diversity_ci_high
            * 5.0,

        "clustered_p_value":
            diversity_p,

        "spearman_rho":
            diversity_rho,
    }
)


# -----------------------------------------------------------------------------
# Daily movement distance
# -----------------------------------------------------------------------------

(
    distance_data,
    distance_result,
    distance_slope,
    distance_ci_low,
    distance_ci_high,
    distance_p,
    distance_rho,
) = clustered_continuous_model(
    df,
    "movement_distance_km",
)


results.append(
    {
        "outcome":
            "Daily movement distance",

        "n_weekly_observations":
            len(
                distance_data
            ),

        "n_participants":
            distance_data[
                "participant_id"
            ].nunique(),

        "slope_per_1_phq":
            distance_slope,

        "slope_per_5_phq":
            distance_slope
            * 5.0,

        "ci_low_per_5_phq":
            distance_ci_low
            * 5.0,

        "ci_high_per_5_phq":
            distance_ci_high
            * 5.0,

        "clustered_p_value":
            distance_p,

        "spearman_rho":
            distance_rho,
    }
)


stats_df = pd.DataFrame(
    results
)


stats_df.to_csv(
    CSV_PATH,
    index=False,
)


# =============================================================================
# PRINT RESULTS
# =============================================================================

print("=" * 100)
print("CONTINUOUS PHQ-9 → GPS MOBILITY FEATURES")
print("=" * 100)


for _, row in stats_df.iterrows():

    print()

    print(
        row[
            "outcome"
        ]
    )

    print(
        "-" * 60
    )

    print(
        "Weekly observations:",
        int(
            row[
                "n_weekly_observations"
            ]
        ),
    )

    print(
        "Participants:",
        int(
            row[
                "n_participants"
            ]
        ),
    )

    print(
        "Slope per 5 PHQ-9 points:",
        f"{row['slope_per_5_phq']:+.4f}",
    )

    print(
        "95% CI per 5 points:",
        (
            f"["
            f"{row['ci_low_per_5_phq']:+.4f}, "
            f"{row['ci_high_per_5_phq']:+.4f}"
            f"]"
        ),
    )

    print(
        "Cluster-adjusted p:",
        f"{row['clustered_p_value']:.6f}",
    )

    print(
        "Spearman rho:",
        f"{row['spearman_rho']:+.3f}",
    )


# =============================================================================
# PREDICTIONS
# =============================================================================

(
    diversity_x_grid,
    diversity_pred,
    diversity_low,
    diversity_high,
) = prediction_grid(
    diversity_data,
    diversity_result,
)


(
    distance_x_grid,
    distance_pred,
    distance_low,
    distance_high,
) = prediction_grid(
    distance_data,
    distance_result,
)


# =============================================================================
# FIGURE
# =============================================================================

plt.rcParams.update(
    {
        "font.family":
            "DejaVu Sans",

        "font.size":
            9,

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


fig, axes = plt.subplots(
    1,
    2,
    figsize=(
        10,
        4.1,
    ),
)


rng = np.random.default_rng(
    RANDOM_SEED
)


# =============================================================================
# PANEL A — MOBILITY DIVERSITY
# =============================================================================

ax = axes[0]


x_jitter = (
    diversity_data[
        "depression_score"
    ].to_numpy()
    + rng.normal(
        0,
        0.10,
        len(
            diversity_data
        ),
    )
)


ax.scatter(
    x_jitter,
    diversity_data[
        "mobility_diversity"
    ],
    s=15,
    alpha=0.23,
    color=POINT_COLOR,
    edgecolor="none",
)


ax.fill_between(
    diversity_x_grid,
    diversity_low,
    diversity_high,
    color=BAND_COLOR,
    alpha=0.35,
    linewidth=0,
)


ax.plot(
    diversity_x_grid,
    diversity_pred,
    color=LINE_COLOR,
    linewidth=1.8,
)


ax.text(
    0.04,
    0.96,
    (
        f"Δ "
        f"{diversity_slope * 5:+.2f} "
        f"per 5 PHQ-9 points\n"
        f"95% CI "
        f"{diversity_ci_low * 5:+.2f} to "
        f"{diversity_ci_high * 5:+.2f}; "
        f"{format_p(diversity_p)}"
    ),
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=8.5,
    color=LINE_COLOR,
)


ax.set_title(
    "Mobility diversity",
    loc="left",
    fontweight="bold",
)


ax.set_xlabel(
    "PHQ-9 score"
)


ax.set_ylabel(
    "Weekly mean mobility diversity"
)


ax.set_xlim(
    -0.5,
    21.5,
)


ax.set_xticks(
    [
        0,
        5,
        10,
        15,
        20,
    ]
)


ax.grid(
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


ax.text(
    -0.12,
    1.05,
    "a",
    transform=ax.transAxes,
    fontsize=11,
    fontweight="bold",
)


# =============================================================================
# PANEL B — MOVEMENT DISTANCE
# =============================================================================

ax = axes[1]


x_jitter = (
    distance_data[
        "depression_score"
    ].to_numpy()
    + rng.normal(
        0,
        0.10,
        len(
            distance_data
        ),
    )
)


ax.scatter(
    x_jitter,
    distance_data[
        "movement_distance_km"
    ],
    s=15,
    alpha=0.23,
    color=POINT_COLOR,
    edgecolor="none",
)


ax.fill_between(
    distance_x_grid,
    distance_low,
    distance_high,
    color=BAND_COLOR,
    alpha=0.35,
    linewidth=0,
)


ax.plot(
    distance_x_grid,
    distance_pred,
    color=LINE_COLOR,
    linewidth=1.8,
)


ax.text(
    0.04,
    0.96,
    (
        f"Δ "
        f"{distance_slope * 5:+.1f} km/day "
        f"per 5 PHQ-9 points\n"
        f"95% CI "
        f"{distance_ci_low * 5:+.1f} to "
        f"{distance_ci_high * 5:+.1f}; "
        f"{format_p(distance_p)}"
    ),
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=8.5,
    color=LINE_COLOR,
)


ax.set_title(
    "Daily movement distance",
    loc="left",
    fontweight="bold",
)


ax.set_xlabel(
    "PHQ-9 score"
)


ax.set_ylabel(
    "Weekly mean movement distance (km/day)"
)


ax.set_xlim(
    -0.5,
    21.5,
)


ax.set_xticks(
    [
        0,
        5,
        10,
        15,
        20,
    ]
)


# Display only to 99th percentile,
# but analysis still uses all observations.
distance_display_max = float(
    np.nanpercentile(
        distance_data[
            "movement_distance_km"
        ],
        99,
    )
)


ax.set_ylim(
    0,
    distance_display_max
    * 1.08,
)


n_offscale = int(
    (
        distance_data[
            "movement_distance_km"
        ]
        > (
            distance_display_max
            * 1.08
        )
    )
    .sum()
)


if n_offscale > 0:

    ax.text(
        0.98,
        0.04,
        (
            f"{n_offscale} observation"
            + (
                "s"
                if n_offscale != 1
                else ""
            )
            + " above display range"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#777777",
    )


ax.grid(
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


ax.text(
    -0.12,
    1.05,
    "b",
    transform=ax.transAxes,
    fontsize=11,
    fontweight="bold",
)


# =============================================================================
# FOOTNOTE
# =============================================================================

fig.text(
    0.99,
    0.015,
    (
        "Points represent weekly clinical windows; "
        "95% CIs and p values use participant-clustered inference."
    ),
    ha="right",
    va="bottom",
    fontsize=7.5,
    color="#666666",
)


plt.tight_layout(
    rect=[
        0.02,
        0.05,
        1,
        1,
    ],
    w_pad=2.5,
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
