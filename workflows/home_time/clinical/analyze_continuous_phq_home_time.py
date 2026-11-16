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

OUTPUT_DIR = ROOT / "continuous_phq_home_time"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PNG_PATH = OUTPUT_DIR / "continuous_phq_home_time.png"
PDF_PATH = OUTPUT_DIR / "continuous_phq_home_time.pdf"
CSV_PATH = OUTPUT_DIR / "continuous_phq_home_time_stats.csv"
DATA_PATH = OUTPUT_DIR / "continuous_phq_home_time_analysis.csv"


# =============================================================================
# SETTINGS
# =============================================================================

RANDOM_SEED = 20261102

POINT_COLOR = "#9EC9DE"
LINE_COLOR = "#173650"
BAND_COLOR = "#BFCBD4"


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


# Equivalent hours/day based on supported GPS time
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


analysis.to_csv(
    DATA_PATH,
    index=False,
)


print("=" * 100)
print("CONTINUOUS PHQ-9 → TIME AT HOME")
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
    "PHQ-9 range:",
    analysis[
        "depression_score"
    ].min(),
    "to",
    analysis[
        "depression_score"
    ].max(),
)


# =============================================================================
# CLUSTERED OLS
# =============================================================================

X = sm.add_constant(
    analysis[
        [
            "depression_score",
        ]
    ]
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


intercept = float(
    result.params[
        "const"
    ]
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

p_value = float(
    result.pvalues[
        "depression_score"
    ]
)


ci = result.conf_int().loc[
    "depression_score"
]

ci_low_per_point = float(
    ci.iloc[0]
)

ci_high_per_point = float(
    ci.iloc[1]
)

ci_low_per_5 = (
    ci_low_per_point
    * 5.0
)

ci_high_per_5 = (
    ci_high_per_point
    * 5.0
)


# =============================================================================
# SPEARMAN DESCRIPTIVE ASSOCIATION
# =============================================================================

spearman_rho = float(
    analysis[
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


# =============================================================================
# PHQ-SCORE DESCRIPTIVES
# =============================================================================

score_summary = (
    analysis.groupby(
        "depression_score",
        as_index=False,
    )
    .agg(
        mean_home_hours=(
            "equivalent_home_hours",
            "mean",
        ),

        median_home_hours=(
            "equivalent_home_hours",
            "median",
        ),

        n=(
            "equivalent_home_hours",
            "size",
        ),

        n_participants=(
            "participant_id",
            "nunique",
        ),
    )
)


# =============================================================================
# MODEL PREDICTION + CLUSTER-ROBUST CI
# =============================================================================

x_min = float(
    analysis[
        "depression_score"
    ].min()
)

x_max = float(
    analysis[
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


params = result.params.to_numpy()
cov = result.cov_params().to_numpy()


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


# =============================================================================
# OUTPUT STATS
# =============================================================================

stats = pd.DataFrame(
    [
        {
            "n_weekly_observations":
                len(analysis),

            "n_participants":
                analysis[
                    "participant_id"
                ].nunique(),

            "phq_min":
                x_min,

            "phq_max":
                x_max,

            "slope_hours_per_1_phq_point":
                slope_per_point,

            "slope_hours_per_5_phq_points":
                slope_per_5,

            "ci_low_hours_per_5_phq_points":
                ci_low_per_5,

            "ci_high_hours_per_5_phq_points":
                ci_high_per_5,

            "clustered_p_value":
                p_value,

            "spearman_rho":
                spearman_rho,
        }
    ]
)


stats.to_csv(
    CSV_PATH,
    index=False,
)


print()
print("=" * 100)
print("RESULT")
print("=" * 100)

print(
    f"Slope per PHQ-9 point: "
    f"{slope_per_point:+.4f} h/day"
)

print(
    f"Slope per 5 PHQ-9 points: "
    f"{slope_per_5:+.2f} h/day"
)

print(
    f"95% CI per 5 points: "
    f"[{ci_low_per_5:+.2f}, "
    f"{ci_high_per_5:+.2f}]"
)

print(
    f"Cluster-adjusted p: "
    f"{p_value:.6f}"
)

print(
    f"Spearman rho: "
    f"{spearman_rho:+.3f}"
)


# =============================================================================
# FIGURE
# =============================================================================

plt.rcParams.update(
    {
        "font.family":
            "Arial",

        "font.size":
            10,

        "axes.titlesize":
            11,

        "axes.labelsize":
            10,

        "xtick.labelsize":
            9,

        "ytick.labelsize":
            9,
    }
)


fig, ax = plt.subplots(
    figsize=(
        7.2,
        4.6,
    )
)


rng = np.random.default_rng(
    RANDOM_SEED
)


# Small horizontal jitter because PHQ-9 is integer-valued
x_jitter = (
    analysis[
        "depression_score"
    ].to_numpy()
    + rng.normal(
        0,
        0.10,
        len(
            analysis
        ),
    )
)


ax.scatter(
    x_jitter,
    analysis[
        "equivalent_home_hours"
    ],
    s=18,
    alpha=0.25,
    color=POINT_COLOR,
    edgecolor="none",
    zorder=1,
)


# Cluster-robust confidence band
ax.fill_between(
    x_grid,
    pred_low,
    pred_high,
    color=BAND_COLOR,
    alpha=0.35,
    linewidth=0,
    zorder=2,
)


# Fitted trend
ax.plot(
    x_grid,
    pred_mean,
    color=LINE_COLOR,
    linewidth=2.0,
    zorder=3,
)


# PHQ-score means where enough observations exist
score_plot = score_summary[
    score_summary[
        "n"
    ] >= 3
].copy()


ax.scatter(
    score_plot[
        "depression_score"
    ],
    score_plot[
        "mean_home_hours"
    ],
    s=28,
    color=LINE_COLOR,
    edgecolor="white",
    linewidth=0.5,
    zorder=4,
)


# =============================================================================
# ANNOTATION
# =============================================================================

if p_value < 0.001:
    p_text = "p < 0.001"
else:
    p_text = f"p = {p_value:.3f}"


annotation = (
    f"Δ {slope_per_5:+.2f} h/day "
    f"per 5-point PHQ-9 increase\n"
    f"95% CI {ci_low_per_5:+.2f} to "
    f"{ci_high_per_5:+.2f}; "
    f"{p_text}"
)


ax.text(
    0.03,
    0.96,
    annotation,
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=9.5,
    color=LINE_COLOR,
)


# =============================================================================
# STYLE
# =============================================================================

ax.set_title(
    "Time at home across depressive symptom severity",
    loc="left",
    fontweight="bold",
    pad=10,
)


ax.set_xlabel(
    "PHQ-9 score"
)


ax.set_ylabel(
    "Equivalent time at home (hours/day)"
)


ax.set_xlim(
    max(
        -0.5,
        x_min - 0.5,
    ),
    min(
        27.5,
        x_max + 0.5,
    ),
)


ax.set_ylim(
    0,
    24,
)


ax.set_xticks(
    np.arange(
        0,
        28,
        5,
    )
)


ax.grid(
    axis="both",
    linewidth=0.5,
    alpha=0.15,
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
    0.015,
    (
        "Points represent weekly clinical windows; "
        "inference clustered by participant."
    ),
    ha="right",
    va="bottom",
    fontsize=8,
    color="#666666",
)


plt.tight_layout(
    rect=[
        0.02,
        0.05,
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
print(DATA_PATH)
