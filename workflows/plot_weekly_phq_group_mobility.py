from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/"
    "digital_clinic/feature_reliability/gps_clinical"
)

IN_PATH = ROOT / "gps_clinical_weekly_features_with_home.parquet"

OUT_DIR = ROOT / "weekly_phq_group_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_PATH = OUT_DIR / "weekly_phq_group_mobility_markers.png"
CSV_PATH = OUT_DIR / "weekly_phq_group_mobility_markers_stats.csv"

# ============================================================
# LOAD
# ============================================================

df = pd.read_parquet(IN_PATH).copy()

print("=" * 90)
print("WEEKLY PHQ GROUP MOBILITY FIGURE")
print("=" * 90)

print("Input rows:", len(df))
print("Participants:", df["participant_id"].nunique())

df = df[
    df["depression_score"].notna()
].copy()

# ============================================================
# PHQ GROUPS
# ============================================================

def phq_group(x):

    if pd.isna(x):
        return np.nan

    if 0 <= x <= 4:
        return "Lower PHQ-9 (0–4)"

    if x >= 5:
        return "Elevated PHQ-9 (5+)"

    return np.nan


df["phq_group"] = (
    df["depression_score"]
    .apply(phq_group)
)

df = df[
    df["phq_group"].notna()
].copy()

group_order = [
    "Lower PHQ-9 (0–4)",
    "Elevated PHQ-9 (5+)",
]

# ============================================================
# CLINICALLY FRIENDLY FEATURES
# ============================================================

df["mobility_diversity"] = (
    df["location_entropy_mean"]
)

df["time_at_home_pct"] = (
    df["time_at_home_pct"]
)

df["daily_movement_distance_km"] = (
    df["displacement_distance_mean_m"]
    / 1000.0
)

feature_specs = [
    {
        "column": "mobility_diversity",
        "title": "Mobility diversity",
        "ylabel": "Weekly mean mobility diversity",
    },
    {
        "column": "time_at_home_pct",
        "title": "Time at home",
        "ylabel": "Weekly mean time at home (%)",
    },
    {
        "column": "daily_movement_distance_km",
        "title": "Daily movement distance",
        "ylabel": "Weekly mean movement distance (km/day)",
    },
]

for spec in feature_specs:

    col = spec["column"]

    df.loc[
        ~np.isfinite(
            pd.to_numeric(
                df[col],
                errors="coerce",
            )
        ),
        col,
    ] = np.nan

# ============================================================
# HELPERS
# ============================================================

group_colors = {
    "Lower PHQ-9 (0–4)": "#4C92B9",
    "Elevated PHQ-9 (5+)": "#D98C5A",
}


def mean_ci(values):

    x = pd.Series(
        values
    ).dropna()

    n = len(x)

    if n == 0:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    mean = float(
        x.mean()
    )

    if n == 1:
        return (
            mean,
            mean,
            mean,
        )

    se = float(
        x.std(ddof=1)
        / np.sqrt(n)
    )

    ci = 1.96 * se

    return (
        mean,
        mean - ci,
        mean + ci,
    )


def format_p(p):

    if pd.isna(p):
        return "p = NA"

    if p < 0.001:
        return "p < 0.001"

    return f"p = {p:.3f}"


def fit_mixed_model(
    data,
    outcome_col,
):

    x = data[
        [
            "participant_id",
            "phq_group",
            outcome_col,
        ]
    ].dropna().copy()

    x["phq_group"] = pd.Categorical(
        x["phq_group"],
        categories=group_order,
        ordered=True,
    )

    n_rows = len(x)

    n_participants = (
        x[
            "participant_id"
        ].nunique()
    )

    lower_mean = float(
        x.loc[
            x[
                "phq_group"
            ].eq(
                group_order[0]
            ),
            outcome_col,
        ]
        .mean()
    )

    elevated_mean = float(
        x.loc[
            x[
                "phq_group"
            ].eq(
                group_order[1]
            ),
            outcome_col,
        ]
        .mean()
    )

    raw_difference = (
        elevated_mean
        - lower_mean
    )

    try:

        model = smf.mixedlm(
            formula=(
                f"{outcome_col} ~ "
                "C(phq_group)"
            ),
            data=x,
            groups=x[
                "participant_id"
            ],
        )

        result = model.fit(
            reml=False,
            method="lbfgs",
            maxiter=1000,
            disp=False,
        )

        term = (
            "C(phq_group)"
            "[T.Elevated PHQ-9 (5+)]"
        )

        adjusted_beta = float(
            result.params.get(
                term,
                np.nan,
            )
        )

        p_value = float(
            result.pvalues.get(
                term,
                np.nan,
            )
        )

        if (
            term
            in result.params.index
        ):

            ci = (
                result
                .conf_int()
                .loc[
                    term
                ]
            )

            adjusted_ci_low = float(
                ci.iloc[0]
            )

            adjusted_ci_high = float(
                ci.iloc[1]
            )

        else:

            adjusted_ci_low = np.nan
            adjusted_ci_high = np.nan

    except Exception as exc:

        print(
            "Mixed model failed:",
            outcome_col,
            repr(exc),
        )

        adjusted_beta = np.nan
        adjusted_ci_low = np.nan
        adjusted_ci_high = np.nan
        p_value = np.nan

    return {
        "feature":
            outcome_col,

        "n_rows":
            n_rows,

        "n_participants":
            n_participants,

        "lower_mean":
            lower_mean,

        "elevated_mean":
            elevated_mean,

        "raw_difference_elevated_minus_lower":
            raw_difference,

        "adjusted_mixed_model_beta":
            adjusted_beta,

        "adjusted_ci_low":
            adjusted_ci_low,

        "adjusted_ci_high":
            adjusted_ci_high,

        "p_value":
            p_value,
    }


# ============================================================
# STATS
# ============================================================

stats_rows = []

for spec in feature_specs:

    row = fit_mixed_model(
        df,
        spec["column"],
    )

    row[
        "display_title"
    ] = spec[
        "title"
    ]

    stats_rows.append(
        row
    )


stats_df = pd.DataFrame(
    stats_rows
)

stats_df.to_csv(
    CSV_PATH,
    index=False,
)

print()
print("=" * 90)
print("GROUP COUNTS")
print("=" * 90)

print(
    df[
        "phq_group"
    ]
    .value_counts()
    .to_string()
)

print()
print("=" * 90)
print("MODEL SUMMARY")
print("=" * 90)

print(
    stats_df[
        [
            "display_title",
            "n_rows",
            "n_participants",
            "lower_mean",
            "elevated_mean",
            "raw_difference_elevated_minus_lower",
            "adjusted_mixed_model_beta",
            "adjusted_ci_low",
            "adjusted_ci_high",
            "p_value",
        ]
    ]
    .to_string(
        index=False
    )
)

# ============================================================
# FIGURE
# ============================================================

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "figure.titlesize": 20,
    }
)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(
        15,
        5.4,
    ),
)

fig.suptitle(
    "Mobility patterns by depression symptom level",
    y=1.03,
    fontweight="bold",
)

panel_labels = [
    "A",
    "B",
    "C",
]

rng = np.random.default_rng(
    20261102
)

for (
    ax,
    spec,
    panel_label,
) in zip(
    axes,
    feature_specs,
    panel_labels,
):

    col = spec[
        "column"
    ]

    for group_i, group in enumerate(
        group_order
    ):

        y = (
            df.loc[
                df[
                    "phq_group"
                ].eq(
                    group
                ),
                col,
            ]
            .dropna()
            .to_numpy()
        )

        jitter = rng.normal(
            0,
            0.045,
            size=len(y),
        )

        ax.scatter(
            np.full(
                len(y),
                group_i,
            )
            + jitter,
            y,
            s=34,
            alpha=0.78,
            color=(
                group_colors[
                    group
                ]
            ),
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )

        (
            mean,
            lo,
            hi,
        ) = mean_ci(
            y
        )

        ax.errorbar(
            group_i,
            mean,
            yerr=[
                [
                    mean - lo
                ],
                [
                    hi - mean
                ],
            ],
            fmt="_",
            markersize=30,
            elinewidth=3,
            capsize=0,
            color="#17324D",
            zorder=4,
        )

    stat_row = (
        stats_df.loc[
            stats_df[
                "feature"
            ].eq(
                col
            )
        ]
        .iloc[0]
    )

    lower_mean = stat_row[
        "lower_mean"
    ]

    elevated_mean = stat_row[
        "elevated_mean"
    ]

    raw_difference = stat_row[
        "raw_difference_elevated_minus_lower"
    ]

    p_value = stat_row[
        "p_value"
    ]

    stat_text = (
        f"Lower: {lower_mean:.2f}\n"
        f"Elevated: {elevated_mean:.2f}\n"
        f"Difference: {raw_difference:+.2f}\n"
        f"{format_p(p_value)}"
    )

    ax.text(
        0.03,
        0.97,
        stat_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
    )

    ax.set_title(
        spec[
            "title"
        ],
        pad=12,
    )

    ax.set_ylabel(
        spec[
            "ylabel"
        ]
    )

    ax.set_xticks(
        [
            0,
            1,
        ]
    )

    ax.set_xticklabels(
        [
            "Lower",
            "Elevated",
        ]
    )

    ax.set_xlabel(
        "Depression symptoms"
    )

    ax.grid(
        axis="y",
        alpha=0.22,
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
        -0.17,
        1.08,
        panel_label,
        transform=ax.transAxes,
        fontsize=19,
        fontweight="bold",
    )

legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=(
            group_colors[
                group_order[0]
            ]
        ),
        markersize=9,
        label=group_order[0],
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=(
            group_colors[
                group_order[1]
            ]
        ),
        markersize=9,
        label=group_order[1],
    ),
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=2,
    bbox_to_anchor=(
        0.5,
        -0.035,
    ),
    frameon=False,
    fontsize=12,
)

plt.tight_layout(
    rect=[
        0,
        0.06,
        1,
        0.96,
    ]
)

plt.savefig(
    FIG_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print()
print("=" * 90)
print("SAVED")
print("=" * 90)

print(FIG_PATH)
print(CSV_PATH)
