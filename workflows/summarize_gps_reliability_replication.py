"""Summarize calibration and independent-holdout GPS reliability replication.

Reads already-computed frozen outputs only.
No model fitting or tuning is performed.

Primary question:
Does adding temporal missingness geometry improve prediction beyond coverage
alone, and does that improvement replicate in an independent participant holdout?
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

OLD_ROOT = Path(
    "/shared/Digital Clinic/"
    "raphe_outputs/digital_clinic/"
    "feature_reliability/gps_observability"
)

NEW_ROOT = Path(
    "/shared/Digital Clinic/"
    "cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_observability"
)

NEW_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


CALIBRATION_PATH = (
    OLD_ROOT
    / "gps_missingness_continuum_bootstrap_lopo_ci.csv"
)

HOLDOUT_PERFORMANCE_PATH = (
    OLD_ROOT
    / "gps_reliability_holdout_frozen_model_performance.csv"
)

HOLDOUT_BOOTSTRAP_PATH = (
    OLD_ROOT
    / "gps_reliability_holdout_frozen_model_bootstrap.csv"
)


SUMMARY_PATH = (
    NEW_ROOT
    / "gps_reliability_calibration_vs_holdout_summary.csv"
)

TABLE_PATH = (
    NEW_ROOT
    / "gps_reliability_calibration_vs_holdout_table.csv"
)

FIGURE_PATH = (
    NEW_ROOT
    / "gps_reliability_calibration_vs_holdout_replication.png"
)

FIGURE_PDF_PATH = (
    NEW_ROOT
    / "gps_reliability_calibration_vs_holdout_replication.pdf"
)

MODEL_PERFORMANCE_FIGURE_PATH = (
    NEW_ROOT
    / "gps_reliability_coverage_vs_geometry_performance.png"
)


# ---------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------

calibration = pd.read_csv(
    CALIBRATION_PATH
)

holdout_performance = pd.read_csv(
    HOLDOUT_PERFORMANCE_PATH
)

holdout_bootstrap = pd.read_csv(
    HOLDOUT_BOOTSTRAP_PATH
)


# ---------------------------------------------------------------------
# Basic integrity
# ---------------------------------------------------------------------

expected_features = {
    "entropy",
    "dominant_fraction",
    "effective_places",
}


for name, df in [
    (
        "calibration",
        calibration,
    ),
    (
        "holdout performance",
        holdout_performance,
    ),
    (
        "holdout bootstrap",
        holdout_bootstrap,
    ),
]:

    found = set(
        df[
            "feature"
        ].astype(str)
    )

    if found != expected_features:

        raise RuntimeError(
            f"{name}: unexpected feature set: {found}"
        )


if len(
    calibration
) != 3:

    raise RuntimeError(
        "Calibration summary must contain exactly 3 rows."
    )


if len(
    holdout_performance
) != 3:

    raise RuntimeError(
        "Holdout performance must contain exactly 3 rows."
    )


if len(
    holdout_bootstrap
) != 3:

    raise RuntimeError(
        "Holdout bootstrap must contain exactly 3 rows."
    )


# ---------------------------------------------------------------------
# Build stage-specific tables
# ---------------------------------------------------------------------

calibration = calibration.copy()

calibration[
    "stage"
] = "Calibration"

calibration[
    "participants_total"
] = calibration[
    "participants"
]

calibration[
    "participants_normalized"
] = calibration[
    "participants"
]


holdout = holdout_performance.merge(
    holdout_bootstrap[
        [
            "feature",
            "mae_reduction_ci95_lower",
            "mae_reduction_ci95_upper",
            "bootstrap_probability_mae_improvement",
            "rmse_reduction_ci95_lower",
            "rmse_reduction_ci95_upper",
            "bootstrap_probability_rmse_improvement",
            "r2_gain_ci95_lower",
            "r2_gain_ci95_upper",
            "bootstrap_probability_r2_gain",
        ]
    ],
    on="feature",
    how="left",
    validate="one_to_one",
)

holdout[
    "stage"
] = "Independent holdout"


# ---------------------------------------------------------------------
# Harmonize
# ---------------------------------------------------------------------

columns = [
    "stage",
    "feature",
    "participants_total",
    "participants_normalized",
    "predictions",
    "coverage_only_mae",
    "coverage_geometry_mae",
    "mae_reduction_fraction",
    "mae_reduction_ci95_lower",
    "mae_reduction_ci95_upper",
    "bootstrap_probability_mae_improvement",
    "coverage_only_rmse",
    "coverage_geometry_rmse",
    "rmse_reduction_fraction",
    "rmse_reduction_ci95_lower",
    "rmse_reduction_ci95_upper",
    "bootstrap_probability_rmse_improvement",
    "coverage_only_r2",
    "coverage_geometry_r2",
    "r2_gain",
    "r2_gain_ci95_lower",
    "r2_gain_ci95_upper",
    "bootstrap_probability_r2_gain",
]


combined = pd.concat(
    [
        calibration[
            columns
        ],
        holdout[
            columns
        ],
    ],
    ignore_index=True,
)


feature_order = [
    "dominant_fraction",
    "effective_places",
    "entropy",
]

feature_labels = {
    "dominant_fraction":
        "Dominant-place fraction",

    "effective_places":
        "Effective places",

    "entropy":
        "Location entropy",
}


combined[
    "feature_order"
] = combined[
    "feature"
].map(
    {
        feature: index
        for index, feature
        in enumerate(
            feature_order
        )
    }
)


combined = combined.sort_values(
    [
        "feature_order",
        "stage",
    ]
).reset_index(
    drop=True
)


combined.to_csv(
    SUMMARY_PATH,
    index=False,
)


# ---------------------------------------------------------------------
# Human-readable comparison table
# ---------------------------------------------------------------------

table = combined[
    [
        "stage",
        "feature",
        "participants_normalized",
        "predictions",
        "coverage_only_mae",
        "coverage_geometry_mae",
        "mae_reduction_fraction",
        "mae_reduction_ci95_lower",
        "mae_reduction_ci95_upper",
        "coverage_only_rmse",
        "coverage_geometry_rmse",
        "rmse_reduction_fraction",
        "rmse_reduction_ci95_lower",
        "rmse_reduction_ci95_upper",
        "coverage_only_r2",
        "coverage_geometry_r2",
        "r2_gain",
        "r2_gain_ci95_lower",
        "r2_gain_ci95_upper",
    ]
].copy()


table[
    "feature"
] = table[
    "feature"
].map(
    feature_labels
)


table[
    "mae_reduction_percent"
] = (
    100
    * table[
        "mae_reduction_fraction"
    ]
)


table[
    "mae_reduction_ci95"
] = table.apply(
    lambda row:
        (
            f"{100 * row['mae_reduction_ci95_lower']:.1f}% "
            f"to "
            f"{100 * row['mae_reduction_ci95_upper']:.1f}%"
        ),
    axis=1,
)


table[
    "rmse_reduction_percent"
] = (
    100
    * table[
        "rmse_reduction_fraction"
    ]
)


table[
    "rmse_reduction_ci95"
] = table.apply(
    lambda row:
        (
            f"{100 * row['rmse_reduction_ci95_lower']:.1f}% "
            f"to "
            f"{100 * row['rmse_reduction_ci95_upper']:.1f}%"
        ),
    axis=1,
)


table[
    "r2_gain_ci95"
] = table.apply(
    lambda row:
        (
            f"{row['r2_gain_ci95_lower']:.3f} "
            f"to "
            f"{row['r2_gain_ci95_upper']:.3f}"
        ),
    axis=1,
)


table.to_csv(
    TABLE_PATH,
    index=False,
)


# ---------------------------------------------------------------------
# Replication figure:
# effect of adding missingness geometry
# ---------------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.8, 4.6),
)

stage_order = [
    "Calibration",
    "Independent holdout",
]

offsets = {
    "Calibration": -0.10,
    "Independent holdout": 0.10,
}

x = np.arange(
    len(feature_order)
)

short_labels = {
    "dominant_fraction":
        "Dominant-place\nfraction",

    "effective_places":
        "Effective\nplaces",

    "entropy":
        "Location\nentropy",
}


# -----------------------------
# Panel A: MAE reduction
# -----------------------------

ax = axes[0]


for stage in stage_order:

    stage_df = (
        combined[
            combined[
                "stage"
            ]
            == stage
        ]
        .set_index(
            "feature"
        )
        .loc[
            feature_order
        ]
    )


    y = (
        100
        * stage_df[
            "mae_reduction_fraction"
        ].to_numpy()
    )


    lower = (
        100
        * (
            stage_df[
                "mae_reduction_fraction"
            ]
            - stage_df[
                "mae_reduction_ci95_lower"
            ]
        ).to_numpy()
    )


    upper = (
        100
        * (
            stage_df[
                "mae_reduction_ci95_upper"
            ]
            - stage_df[
                "mae_reduction_fraction"
            ]
        ).to_numpy()
    )


    positions = (
        x
        + offsets[
            stage
        ]
    )


    ax.errorbar(
        positions,
        y,
        yerr=[
            lower,
            upper,
        ],
        fmt="o",
        markersize=7,
        capsize=4,
        linewidth=1.8,
        label=stage,
    )


    for xpos, value in zip(
        positions,
        y,
    ):

        ax.annotate(
            f"{value:.1f}%",
            (
                xpos,
                value,
            ),
            xytext=(
                0,
                8,
            ),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


ax.axhline(
    0,
    linewidth=1,
    linestyle="--",
    color="0.35",
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    [
        short_labels[
            feature
        ]
        for feature
        in feature_order
    ]
)

ax.set_ylabel(
    "MAE reduction (%)"
)

ax.set_title(
    "A. Reduction in prediction error"
)

ax.legend(
    frameon=False,
    loc="upper right",
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


# -----------------------------
# Panel B: R² gain
# -----------------------------

ax = axes[1]


for stage in stage_order:

    stage_df = (
        combined[
            combined[
                "stage"
            ]
            == stage
        ]
        .set_index(
            "feature"
        )
        .loc[
            feature_order
        ]
    )


    y = stage_df[
        "r2_gain"
    ].to_numpy()


    lower = (
        stage_df[
            "r2_gain"
        ]
        - stage_df[
            "r2_gain_ci95_lower"
        ]
    ).to_numpy()


    upper = (
        stage_df[
            "r2_gain_ci95_upper"
        ]
        - stage_df[
            "r2_gain"
        ]
    ).to_numpy()


    positions = (
        x
        + offsets[
            stage
        ]
    )


    ax.errorbar(
        positions,
        y,
        yerr=[
            lower,
            upper,
        ],
        fmt="o",
        markersize=7,
        capsize=4,
        linewidth=1.8,
        label=stage,
    )


    for xpos, value in zip(
        positions,
        y,
    ):

        ax.annotate(
            f"{value:.2f}",
            (
                xpos,
                value,
            ),
            xytext=(
                0,
                8,
            ),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


ax.axhline(
    0,
    linewidth=1,
    linestyle="--",
    color="0.35",
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    [
        short_labels[
            feature
        ]
        for feature
        in feature_order
    ]
)

ax.set_ylabel(
    "ΔR²"
)

ax.set_title(
    "B. Gain in explained variance"
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


fig.suptitle(
    "GPS reliability improvement from temporal missingness geometry",
    fontsize=14,
    y=0.98,
)


fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.94,
    ]
)


fig.savefig(
    FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    FIGURE_PDF_PATH,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ---------------------------------------------------------------------
# Coverage-only vs geometry absolute performance
# ---------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(
        10,
        5,
    )
)


labels = []

coverage_values = []

geometry_values = []


for feature in feature_order:

    for stage in stage_order:

        row = combined[
            (
                combined[
                    "feature"
                ]
                == feature
            )
            & (
                combined[
                    "stage"
                ]
                == stage
            )
        ].iloc[0]


        short_stage = (
            "Calibration"
            if stage
            == "Calibration"
            else "Holdout"
        )


        labels.append(
            f"{feature_labels[feature]}\n{short_stage}"
        )

        coverage_values.append(
            row[
                "coverage_only_mae"
            ]
        )

        geometry_values.append(
            row[
                "coverage_geometry_mae"
            ]
        )


positions = np.arange(
    len(
        labels
    )
)

width = 0.36


ax.bar(
    positions
    - width / 2,
    coverage_values,
    width,
    label="Coverage only",
)

ax.bar(
    positions
    + width / 2,
    geometry_values,
    width,
    label="Coverage + geometry",
)


ax.set_xticks(
    positions
)

ax.set_xticklabels(
    labels,
    rotation=18,
    ha="right",
)

ax.set_ylabel(
    "MAE of normalized feature error"
)

ax.set_title(
    "Frozen GPS reliability prediction performance"
)

ax.legend(
    frameon=False,
)


fig.tight_layout()


fig.savefig(
    MODEL_PERFORMANCE_FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ---------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------

print()
print("=" * 92)
print("CORTEX 2.0 GPS RELIABILITY — CALIBRATION VS INDEPENDENT HOLDOUT")
print("=" * 92)


report_columns = [
    "stage",
    "feature",
    "participants_normalized",
    "predictions",
    "coverage_only_mae",
    "coverage_geometry_mae",
    "mae_reduction_fraction",
    "mae_reduction_ci95_lower",
    "mae_reduction_ci95_upper",
    "coverage_only_r2",
    "coverage_geometry_r2",
    "r2_gain",
    "r2_gain_ci95_lower",
    "r2_gain_ci95_upper",
]


print(
    combined[
        report_columns
    ].to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


print()
print("=" * 92)
print("REPLICATION CHECK")
print("=" * 92)


for feature in feature_order:

    calibration_row = combined[
        (
            combined[
                "feature"
            ]
            == feature
        )
        & (
            combined[
                "stage"
            ]
            == "Calibration"
        )
    ].iloc[0]


    holdout_row = combined[
        (
            combined[
                "feature"
            ]
            == feature
        )
        & (
            combined[
                "stage"
            ]
            == "Independent holdout"
        )
    ].iloc[0]


    calibration_mae_supported = (
        calibration_row[
            "mae_reduction_ci95_lower"
        ]
        > 0
    )

    holdout_mae_supported = (
        holdout_row[
            "mae_reduction_ci95_lower"
        ]
        > 0
    )


    print(
        f"{feature_labels[feature]}:"
    )

    print(
        "  Calibration MAE reduction:",
        f"{100 * calibration_row['mae_reduction_fraction']:.1f}%",
        (
            f"[{100 * calibration_row['mae_reduction_ci95_lower']:.1f}%, "
            f"{100 * calibration_row['mae_reduction_ci95_upper']:.1f}%]"
        ),
    )

    print(
        "  Holdout MAE reduction:",
        f"{100 * holdout_row['mae_reduction_fraction']:.1f}%",
        (
            f"[{100 * holdout_row['mae_reduction_ci95_lower']:.1f}%, "
            f"{100 * holdout_row['mae_reduction_ci95_upper']:.1f}%]"
        ),
    )

    print(
        "  Positive CI calibration:",
        calibration_mae_supported,
    )

    print(
        "  Positive CI holdout:",
        holdout_mae_supported,
    )


print()
print("=" * 92)
print("OUTPUTS")
print("=" * 92)

print(
    SUMMARY_PATH
)

print(
    TABLE_PATH
)

print(
    FIGURE_PATH
)

print(
    FIGURE_PDF_PATH
)

print(
    MODEL_PERFORMANCE_FIGURE_PATH
)
