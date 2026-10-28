"""Summarize personalized GPS degradation -> clinical prediction degradation.

Combines:
- LPE511: displacement distance
- BPH745: dominant-place fraction
- VTE817: displacement distance

Important statistical unit:
    clinical assessment/window

The five degradation replicates are first averaged WITHIN each
participant/window/coverage/pattern. Uncertainty is then estimated by
bootstrapping clinical windows, not by pretending degradation replicates
are independent clinical observations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(
    "/shared/Digital Clinic/"
    "cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/"
    "personalized_clinical_degradation"
)

PARTICIPANTS = [
    "LPE511",
    "BPH745",
    "VTE817",
]

N_BOOT = 5000
BOOT_SEED = 20261028


# ============================================================
# HELPERS
# ============================================================

def bootstrap_mean_ci(
    values,
    *,
    n_boot=N_BOOT,
    seed=BOOT_SEED,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    n = len(values)

    if n == 0:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    mean = float(
        np.mean(values)
    )

    if n == 1:
        return (
            mean,
            mean,
            mean,
        )

    rng = np.random.default_rng(
        seed
    )

    boot = np.empty(
        n_boot,
        dtype=float,
    )

    for i in range(
        n_boot
    ):
        sample = rng.choice(
            values,
            size=n,
            replace=True,
        )

        boot[i] = np.mean(
            sample
        )

    low, high = np.percentile(
        boot,
        [
            2.5,
            97.5,
        ],
    )

    return (
        mean,
        float(low),
        float(high),
    )


# ============================================================
# LOAD
# ============================================================

frames = []

for participant_id in PARTICIPANTS:

    path = (
        ROOT
        / (
            f"{participant_id}_"
            "gps_clinical_degradation.parquet"
        )
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    df = pd.read_parquet(
        path
    )

    frames.append(
        df
    )


raw = pd.concat(
    frames,
    ignore_index=True,
)

raw[
    "missing_fraction"
] = (
    1.0
    - raw[
        "coverage"
    ]
)

raw[
    "missing_percent"
] = (
    100.0
    * raw[
        "missing_fraction"
    ]
)


print("=" * 100)
print(
    "PERSONALIZED GPS CLINICAL "
    "DEGRADATION SUMMARY"
)
print("=" * 100)

print(
    "Raw degradation rows:",
    len(raw),
)

print(
    "Participants:",
    sorted(
        raw[
            "participant_id"
        ].unique()
    ),
)


# ============================================================
# COLLAPSE REPLICATES WITHIN CLINICAL WINDOW
# ============================================================

window_level = (
    raw.groupby(
        [
            "participant_id",
            "clinical_feature",
            "clinical_window_id",
            "assessment_date",
            "coverage",
            "missing_fraction",
            "missing_percent",
            "pattern",
        ],
        as_index=False,
    )
    .agg(
        actual_score=(
            "actual_score",
            "first",
        ),

        reference_feature=(
            "reference_feature",
            "first",
        ),

        reference_prediction=(
            "reference_prediction",
            "first",
        ),

        reference_absolute_error=(
            "reference_absolute_error",
            "first",
        ),

        mean_degraded_feature=(
            "degraded_feature",
            "mean",
        ),

        mean_feature_absolute_error=(
            "feature_absolute_error",
            "mean",
        ),

        mean_feature_relative_error=(
            "feature_relative_error",
            "mean",
        ),

        mean_prediction_perturbation=(
            "prediction_perturbation",
            "mean",
        ),

        mean_degraded_absolute_error=(
            "degraded_absolute_error",
            "mean",
        ),

        mean_delta_mae=(
            "delta_mae",
            "mean",
        ),

        fraction_replicates_worsened=(
            "delta_mae",
            lambda x:
                float(
                    (
                        x.dropna()
                        > 0
                    ).mean()
                )
                if x.notna().any()
                else np.nan,
        ),

        mean_supported_fraction=(
            "mean_supported_fraction",
            "mean",
        ),

        mean_raw_fraction_retained=(
            "mean_raw_fraction_retained",
            "mean",
        ),

        n_replicates=(
            "replicate",
            "nunique",
        ),
    )
)


WINDOW_PATH = (
    ROOT
    / "combined_window_level_degradation.csv"
)

window_level.to_csv(
    WINDOW_PATH,
    index=False,
)


# ============================================================
# CONDITION SUMMARY + BOOTSTRAP CIs ACROSS WINDOWS
# ============================================================

summary_rows = []


group_columns = [
    "participant_id",
    "clinical_feature",
    "coverage",
    "missing_fraction",
    "missing_percent",
    "pattern",
]


for keys, group in (
    window_level.groupby(
        group_columns
    )
):

    (
        participant_id,
        clinical_feature,
        coverage,
        missing_fraction,
        missing_percent,
        pattern,
    ) = keys

    (
        delta_mean,
        delta_low,
        delta_high,
    ) = bootstrap_mean_ci(
        group[
            "mean_delta_mae"
        ]
    )

    (
        perturb_mean,
        perturb_low,
        perturb_high,
    ) = bootstrap_mean_ci(
        group[
            "mean_prediction_perturbation"
        ],
        seed=(
            BOOT_SEED
            + 1
        ),
    )

    (
        relative_mean,
        relative_low,
        relative_high,
    ) = bootstrap_mean_ci(
        group[
            "mean_feature_relative_error"
        ],
        seed=(
            BOOT_SEED
            + 2
        ),
    )

    summary_rows.append(
        {
            "participant_id":
                participant_id,

            "clinical_feature":
                clinical_feature,

            "coverage":
                coverage,

            "missing_fraction":
                missing_fraction,

            "missing_percent":
                missing_percent,

            "pattern":
                pattern,

            "n_clinical_windows":
                group[
                    "clinical_window_id"
                ].nunique(),

            "mean_delta_mae":
                delta_mean,

            "delta_mae_ci_low":
                delta_low,

            "delta_mae_ci_high":
                delta_high,

            "median_delta_mae":
                float(
                    group[
                        "mean_delta_mae"
                    ].median()
                ),

            "fraction_windows_worsened":
                float(
                    (
                        group[
                            "mean_delta_mae"
                        ]
                        > 0
                    ).mean()
                ),

            "mean_prediction_perturbation":
                perturb_mean,

            "prediction_perturbation_ci_low":
                perturb_low,

            "prediction_perturbation_ci_high":
                perturb_high,

            "mean_feature_relative_error":
                relative_mean,

            "feature_relative_error_ci_low":
                relative_low,

            "feature_relative_error_ci_high":
                relative_high,

            "median_feature_relative_error":
                float(
                    group[
                        "mean_feature_relative_error"
                    ].median()
                ),

            "mean_supported_fraction":
                float(
                    group[
                        "mean_supported_fraction"
                    ].mean()
                ),

            "mean_raw_fraction_retained":
                float(
                    group[
                        "mean_raw_fraction_retained"
                    ].mean()
                ),
        }
    )


summary = pd.DataFrame(
    summary_rows
).sort_values(
    [
        "participant_id",
        "pattern",
        "missing_percent",
    ]
)


SUMMARY_PATH = (
    ROOT
    / "combined_condition_summary.csv"
)

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)


# ============================================================
# PAIRED GEOMETRY CONTRAST
#
# Positive value:
# contiguous missingness is more harmful than distributed
# at the SAME participant/window/coverage.
# ============================================================

geometry_base = (
    window_level[
        [
            "participant_id",
            "clinical_feature",
            "clinical_window_id",
            "coverage",
            "missing_percent",
            "pattern",
            "mean_delta_mae",
            "mean_prediction_perturbation",
            "mean_feature_relative_error",
        ]
    ]
    .pivot(
        index=[
            "participant_id",
            "clinical_feature",
            "clinical_window_id",
            "coverage",
            "missing_percent",
        ],
        columns="pattern",
        values=[
            "mean_delta_mae",
            "mean_prediction_perturbation",
            "mean_feature_relative_error",
        ],
    )
    .reset_index()
)


geometry_base.columns = [
    (
        "_".join(
            [
                str(x)
                for x in col
                if str(x)
                not in {
                    "",
                    "None",
                }
            ]
        )
        if isinstance(
            col,
            tuple,
        )
        else col
    )
    for col in geometry_base.columns
]


geometry_base[
    "delta_mae_contiguous_minus_distributed"
] = (
    geometry_base[
        "mean_delta_mae_contiguous"
    ]
    - geometry_base[
        "mean_delta_mae_distributed"
    ]
)

geometry_base[
    "prediction_perturbation_contiguous_minus_distributed"
] = (
    geometry_base[
        "mean_prediction_perturbation_contiguous"
    ]
    - geometry_base[
        "mean_prediction_perturbation_distributed"
    ]
)

geometry_base[
    "relative_feature_error_contiguous_minus_distributed"
] = (
    geometry_base[
        "mean_feature_relative_error_contiguous"
    ]
    - geometry_base[
        "mean_feature_relative_error_distributed"
    ]
)


geometry_rows = []


for keys, group in (
    geometry_base.groupby(
        [
            "participant_id",
            "clinical_feature",
            "coverage",
            "missing_percent",
        ]
    )
):

    (
        participant_id,
        clinical_feature,
        coverage,
        missing_percent,
    ) = keys

    (
        delta_mean,
        delta_low,
        delta_high,
    ) = bootstrap_mean_ci(
        group[
            "delta_mae_contiguous_minus_distributed"
        ],
        seed=(
            BOOT_SEED
            + 10
        ),
    )

    (
        perturb_mean,
        perturb_low,
        perturb_high,
    ) = bootstrap_mean_ci(
        group[
            "prediction_perturbation_contiguous_minus_distributed"
        ],
        seed=(
            BOOT_SEED
            + 11
        ),
    )

    geometry_rows.append(
        {
            "participant_id":
                participant_id,

            "clinical_feature":
                clinical_feature,

            "coverage":
                coverage,

            "missing_percent":
                missing_percent,

            "n_paired_windows":
                len(group),

            "delta_mae_contiguous_minus_distributed":
                delta_mean,

            "delta_mae_geometry_ci_low":
                delta_low,

            "delta_mae_geometry_ci_high":
                delta_high,

            "prediction_perturbation_contiguous_minus_distributed":
                perturb_mean,

            "prediction_geometry_ci_low":
                perturb_low,

            "prediction_geometry_ci_high":
                perturb_high,
        }
    )


geometry_summary = pd.DataFrame(
    geometry_rows
).sort_values(
    [
        "participant_id",
        "missing_percent",
    ]
)


GEOMETRY_PATH = (
    ROOT
    / "combined_geometry_contrasts.csv"
)

geometry_summary.to_csv(
    GEOMETRY_PATH,
    index=False,
)


# ============================================================
# PARTICIPANT-LEVEL DOSE RESPONSE
#
# This is descriptive: slope of mean clinical degradation
# against missing fraction after averaging geometry.
# ============================================================

participant_rows = []


for participant_id, group in (
    window_level.groupby(
        "participant_id"
    )
):

    feature = (
        group[
            "clinical_feature"
        ].iloc[0]
    )

    dose = (
        group.groupby(
            "missing_fraction",
            as_index=False,
        )
        .agg(
            mean_delta_mae=(
                "mean_delta_mae",
                "mean",
            ),

            mean_prediction_perturbation=(
                "mean_prediction_perturbation",
                "mean",
            ),

            mean_feature_relative_error=(
                "mean_feature_relative_error",
                "mean",
            ),
        )
        .sort_values(
            "missing_fraction"
        )
    )

    x = (
        dose[
            "missing_fraction"
        ].to_numpy(
            dtype=float
        )
    )

    delta_y = (
        dose[
            "mean_delta_mae"
        ].to_numpy(
            dtype=float
        )
    )

    perturb_y = (
        dose[
            "mean_prediction_perturbation"
        ].to_numpy(
            dtype=float
        )
    )

    delta_slope = float(
        np.polyfit(
            x,
            delta_y,
            1,
        )[0]
    )

    perturb_slope = float(
        np.polyfit(
            x,
            perturb_y,
            1,
        )[0]
    )

    participant_rows.append(
        {
            "participant_id":
                participant_id,

            "clinical_feature":
                feature,

            "n_reference_windows":
                group[
                    "clinical_window_id"
                ].nunique(),

            "delta_mae_per_10pct_additional_missingness":
                (
                    delta_slope
                    * 0.10
                ),

            "prediction_perturbation_per_10pct_additional_missingness":
                (
                    perturb_slope
                    * 0.10
                ),

            "mean_delta_mae_at_10pct_missing":
                float(
                    dose.loc[
                        np.isclose(
                            dose[
                                "missing_fraction"
                            ],
                            0.10,
                        ),
                        "mean_delta_mae",
                    ].mean()
                ),

            "mean_delta_mae_at_75pct_missing":
                float(
                    dose.loc[
                        np.isclose(
                            dose[
                                "missing_fraction"
                            ],
                            0.75,
                        ),
                        "mean_delta_mae",
                    ].mean()
                ),
        }
    )


participant_summary = (
    pd.DataFrame(
        participant_rows
    )
)


PARTICIPANT_PATH = (
    ROOT
    / "combined_participant_clinical_sensitivity.csv"
)

participant_summary.to_csv(
    PARTICIPANT_PATH,
    index=False,
)


# ============================================================
# FIGURE 1:
# CLINICAL ERROR INCREASE
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        9,
        6,
    )
)

for (
    participant_id,
    pattern
), group in (
    summary.groupby(
        [
            "participant_id",
            "pattern",
        ]
    )
):

    group = (
        group.sort_values(
            "missing_percent"
        )
    )

    linestyle = (
        "-"
        if pattern
        == "distributed"
        else "--"
    )

    ax.plot(
        group[
            "missing_percent"
        ],
        group[
            "mean_delta_mae"
        ],
        marker="o",
        linestyle=linestyle,
        label=(
            f"{participant_id} | "
            f"{pattern}"
        ),
    )

    ax.fill_between(
        group[
            "missing_percent"
        ],
        group[
            "delta_mae_ci_low"
        ],
        group[
            "delta_mae_ci_high"
        ],
        alpha=0.10,
    )


ax.axhline(
    0,
    linewidth=1,
)

ax.set_xlabel(
    "GPS time masked (%)"
)

ax.set_ylabel(
    "Increase in depression prediction MAE "
    "(PHQ points)"
)

ax.set_xticks(
    [
        10,
        25,
        50,
        75,
    ]
)

ax.legend(
    frameon=False,
)

fig.tight_layout()

CLINICAL_FIG_PATH = (
    ROOT
    / "clinical_error_increase_vs_missingness.png"
)

fig.savefig(
    CLINICAL_FIG_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ============================================================
# FIGURE 2:
# PREDICTION PERTURBATION
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        9,
        6,
    )
)

for (
    participant_id,
    pattern
), group in (
    summary.groupby(
        [
            "participant_id",
            "pattern",
        ]
    )
):

    group = (
        group.sort_values(
            "missing_percent"
        )
    )

    linestyle = (
        "-"
        if pattern
        == "distributed"
        else "--"
    )

    ax.plot(
        group[
            "missing_percent"
        ],
        group[
            "mean_prediction_perturbation"
        ],
        marker="o",
        linestyle=linestyle,
        label=(
            f"{participant_id} | "
            f"{pattern}"
        ),
    )


ax.set_xlabel(
    "GPS time masked (%)"
)

ax.set_ylabel(
    "Mean change in predicted depression score "
    "(PHQ points)"
)

ax.set_xticks(
    [
        10,
        25,
        50,
        75,
    ]
)

ax.legend(
    frameon=False,
)

fig.tight_layout()

PERTURB_FIG_PATH = (
    ROOT
    / "prediction_perturbation_vs_missingness.png"
)

fig.savefig(
    PERTURB_FIG_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ============================================================
# FIGURE 3:
# NORMALIZED FEATURE DEGRADATION
#
# Relative rather than raw feature error is used because
# dominant-place fraction and displacement distance have
# fundamentally different physical scales.
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        9,
        6,
    )
)

for (
    participant_id,
    pattern
), group in (
    summary.groupby(
        [
            "participant_id",
            "pattern",
        ]
    )
):

    group = (
        group.sort_values(
            "missing_percent"
        )
    )

    linestyle = (
        "-"
        if pattern
        == "distributed"
        else "--"
    )

    ax.plot(
        group[
            "missing_percent"
        ],
        (
            100.0
            * group[
                "mean_feature_relative_error"
            ]
        ),
        marker="o",
        linestyle=linestyle,
        label=(
            f"{participant_id} | "
            f"{pattern}"
        ),
    )


ax.set_xlabel(
    "GPS time masked (%)"
)

ax.set_ylabel(
    "Mean personalized feature error "
    "(% of reference value)"
)

ax.set_xticks(
    [
        10,
        25,
        50,
        75,
    ]
)

ax.legend(
    frameon=False,
)

fig.tight_layout()

FEATURE_FIG_PATH = (
    ROOT
    / "relative_feature_error_vs_missingness.png"
)

fig.savefig(
    FEATURE_FIG_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ============================================================
# COMPACT TERMINAL RESULTS
# ============================================================

print()
print("=" * 100)
print(
    "PARTICIPANT CLINICAL SENSITIVITY"
)
print("=" * 100)

print(
    participant_summary.to_string(
        index=False
    )
)


print()
print("=" * 100)
print(
    "MEAN DELTA MAE BY COVERAGE / GEOMETRY"
)
print("=" * 100)

display = (
    summary[
        [
            "participant_id",
            "clinical_feature",
            "missing_percent",
            "pattern",
            "n_clinical_windows",
            "mean_delta_mae",
            "delta_mae_ci_low",
            "delta_mae_ci_high",
            "fraction_windows_worsened",
            "mean_prediction_perturbation",
            "mean_feature_relative_error",
        ]
    ]
    .copy()
)

display[
    "mean_feature_relative_error"
] = (
    100.0
    * display[
        "mean_feature_relative_error"
    ]
)

display = display.rename(
    columns={
        "mean_feature_relative_error":
            "mean_feature_error_pct"
    }
)

print(
    display.to_string(
        index=False,
    )
)


print()
print("=" * 100)
print(
    "CONTIGUOUS - DISTRIBUTED GEOMETRY EFFECT"
)
print("=" * 100)

print(
    geometry_summary[
        [
            "participant_id",
            "missing_percent",
            "delta_mae_contiguous_minus_distributed",
            "delta_mae_geometry_ci_low",
            "delta_mae_geometry_ci_high",
            "prediction_perturbation_contiguous_minus_distributed",
        ]
    ].to_string(
        index=False
    )
)


print()
print("Saved:")
print(
    " ",
    WINDOW_PATH,
)
print(
    " ",
    SUMMARY_PATH,
)
print(
    " ",
    GEOMETRY_PATH,
)
print(
    " ",
    PARTICIPANT_PATH,
)

print()
print("Figures:")
print(
    " ",
    CLINICAL_FIG_PATH,
)
print(
    " ",
    PERTURB_FIG_PATH,
)
print(
    " ",
    FEATURE_FIG_PATH,
)
