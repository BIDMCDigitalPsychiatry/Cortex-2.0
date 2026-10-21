"""Analyze RAPHE GPS reliability across a continuum of missingness geometry.

Combines:
- original contiguous missingness
- balanced k=2,4,8,16 missing-run experiments
- original random-distributed missingness

All GPS feature algorithms and parameters remain frozen.

Primary natural-variability normalization:
    median absolute difference across all pairs of well-observed
    reference days within participant.

This workflow does NOT define a good/bad reliability threshold.

Questions
---------
1. At the same coverage, does increasingly concentrated missingness produce
   larger participant-normalized feature error?

2. Does missingness geometry improve out-of-participant prediction of
   normalized feature error beyond coverage alone?

Prediction is evaluated with leave-one-participant-out cross-validation on
participant-balanced summaries.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "/shared/Digital Clinic/"
    "raphe_outputs/digital_clinic/"
    "feature_reliability/"
    "gps_observability"
)

EXTREMES_PATH = (
    ROOT
    / "gps_validation_degradation_with_missingness_structure.parquet"
)

INTERMEDIATE_DIR = (
    ROOT
    / "intermediate_missingness_runs"
)

VARIABILITY_PATH = (
    ROOT
    / "gps_validation_reference_feature_variability_by_participant.csv"
)


COMBINED_OUT = (
    ROOT
    / "gps_missingness_continuum_normalized.parquet"
)

PARTICIPANT_DESIGN_OUT = (
    ROOT
    / "gps_missingness_continuum_participant_design.csv"
)

CURVE_OUT = (
    ROOT
    / "gps_missingness_continuum_reliability_curves.csv"
)

TREND_OUT = (
    ROOT
    / "gps_missingness_continuum_within_participant_trends.csv"
)

TREND_SUMMARY_OUT = (
    ROOT
    / "gps_missingness_continuum_trend_summary.csv"
)

CV_OUT = (
    ROOT
    / "gps_missingness_continuum_lopo_predictions.csv"
)

CV_SUMMARY_OUT = (
    ROOT
    / "gps_missingness_continuum_lopo_summary.csv"
)


# ============================================================
# DESIGN
# ============================================================

EXPECTED_INTERMEDIATE_FILES = 80
EXPECTED_INTERMEDIATE_ROWS = 8960
EXPECTED_EXTREME_ROWS = 4480
EXPECTED_COMBINED_ROWS = 13440

EXPECTED_PARTICIPANTS = 16


DESIGN_ORDER = {
    "k1_contiguous": 1,
    "k2": 2,
    "k4": 4,
    "k8": 8,
    "k16": 16,
    "random_distributed": 999,
}


FEATURES = {
    "entropy": {
        "absolute_error":
            "abs_entropy_error",

        "normalized_error":
            "entropy_normalized_error",

        "denominator":
            "entropy_pairwise_median_abs_diff",
    },

    "dominant_fraction": {
        "absolute_error":
            "abs_dominant_fraction_error",

        "normalized_error":
            "dominant_fraction_normalized_error",

        "denominator":
            "dominant_fraction_pairwise_median_abs_diff",
    },

    "effective_places": {
        "absolute_error":
            "abs_effective_places_error",

        "normalized_error":
            "effective_places_normalized_error",

        "denominator":
            "effective_places_pairwise_median_abs_diff",
    },
}


# ============================================================
# HELPERS
# ============================================================

def safe_spearman(
    x,
    y,
):
    x = pd.to_numeric(
        pd.Series(x),
        errors="coerce",
    )

    y = pd.to_numeric(
        pd.Series(y),
        errors="coerce",
    )

    valid = (
        x.notna()
        & y.notna()
        & np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[
        valid
    ]

    y = y[
        valid
    ]

    if len(x) < 3:
        return np.nan

    if (
        x.nunique() < 2
        or y.nunique() < 2
    ):
        return np.nan

    return float(
        x.rank(
            method="average"
        ).corr(
            y.rank(
                method="average"
            )
        )
    )


def design_matrix_coverage_only(
    missing_fraction,
):
    m = np.asarray(
        missing_fraction,
        dtype=float,
    )

    return np.column_stack(
        [
            np.ones(
                len(m)
            ),
            m,
            m ** 2,
        ]
    )


def design_matrix_coverage_geometry(
    missing_fraction,
    concentration,
):
    m = np.asarray(
        missing_fraction,
        dtype=float,
    )

    c = np.asarray(
        concentration,
        dtype=float,
    )

    return np.column_stack(
        [
            np.ones(
                len(m)
            ),
            m,
            m ** 2,
            c,
            c ** 2,
            m * c,
        ]
    )


def fit_lstsq(
    x,
    y,
):
    coefficients, _, _, _ = (
        np.linalg.lstsq(
            x,
            y,
            rcond=None,
        )
    )

    return coefficients


def summarize_prediction_errors(
    y,
    yhat,
):
    y = np.asarray(
        y,
        dtype=float,
    )

    yhat = np.asarray(
        yhat,
        dtype=float,
    )

    residual = (
        y
        - yhat
    )

    mae = float(
        np.mean(
            np.abs(
                residual
            )
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                residual ** 2
            )
        )
    )

    denominator = float(
        np.sum(
            (
                y
                - np.mean(
                    y
                )
            )
            ** 2
        )
    )

    if denominator > 0:
        r2 = float(
            1.0
            - (
                np.sum(
                    residual ** 2
                )
                / denominator
            )
        )
    else:
        r2 = np.nan

    return {
        "mae":
            mae,

        "rmse":
            rmse,

        "r2":
            r2,
    }


# ============================================================
# LOAD ORIGINAL EXTREMES
# ============================================================

extremes = pd.read_parquet(
    EXTREMES_PATH
)


if len(
    extremes
) != EXPECTED_EXTREME_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_EXTREME_ROWS} extreme rows, "
        f"found {len(extremes)}."
    )


extremes[
    "date"
] = extremes[
    "date"
].astype(str)


extremes[
    "missingness_design"
] = np.where(
    extremes[
        "missingness_pattern"
    ]
    == "contiguous",
    "k1_contiguous",
    "random_distributed",
)


# ============================================================
# LOAD INTERMEDIATE CONDITIONS
# ============================================================

intermediate_files = sorted(
    INTERMEDIATE_DIR.glob(
        "coverage_*_k*_rep_*.parquet"
    )
)


if len(
    intermediate_files
) != EXPECTED_INTERMEDIATE_FILES:
    raise RuntimeError(
        f"Expected {EXPECTED_INTERMEDIATE_FILES} intermediate files, "
        f"found {len(intermediate_files)}."
    )


intermediate = pd.concat(
    [
        pd.read_parquet(
            path
        )
        for path in intermediate_files
    ],
    ignore_index=True,
)


if len(
    intermediate
) != EXPECTED_INTERMEDIATE_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_INTERMEDIATE_ROWS} intermediate rows, "
        f"found {len(intermediate)}."
    )


intermediate[
    "date"
] = intermediate[
    "date"
].astype(str)


# ============================================================
# NATURAL VARIABILITY DENOMINATORS
# ============================================================

variability = pd.read_csv(
    VARIABILITY_PATH
)


if (
    variability[
        "participant_id"
    ].nunique()
    != EXPECTED_PARTICIPANTS
):
    raise RuntimeError(
        "Expected 16 participant-specific variability rows."
    )


denominator_columns = [
    FEATURES[
        feature
    ][
        "denominator"
    ]
    for feature in FEATURES
]


variability_keep = variability[
    [
        "participant_id",
    ]
    + denominator_columns
].copy()


# ============================================================
# NORMALIZE INTERMEDIATE ERRORS
# ============================================================

intermediate = intermediate.merge(
    variability_keep,
    on="participant_id",
    how="left",
    validate="many_to_one",
)


for feature, spec in FEATURES.items():

    denominator = pd.to_numeric(
        intermediate[
            spec[
                "denominator"
            ]
        ],
        errors="coerce",
    )

    absolute_error = pd.to_numeric(
        intermediate[
            spec[
                "absolute_error"
            ]
        ],
        errors="coerce",
    )


    if (
        denominator.isna().any()
        or (
            denominator
            <= 0
        ).any()
    ):
        raise RuntimeError(
            f"Invalid normalization denominator for {feature}."
        )


    intermediate[
        spec[
            "normalized_error"
        ]
    ] = (
        absolute_error
        / denominator
    )


# ============================================================
# VERIFY ORIGINAL NORMALIZATION AGAINST SAME DENOMINATORS
# ============================================================

extreme_check = extremes.merge(
    variability_keep,
    on="participant_id",
    how="left",
    validate="many_to_one",
    suffixes=(
        "",
        "_variability",
    ),
)


for feature, spec in FEATURES.items():

    recomputed = (
        pd.to_numeric(
            extreme_check[
                spec[
                    "absolute_error"
                ]
            ],
            errors="coerce",
        )
        / pd.to_numeric(
            extreme_check[
                spec[
                    "denominator"
                ]
            ],
            errors="coerce",
        )
    )


    existing = pd.to_numeric(
        extreme_check[
            spec[
                "normalized_error"
            ]
        ],
        errors="coerce",
    )


    difference = (
        recomputed
        - existing
    ).abs()


    max_difference = float(
        difference.max()
    )


    if max_difference > 1e-10:
        raise RuntimeError(
            f"Original normalization mismatch for {feature}: "
            f"max difference={max_difference}"
        )


# ============================================================
# HARMONIZE COMMON STRUCTURAL VARIABLES
# ============================================================

if (
    "mask_planned_coverage"
    in extremes.columns
):
    extremes[
        "planned_coverage_canonical"
    ] = extremes[
        "mask_planned_coverage"
    ]
else:
    extremes[
        "planned_coverage_canonical"
    ] = extremes[
        "planned_coverage"
    ]


intermediate[
    "planned_coverage_canonical"
] = intermediate[
    "planned_coverage"
]


for frame in [
    extremes,
    intermediate,
]:

    frame[
        "missing_fraction"
    ] = (
        1.0
        - pd.to_numeric(
            frame[
                "planned_coverage_canonical"
            ],
            errors="coerce",
        )
    )


    frame[
        "longest_gap_hours"
    ] = (
        pd.to_numeric(
            frame[
                "longest_missing_ms"
            ],
            errors="coerce",
        )
        / 3_600_000
    )


# ============================================================
# SELECT COMMON COLUMNS
# ============================================================

common_columns = [
    "participant_id",
    "date",
    "sampling_density_quartile",
    "target_coverage",
    "replicate",
    "missingness_design",
    "planned_coverage_canonical",
    "missing_fraction",
    "supported_fraction_of_day",
    "raw_fraction_retained",
    "missing_run_count",
    "longest_missing_bins",
    "longest_missing_ms",
    "longest_gap_hours",
    "longest_missing_fraction_of_missing",
    "missing_run_concentration",
    "missing_fragmentation",
    "abs_entropy_error",
    "abs_dominant_fraction_error",
    "abs_effective_places_error",
    "entropy_normalized_error",
    "dominant_fraction_normalized_error",
    "effective_places_normalized_error",
]


missing_extreme_columns = (
    set(
        common_columns
    )
    - set(
        extremes.columns
    )
)


missing_intermediate_columns = (
    set(
        common_columns
    )
    - set(
        intermediate.columns
    )
)


if missing_extreme_columns:
    raise RuntimeError(
        "Missing extreme columns: "
        f"{sorted(missing_extreme_columns)}"
    )


if missing_intermediate_columns:
    raise RuntimeError(
        "Missing intermediate columns: "
        f"{sorted(missing_intermediate_columns)}"
    )


combined = pd.concat(
    [
        extremes[
            common_columns
        ],
        intermediate[
            common_columns
        ],
    ],
    ignore_index=True,
)


if len(
    combined
) != EXPECTED_COMBINED_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_COMBINED_ROWS} combined rows, "
        f"found {len(combined)}."
    )


if (
    combined[
        "participant_id"
    ].nunique()
    != EXPECTED_PARTICIPANTS
):
    raise RuntimeError(
        "Unexpected participant count after combination."
    )


combined[
    "design_order"
] = combined[
    "missingness_design"
].map(
    DESIGN_ORDER
)


if combined[
    "design_order"
].isna().any():
    raise RuntimeError(
        "Unknown missingness design."
    )


combined.to_parquet(
    COMBINED_OUT,
    index=False,
)


# ============================================================
# PARTICIPANT-BALANCED DESIGN SUMMARY
#
# First summarize across:
#   7 days × 5 replicates
#
# for each participant × coverage × design.
# ============================================================

participant_design_rows = []


for (
    participant_id,
    quartile,
    coverage,
    design,
), x in combined.groupby(
    [
        "participant_id",
        "sampling_density_quartile",
        "target_coverage",
        "missingness_design",
    ],
    sort=True,
):

    row = {
        "participant_id":
            participant_id,

        "sampling_density_quartile":
            quartile,

        "target_coverage":
            float(
                coverage
            ),

        "missingness_design":
            design,

        "design_order":
            DESIGN_ORDER[
                design
            ],

        "observations":
            len(
                x
            ),

        "median_planned_coverage":
            x[
                "planned_coverage_canonical"
            ].median(),

        "median_missing_fraction":
            x[
                "missing_fraction"
            ].median(),

        "median_supported_fraction_of_day":
            x[
                "supported_fraction_of_day"
            ].median(),

        "median_missing_run_count":
            x[
                "missing_run_count"
            ].median(),

        "median_longest_gap_hours":
            x[
                "longest_gap_hours"
            ].median(),

        "median_longest_missing_share":
            x[
                "longest_missing_fraction_of_missing"
            ].median(),

        "median_missing_run_concentration":
            x[
                "missing_run_concentration"
            ].median(),

        "median_missing_fragmentation":
            x[
                "missing_fragmentation"
            ].median(),
    }


    for feature, spec in FEATURES.items():

        row[
            f"median_{feature}_normalized_error"
        ] = (
            pd.to_numeric(
                x[
                    spec[
                        "normalized_error"
                    ]
                ],
                errors="coerce",
            )
            .median()
        )


    participant_design_rows.append(
        row
    )


participant_design = pd.DataFrame(
    participant_design_rows
)


expected_participant_design_rows = (
    16
    * 4
    * 6
)


if len(
    participant_design
) != expected_participant_design_rows:
    raise RuntimeError(
        f"Expected {expected_participant_design_rows} participant-design rows, "
        f"found {len(participant_design)}."
    )


participant_design.to_csv(
    PARTICIPANT_DESIGN_OUT,
    index=False,
)


# ============================================================
# PARTICIPANT-BALANCED RELIABILITY CURVES
# ============================================================

curve_rows = []


for (
    coverage,
    design,
), x in participant_design.groupby(
    [
        "target_coverage",
        "missingness_design",
    ],
    sort=True,
):

    row = {
        "target_coverage":
            float(
                coverage
            ),

        "missingness_design":
            design,

        "design_order":
            DESIGN_ORDER[
                design
            ],

        "participants":
            x[
                "participant_id"
            ].nunique(),

        "median_missing_run_count":
            x[
                "median_missing_run_count"
            ].median(),

        "median_longest_gap_hours":
            x[
                "median_longest_gap_hours"
            ].median(),

        "median_longest_missing_share":
            x[
                "median_longest_missing_share"
            ].median(),

        "median_missing_run_concentration":
            x[
                "median_missing_run_concentration"
            ].median(),

        "median_missing_fragmentation":
            x[
                "median_missing_fragmentation"
            ].median(),
    }


    for feature in FEATURES:

        values = pd.to_numeric(
            x[
                f"median_{feature}_normalized_error"
            ],
            errors="coerce",
        )


        row.update({
            f"q25_{feature}_normalized_error":
                values.quantile(
                    0.25
                ),

            f"median_{feature}_normalized_error":
                values.median(),

            f"q75_{feature}_normalized_error":
                values.quantile(
                    0.75
                ),

            f"p90_{feature}_normalized_error":
                values.quantile(
                    0.90
                ),
        })


    curve_rows.append(
        row
    )


curves = pd.DataFrame(
    curve_rows
)


curves = curves.sort_values(
    [
        "target_coverage",
        "design_order",
    ],
    ascending=[
        False,
        True,
    ],
)


curves.to_csv(
    CURVE_OUT,
    index=False,
)


# ============================================================
# WITHIN-PARTICIPANT STRUCTURAL TREND
#
# At each fixed coverage, each participant has six structural
# levels. Compute Spearman rho between concentration and error.
# ============================================================

trend_rows = []


for feature in FEATURES:

    error_column = (
        f"median_{feature}_normalized_error"
    )


    for (
        coverage,
        participant_id,
    ), x in participant_design.groupby(
        [
            "target_coverage",
            "participant_id",
        ],
        sort=True,
    ):

        if len(
            x
        ) != 6:
            raise RuntimeError(
                f"{participant_id}, coverage={coverage}: "
                f"expected 6 designs, found {len(x)}."
            )


        trend_rows.append({
            "feature":
                feature,

            "target_coverage":
                float(
                    coverage
                ),

            "participant_id":
                participant_id,

            "rho_concentration_error":
                safe_spearman(
                    x[
                        "median_missing_run_concentration"
                    ],
                    x[
                        error_column
                    ],
                ),

            "rho_longest_share_error":
                safe_spearman(
                    x[
                        "median_longest_missing_share"
                    ],
                    x[
                        error_column
                    ],
                ),

            "rho_longest_gap_error":
                safe_spearman(
                    x[
                        "median_longest_gap_hours"
                    ],
                    x[
                        error_column
                    ],
                ),

            "rho_fragmentation_error":
                safe_spearman(
                    x[
                        "median_missing_fragmentation"
                    ],
                    x[
                        error_column
                    ],
                ),
        })


trends = pd.DataFrame(
    trend_rows
)


trends.to_csv(
    TREND_OUT,
    index=False,
)


trend_summary_rows = []


for (
    feature,
    coverage,
), x in trends.groupby(
    [
        "feature",
        "target_coverage",
    ],
    sort=True,
):

    row = {
        "feature":
            feature,

        "target_coverage":
            float(
                coverage
            ),

        "participants":
            len(
                x
            ),
    }


    for correlation_column in [
        "rho_concentration_error",
        "rho_longest_share_error",
        "rho_longest_gap_error",
        "rho_fragmentation_error",
    ]:

        values = pd.to_numeric(
            x[
                correlation_column
            ],
            errors="coerce",
        ).dropna()


        row[
            f"median_{correlation_column}"
        ] = (
            values.median()
        )


        row[
            f"positive_fraction_{correlation_column}"
        ] = (
            (
                values
                > 0
            )
            .mean()
        )


    trend_summary_rows.append(
        row
    )


trend_summary = pd.DataFrame(
    trend_summary_rows
)


trend_summary.to_csv(
    TREND_SUMMARY_OUT,
    index=False,
)


# ============================================================
# LEAVE-ONE-PARTICIPANT-OUT PREDICTION
#
# Compare:
#
# Model A:
#   error ~ missing_fraction + missing_fraction^2
#
# Model B:
#   error ~ missing_fraction + missing_fraction^2
#           + concentration + concentration^2
#           + missing_fraction * concentration
#
# The quadratic coverage term prevents the geometry model from
# receiving credit merely because degradation is nonlinear in
# amount of missingness.
#
# Analysis unit = participant × coverage × structural design.
# ============================================================

prediction_rows = []


participants = sorted(
    participant_design[
        "participant_id"
    ].unique()
)


for feature in FEATURES:

    outcome_column = (
        f"median_{feature}_normalized_error"
    )


    for held_out_participant in participants:

        train = participant_design[
            participant_design[
                "participant_id"
            ]
            != held_out_participant
        ].copy()


        test = participant_design[
            participant_design[
                "participant_id"
            ]
            == held_out_participant
        ].copy()


        y_train = pd.to_numeric(
            train[
                outcome_column
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )


        y_test = pd.to_numeric(
            test[
                outcome_column
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )


        train_missing = pd.to_numeric(
            train[
                "median_missing_fraction"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )


        test_missing = pd.to_numeric(
            test[
                "median_missing_fraction"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )


        train_concentration = pd.to_numeric(
            train[
                "median_missing_run_concentration"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )


        test_concentration = pd.to_numeric(
            test[
                "median_missing_run_concentration"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )


        # ----------------------------------------------------
        # Coverage-only
        # ----------------------------------------------------

        x_train_coverage = (
            design_matrix_coverage_only(
                train_missing
            )
        )


        x_test_coverage = (
            design_matrix_coverage_only(
                test_missing
            )
        )


        beta_coverage = fit_lstsq(
            x_train_coverage,
            y_train,
        )


        prediction_coverage = (
            x_test_coverage
            @ beta_coverage
        )


        # ----------------------------------------------------
        # Coverage + geometry
        # ----------------------------------------------------

        x_train_geometry = (
            design_matrix_coverage_geometry(
                train_missing,
                train_concentration,
            )
        )


        x_test_geometry = (
            design_matrix_coverage_geometry(
                test_missing,
                test_concentration,
            )
        )


        beta_geometry = fit_lstsq(
            x_train_geometry,
            y_train,
        )


        prediction_geometry = (
            x_test_geometry
            @ beta_geometry
        )


        for position, (
            index,
            row,
        ) in enumerate(
            test.iterrows()
        ):

            prediction_rows.append({
                "feature":
                    feature,

                "held_out_participant":
                    held_out_participant,

                "target_coverage":
                    row[
                        "target_coverage"
                    ],

                "missingness_design":
                    row[
                        "missingness_design"
                    ],

                "missing_fraction":
                    row[
                        "median_missing_fraction"
                    ],

                "missing_run_concentration":
                    row[
                        "median_missing_run_concentration"
                    ],

                "observed_normalized_error":
                    y_test[
                        position
                    ],

                "predicted_coverage_only":
                    prediction_coverage[
                        position
                    ],

                "predicted_coverage_geometry":
                    prediction_geometry[
                        position
                    ],
            })


predictions = pd.DataFrame(
    prediction_rows
)


predictions.to_csv(
    CV_OUT,
    index=False,
)


# ============================================================
# CV SUMMARY
# ============================================================

cv_summary_rows = []


for feature, x in predictions.groupby(
    "feature",
    sort=True,
):

    y = x[
        "observed_normalized_error"
    ].to_numpy(
        dtype=float
    )


    coverage_metrics = (
        summarize_prediction_errors(
            y,
            x[
                "predicted_coverage_only"
            ].to_numpy(
                dtype=float
            ),
        )
    )


    geometry_metrics = (
        summarize_prediction_errors(
            y,
            x[
                "predicted_coverage_geometry"
            ].to_numpy(
                dtype=float
            ),
        )
    )


    cv_summary_rows.append({
        "feature":
            feature,

        "n_predictions":
            len(
                x
            ),

        "coverage_only_mae":
            coverage_metrics[
                "mae"
            ],

        "coverage_geometry_mae":
            geometry_metrics[
                "mae"
            ],

        "mae_reduction_fraction":
            (
                (
                    coverage_metrics[
                        "mae"
                    ]
                    - geometry_metrics[
                        "mae"
                    ]
                )
                / coverage_metrics[
                    "mae"
                ]
            ),

        "coverage_only_rmse":
            coverage_metrics[
                "rmse"
            ],

        "coverage_geometry_rmse":
            geometry_metrics[
                "rmse"
            ],

        "rmse_reduction_fraction":
            (
                (
                    coverage_metrics[
                        "rmse"
                    ]
                    - geometry_metrics[
                        "rmse"
                    ]
                )
                / coverage_metrics[
                    "rmse"
                ]
            ),

        "coverage_only_r2":
            coverage_metrics[
                "r2"
            ],

        "coverage_geometry_r2":
            geometry_metrics[
                "r2"
            ],

        "r2_gain":
            (
                geometry_metrics[
                    "r2"
                ]
                - coverage_metrics[
                    "r2"
                ]
            ),
    })


cv_summary = pd.DataFrame(
    cv_summary_rows
)


cv_summary.to_csv(
    CV_SUMMARY_OUT,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

print("=" * 80)
print("RAPHE GPS CONTINUOUS MISSINGNESS RELIABILITY")
print("=" * 80)

print(
    "Combined rows:",
    f"{len(combined):,}",
)

print(
    "Participants:",
    combined[
        "participant_id"
    ].nunique(),
)

print(
    "Structural designs:",
    ", ".join(
        curves[
            "missingness_design"
        ]
        .drop_duplicates()
        .tolist()
    ),
)

print(
    "Participant-design rows:",
    len(
        participant_design
    ),
)


print()
print("=" * 80)
print("PARTICIPANT-BALANCED RELIABILITY CONTINUUM")
print("=" * 80)

print(
    curves[
        [
            "target_coverage",
            "missingness_design",
            "median_longest_gap_hours",
            "median_missing_run_concentration",
            "median_entropy_normalized_error",
            "median_dominant_fraction_normalized_error",
            "median_effective_places_normalized_error",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print()
print("=" * 80)
print("WITHIN-PARTICIPANT STRUCTURAL TREND")
print("=" * 80)

print(
    trend_summary[
        [
            "feature",
            "target_coverage",
            "median_rho_concentration_error",
            "positive_fraction_rho_concentration_error",
            "median_rho_longest_gap_error",
            "positive_fraction_rho_longest_gap_error",
            "median_rho_fragmentation_error",
            "positive_fraction_rho_fragmentation_error",
        ]
    ]
    .sort_values(
        [
            "feature",
            "target_coverage",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print()
print("=" * 80)
print("LEAVE-ONE-PARTICIPANT-OUT PREDICTION")
print("=" * 80)

print(
    cv_summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print()
print("=" * 80)
print("OUTPUTS")
print("=" * 80)

print(COMBINED_OUT)
print(PARTICIPANT_DESIGN_OUT)
print(CURVE_OUT)
print(TREND_OUT)
print(TREND_SUMMARY_OUT)
print(CV_OUT)
print(CV_SUMMARY_OUT)

print()
print("=" * 80)
print("DONE")
print("=" * 80)
