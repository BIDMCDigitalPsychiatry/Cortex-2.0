"""Participant-cluster bootstrap for Cortex 2.0 GPS reliability results.

Bootstraps participants, not individual days/conditions.

Provides uncertainty for:
1. participant-balanced reliability continuum curves;
2. within-participant structural correlations;
3. leave-one-participant-out prediction improvement from adding
   missingness geometry beyond coverage.

The existing LOPO predictions remain fixed; participants are resampled as
paired test clusters when estimating uncertainty in predictive performance.

No reliability threshold or hypothesis-test p-value is introduced.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "/shared/Digital Clinic/"
    "cortex2_outputs/digital_clinic/"
    "feature_reliability/"
    "gps_observability"
)

PARTICIPANT_DESIGN_PATH = (
    ROOT
    / "gps_missingness_continuum_participant_design.csv"
)

TREND_PATH = (
    ROOT
    / "gps_missingness_continuum_within_participant_trends.csv"
)

LOPO_PATH = (
    ROOT
    / "gps_missingness_continuum_lopo_predictions.csv"
)


CURVE_OUT = (
    ROOT
    / "gps_missingness_continuum_bootstrap_curve_ci.csv"
)

TREND_OUT = (
    ROOT
    / "gps_missingness_continuum_bootstrap_trend_ci.csv"
)

LOPO_OUT = (
    ROOT
    / "gps_missingness_continuum_bootstrap_lopo_ci.csv"
)


N_BOOT = 5000
SEED = 20261021

FEATURES = [
    "entropy",
    "dominant_fraction",
    "effective_places",
]

COVERAGES = [
    0.90,
    0.75,
    0.50,
    0.25,
]

DESIGNS = [
    "k1_contiguous",
    "k2",
    "k4",
    "k8",
    "k16",
    "random_distributed",
]


def percentile_ci(
    values,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(values) == 0:
        return (
            np.nan,
            np.nan,
        )

    return (
        float(
            np.quantile(
                values,
                0.025,
            )
        ),
        float(
            np.quantile(
                values,
                0.975,
            )
        ),
    )


def prediction_metrics(
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

    return (
        mae,
        rmse,
        r2,
    )


# ============================================================
# LOAD
# ============================================================

participant_design = pd.read_csv(
    PARTICIPANT_DESIGN_PATH
)

trends = pd.read_csv(
    TREND_PATH
)

predictions = pd.read_csv(
    LOPO_PATH
)


participants = sorted(
    participant_design[
        "participant_id"
    ]
    .unique()
    .tolist()
)


if len(
    participants
) != 16:
    raise RuntimeError(
        f"Expected 16 participants, "
        f"found {len(participants)}."
    )


participant_to_index = {
    participant_id:
        index
    for index, participant_id
    in enumerate(
        participants
    )
}


rng = np.random.default_rng(
    SEED
)


bootstrap_indices = rng.integers(
    0,
    len(
        participants
    ),
    size=(
        N_BOOT,
        len(
            participants
        ),
    ),
)


print("=" * 80)
print("Cortex 2.0 GPS PARTICIPANT-CLUSTER BOOTSTRAP")
print("=" * 80)

print(
    "Participants:",
    len(
        participants
    ),
)

print(
    "Bootstrap resamples:",
    N_BOOT,
)

print(
    "Seed:",
    SEED,
)


# ============================================================
# 1. RELIABILITY CONTINUUM CURVES
# ============================================================

curve_rows = []


for feature in FEATURES:

    outcome = (
        f"median_{feature}_normalized_error"
    )


    array = np.full(
        (
            len(
                participants
            ),
            len(
                COVERAGES
            ),
            len(
                DESIGNS
            ),
        ),
        np.nan,
        dtype=float,
    )


    for participant_position, participant_id in enumerate(
        participants
    ):

        participant_rows = participant_design[
            participant_design[
                "participant_id"
            ]
            == participant_id
        ]


        for coverage_position, coverage in enumerate(
            COVERAGES
        ):

            for design_position, design in enumerate(
                DESIGNS
            ):

                selected = participant_rows[
                    np.isclose(
                        participant_rows[
                            "target_coverage"
                        ],
                        coverage,
                    )
                    & (
                        participant_rows[
                            "missingness_design"
                        ]
                        == design
                    )
                ]


                if len(
                    selected
                ) != 1:
                    raise RuntimeError(
                        f"{participant_id} {coverage} {design}: "
                        f"expected one row, found {len(selected)}."
                    )


                array[
                    participant_position,
                    coverage_position,
                    design_position,
                ] = float(
                    selected[
                        outcome
                    ].iloc[0]
                )


    if not np.isfinite(
        array
    ).all():
        raise RuntimeError(
            f"Non-finite continuum value for {feature}."
        )


    point = np.median(
        array,
        axis=0,
    )


    bootstrap_values = np.empty(
        (
            N_BOOT,
            len(
                COVERAGES
            ),
            len(
                DESIGNS
            ),
        ),
        dtype=float,
    )


    for bootstrap_index in range(
        N_BOOT
    ):

        bootstrap_values[
            bootstrap_index
        ] = np.median(
            array[
                bootstrap_indices[
                    bootstrap_index
                ]
            ],
            axis=0,
        )


    for coverage_position, coverage in enumerate(
        COVERAGES
    ):

        for design_position, design in enumerate(
            DESIGNS
        ):

            values = bootstrap_values[
                :,
                coverage_position,
                design_position,
            ]


            lower, upper = percentile_ci(
                values
            )


            curve_rows.append({
                "feature":
                    feature,

                "target_coverage":
                    coverage,

                "missingness_design":
                    design,

                "point_estimate":
                    point[
                        coverage_position,
                        design_position,
                    ],

                "bootstrap_median":
                    float(
                        np.median(
                            values
                        )
                    ),

                "ci95_lower":
                    lower,

                "ci95_upper":
                    upper,
            })


curve_ci = pd.DataFrame(
    curve_rows
)


curve_ci.to_csv(
    CURVE_OUT,
    index=False,
)


# ============================================================
# 2. WITHIN-PARTICIPANT STRUCTURAL TREND
# ============================================================

TREND_METRICS = {
    "concentration":
        "rho_concentration_error",

    "longest_gap":
        "rho_longest_gap_error",

    "fragmentation":
        "rho_fragmentation_error",
}


trend_rows = []


for feature in FEATURES:

    for coverage in COVERAGES:

        selected = trends[
            (
                trends[
                    "feature"
                ]
                == feature
            )
            & np.isclose(
                trends[
                    "target_coverage"
                ],
                coverage,
            )
        ].copy()


        selected = selected.set_index(
            "participant_id"
        ).reindex(
            participants
        )


        if len(
            selected
        ) != 16:
            raise RuntimeError(
                "Unexpected trend participant count."
            )


        for metric_name, column in TREND_METRICS.items():

            values = pd.to_numeric(
                selected[
                    column
                ],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )


            valid = np.isfinite(
                values
            )


            point_median = float(
                np.nanmedian(
                    values
                )
            )


            point_positive_fraction = float(
                np.mean(
                    values[
                        valid
                    ]
                    > 0
                )
            )


            bootstrap_medians = np.empty(
                N_BOOT,
                dtype=float,
            )

            bootstrap_positive_fractions = np.empty(
                N_BOOT,
                dtype=float,
            )


            for bootstrap_index in range(
                N_BOOT
            ):

                sampled = values[
                    bootstrap_indices[
                        bootstrap_index
                    ]
                ]


                sampled_valid = sampled[
                    np.isfinite(
                        sampled
                    )
                ]


                bootstrap_medians[
                    bootstrap_index
                ] = np.median(
                    sampled_valid
                )


                bootstrap_positive_fractions[
                    bootstrap_index
                ] = np.mean(
                    sampled_valid
                    > 0
                )


            median_lower, median_upper = percentile_ci(
                bootstrap_medians
            )

            positive_lower, positive_upper = percentile_ci(
                bootstrap_positive_fractions
            )


            if metric_name == "fragmentation":

                expected_direction_probability = float(
                    np.mean(
                        bootstrap_medians
                        < 0
                    )
                )

            else:

                expected_direction_probability = float(
                    np.mean(
                        bootstrap_medians
                        > 0
                    )
                )


            trend_rows.append({
                "feature":
                    feature,

                "target_coverage":
                    coverage,

                "descriptor":
                    metric_name,

                "point_median_rho":
                    point_median,

                "median_rho_ci95_lower":
                    median_lower,

                "median_rho_ci95_upper":
                    median_upper,

                "point_positive_participant_fraction":
                    point_positive_fraction,

                "positive_fraction_ci95_lower":
                    positive_lower,

                "positive_fraction_ci95_upper":
                    positive_upper,

                "bootstrap_probability_expected_direction":
                    expected_direction_probability,
            })


trend_ci = pd.DataFrame(
    trend_rows
)


trend_ci.to_csv(
    TREND_OUT,
    index=False,
)


# ============================================================
# 3. LOPO PREDICTION UNCERTAINTY
#
# Resample held-out participants as paired test clusters.
# ============================================================

lopo_rows = []


for feature in FEATURES:

    selected = predictions[
        predictions[
            "feature"
        ]
        == feature
    ].copy()


    blocks = []


    for participant_id in participants:

        participant_block = (
            selected[
                selected[
                    "held_out_participant"
                ]
                == participant_id
            ]
            .sort_values(
                [
                    "target_coverage",
                    "missingness_design",
                ]
            )
        )


        if len(
            participant_block
        ) != 24:
            raise RuntimeError(
                f"{feature} {participant_id}: "
                f"expected 24 LOPO predictions, "
                f"found {len(participant_block)}."
            )


        blocks.append(
            participant_block[
                [
                    "observed_normalized_error",
                    "predicted_coverage_only",
                    "predicted_coverage_geometry",
                ]
            ]
            .to_numpy(
                dtype=float
            )
        )


    blocks = np.stack(
        blocks,
        axis=0,
    )


    y = blocks[
        :,
        :,
        0,
    ].reshape(
        -1
    )

    pred_coverage = blocks[
        :,
        :,
        1,
    ].reshape(
        -1
    )

    pred_geometry = blocks[
        :,
        :,
        2,
    ].reshape(
        -1
    )


    (
        point_cov_mae,
        point_cov_rmse,
        point_cov_r2,
    ) = prediction_metrics(
        y,
        pred_coverage,
    )


    (
        point_geo_mae,
        point_geo_rmse,
        point_geo_r2,
    ) = prediction_metrics(
        y,
        pred_geometry,
    )


    point_mae_reduction = (
        (
            point_cov_mae
            - point_geo_mae
        )
        / point_cov_mae
    )


    point_rmse_reduction = (
        (
            point_cov_rmse
            - point_geo_rmse
        )
        / point_cov_rmse
    )


    point_r2_gain = (
        point_geo_r2
        - point_cov_r2
    )


    boot_mae_reduction = np.empty(
        N_BOOT,
        dtype=float,
    )

    boot_rmse_reduction = np.empty(
        N_BOOT,
        dtype=float,
    )

    boot_r2_gain = np.empty(
        N_BOOT,
        dtype=float,
    )


    for bootstrap_index in range(
        N_BOOT
    ):

        sampled_blocks = blocks[
            bootstrap_indices[
                bootstrap_index
            ]
        ]


        sampled_y = sampled_blocks[
            :,
            :,
            0,
        ].reshape(
            -1
        )

        sampled_cov = sampled_blocks[
            :,
            :,
            1,
        ].reshape(
            -1
        )

        sampled_geo = sampled_blocks[
            :,
            :,
            2,
        ].reshape(
            -1
        )


        (
            cov_mae,
            cov_rmse,
            cov_r2,
        ) = prediction_metrics(
            sampled_y,
            sampled_cov,
        )


        (
            geo_mae,
            geo_rmse,
            geo_r2,
        ) = prediction_metrics(
            sampled_y,
            sampled_geo,
        )


        boot_mae_reduction[
            bootstrap_index
        ] = (
            (
                cov_mae
                - geo_mae
            )
            / cov_mae
        )


        boot_rmse_reduction[
            bootstrap_index
        ] = (
            (
                cov_rmse
                - geo_rmse
            )
            / cov_rmse
        )


        boot_r2_gain[
            bootstrap_index
        ] = (
            geo_r2
            - cov_r2
        )


    mae_lower, mae_upper = percentile_ci(
        boot_mae_reduction
    )

    rmse_lower, rmse_upper = percentile_ci(
        boot_rmse_reduction
    )

    r2_lower, r2_upper = percentile_ci(
        boot_r2_gain
    )


    lopo_rows.append({
        "feature":
            feature,

        "participants":
            16,

        "predictions":
            len(
                y
            ),

        "coverage_only_mae":
            point_cov_mae,

        "coverage_geometry_mae":
            point_geo_mae,

        "mae_reduction_fraction":
            point_mae_reduction,

        "mae_reduction_ci95_lower":
            mae_lower,

        "mae_reduction_ci95_upper":
            mae_upper,

        "bootstrap_probability_mae_improvement":
            float(
                np.mean(
                    boot_mae_reduction
                    > 0
                )
            ),

        "coverage_only_rmse":
            point_cov_rmse,

        "coverage_geometry_rmse":
            point_geo_rmse,

        "rmse_reduction_fraction":
            point_rmse_reduction,

        "rmse_reduction_ci95_lower":
            rmse_lower,

        "rmse_reduction_ci95_upper":
            rmse_upper,

        "bootstrap_probability_rmse_improvement":
            float(
                np.mean(
                    boot_rmse_reduction
                    > 0
                )
            ),

        "coverage_only_r2":
            point_cov_r2,

        "coverage_geometry_r2":
            point_geo_r2,

        "r2_gain":
            point_r2_gain,

        "r2_gain_ci95_lower":
            r2_lower,

        "r2_gain_ci95_upper":
            r2_upper,

        "bootstrap_probability_r2_gain":
            float(
                np.mean(
                    boot_r2_gain
                    > 0
                )
            ),
    })


lopo_ci = pd.DataFrame(
    lopo_rows
)


lopo_ci.to_csv(
    LOPO_OUT,
    index=False,
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 80)
print("BOOTSTRAP LOPO PREDICTION UNCERTAINTY")
print("=" * 80)

print(
    lopo_ci[
        [
            "feature",
            "mae_reduction_fraction",
            "mae_reduction_ci95_lower",
            "mae_reduction_ci95_upper",
            "bootstrap_probability_mae_improvement",
            "rmse_reduction_fraction",
            "rmse_reduction_ci95_lower",
            "rmse_reduction_ci95_upper",
            "bootstrap_probability_rmse_improvement",
            "r2_gain",
            "r2_gain_ci95_lower",
            "r2_gain_ci95_upper",
            "bootstrap_probability_r2_gain",
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
print("BOOTSTRAP STRUCTURAL TREND")
print("=" * 80)

print(
    trend_ci[
        [
            "feature",
            "target_coverage",
            "descriptor",
            "point_median_rho",
            "median_rho_ci95_lower",
            "median_rho_ci95_upper",
            "bootstrap_probability_expected_direction",
        ]
    ]
    .sort_values(
        [
            "feature",
            "target_coverage",
            "descriptor",
        ],
        ascending=[
            True,
            False,
            True,
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
print("BOOTSTRAP RELIABILITY CURVE RANGE")
print("=" * 80)

print(
    curve_ci[
        [
            "feature",
            "target_coverage",
            "missingness_design",
            "point_estimate",
            "ci95_lower",
            "ci95_upper",
        ]
    ]
    .sort_values(
        [
            "feature",
            "target_coverage",
            "missingness_design",
        ],
        ascending=[
            True,
            False,
            True,
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
print("OUTPUTS")
print("=" * 80)

print(CURVE_OUT)
print(TREND_OUT)
print(LOPO_OUT)

print()
print("=" * 80)
print("DONE")
print("=" * 80)
