"""Sensitivity analysis for Cortex 2.0 natural-variability normalization.

Compares three participant-specific natural variability scales:

1. median absolute difference across all reference-day pairs
2. reference-day IQR
3. raw MAD around the participant reference median

This is a robustness analysis only. It does not select a threshold
or retune the GPS feature algorithm.
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

NORMALIZED_PATH = (
    ROOT
    / "gps_validation_degradation_normalized_errors.parquet"
)

DENOMINATOR_OUT = (
    ROOT
    / "gps_normalization_scale_sensitivity_denominators.csv"
)

CONDITION_OUT = (
    ROOT
    / "gps_normalization_scale_sensitivity_conditions.csv"
)

PATTERN_OUT = (
    ROOT
    / "gps_normalization_scale_sensitivity_pattern_contrast.csv"
)


FEATURES = [
    "entropy",
    "dominant_fraction",
    "effective_places",
]


SCALES = {
    "pairwise_median": {
        "entropy":
            "entropy_normalized_error",

        "dominant_fraction":
            "dominant_fraction_normalized_error",

        "effective_places":
            "effective_places_normalized_error",
    },

    "iqr": {
        "entropy":
            "entropy_error_over_iqr",

        "dominant_fraction":
            "dominant_fraction_error_over_iqr",

        "effective_places":
            "effective_places_error_over_iqr",
    },

    "mad": {
        "entropy":
            "entropy_error_over_mad",

        "dominant_fraction":
            "dominant_fraction_error_over_mad",

        "effective_places":
            "effective_places_error_over_mad",
    },
}


DENOMINATOR_COLUMNS = {
    "entropy": {
        "pairwise_median":
            "entropy_pairwise_median_abs_diff",

        "iqr":
            "entropy_iqr",

        "mad":
            "entropy_mad",
    },

    "dominant_fraction": {
        "pairwise_median":
            "dominant_fraction_pairwise_median_abs_diff",

        "iqr":
            "dominant_fraction_iqr",

        "mad":
            "dominant_fraction_mad",
    },

    "effective_places": {
        "pairwise_median":
            "effective_places_pairwise_median_abs_diff",

        "iqr":
            "effective_places_iqr",

        "mad":
            "effective_places_mad",
    },
}


def safe_spearman(
    x,
    y,
):
    x = pd.to_numeric(
        x,
        errors="coerce",
    )

    y = pd.to_numeric(
        y,
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


df = pd.read_parquet(
    NORMALIZED_PATH
)


print("=" * 80)
print("Cortex 2.0 GPS NORMALIZATION-SCALE SENSITIVITY")
print("=" * 80)

print(
    "Rows:",
    f"{len(df):,}",
)

print(
    "Participants:",
    df[
        "participant_id"
    ].nunique(),
)


# ============================================================
# UNIQUE PARTICIPANT DENOMINATORS
# ============================================================

denominator_columns = [
    column
    for feature in FEATURES
    for column in DENOMINATOR_COLUMNS[
        feature
    ].values()
]


participants = (
    df[
        [
            "participant_id",
            "sampling_density_quartile",
        ]
        + denominator_columns
    ]
    .drop_duplicates(
        subset=[
            "participant_id",
        ]
    )
    .copy()
)


if len(participants) != 16:

    raise RuntimeError(
        f"Expected 16 participants, "
        f"found {len(participants)}."
    )


# ============================================================
# DENOMINATOR AUDIT + AGREEMENT
# ============================================================

denominator_rows = []


for feature in FEATURES:

    primary_col = (
        DENOMINATOR_COLUMNS[
            feature
        ][
            "pairwise_median"
        ]
    )


    for scale in [
        "pairwise_median",
        "iqr",
        "mad",
    ]:

        column = (
            DENOMINATOR_COLUMNS[
                feature
            ][scale]
        )

        values = pd.to_numeric(
            participants[
                column
            ],
            errors="coerce",
        )


        invalid = (
            values.isna()
            | ~np.isfinite(
                values
            )
            | (
                values
                <= 0
            )
        )


        valid_values = values[
            ~invalid
        ]


        if scale == "pairwise_median":

            spearman_vs_primary = 1.0
            median_ratio_vs_primary = 1.0

        else:

            primary = pd.to_numeric(
                participants[
                    primary_col
                ],
                errors="coerce",
            )

            valid_ratio = (
                values.notna()
                & primary.notna()
                & np.isfinite(values)
                & np.isfinite(primary)
                & (values > 0)
                & (primary > 0)
            )


            spearman_vs_primary = (
                safe_spearman(
                    primary,
                    values,
                )
            )


            median_ratio_vs_primary = float(
                (
                    values[
                        valid_ratio
                    ]
                    / primary[
                        valid_ratio
                    ]
                ).median()
            )


        denominator_rows.append({
            "feature":
                feature,

            "scale":
                scale,

            "participants":
                len(
                    participants
                ),

            "invalid_or_zero_denominators":
                int(
                    invalid.sum()
                ),

            "median_denominator":
                (
                    float(
                        valid_values.median()
                    )
                    if len(
                        valid_values
                    )
                    else np.nan
                ),

            "q25_denominator":
                (
                    float(
                        valid_values.quantile(
                            0.25
                        )
                    )
                    if len(
                        valid_values
                    )
                    else np.nan
                ),

            "q75_denominator":
                (
                    float(
                        valid_values.quantile(
                            0.75
                        )
                    )
                    if len(
                        valid_values
                    )
                    else np.nan
                ),

            "spearman_vs_pairwise_median":
                spearman_vs_primary,

            "median_scale_ratio_vs_pairwise":
                median_ratio_vs_primary,
        })


denominator_summary = pd.DataFrame(
    denominator_rows
)


denominator_summary.to_csv(
    DENOMINATOR_OUT,
    index=False,
)


print()
print("=" * 80)
print("DENOMINATOR SCALE AUDIT")
print("=" * 80)

print(
    denominator_summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# ============================================================
# PARTICIPANT × REPLICATE MEDIANS FOR ALL NORMALIZATIONS
# ============================================================

participant_replicate_rows = []


group_columns = [
    "missingness_pattern",
    "target_coverage",
    "replicate",
    "sampling_density_quartile",
    "participant_id",
]


for key, x in df.groupby(
    group_columns,
    sort=True,
):

    (
        pattern,
        coverage,
        replicate,
        quartile,
        participant_id,
    ) = key


    row = {
        "missingness_pattern":
            pattern,

        "target_coverage":
            float(
                coverage
            ),

        "replicate":
            int(
                replicate
            ),

        "sampling_density_quartile":
            quartile,

        "participant_id":
            participant_id,
    }


    for feature in FEATURES:

        for scale, mapping in SCALES.items():

            column = mapping[
                feature
            ]

            row[
                f"{feature}_{scale}"
            ] = (
                pd.to_numeric(
                    x[
                        column
                    ],
                    errors="coerce",
                )
                .median()
            )


    participant_replicate_rows.append(
        row
    )


participant_replicate = pd.DataFrame(
    participant_replicate_rows
)


# ============================================================
# PARTICIPANT-BALANCED CONDITION SUMMARY
# ============================================================

condition_rows = []


for (
    pattern,
    coverage,
), x in participant_replicate.groupby(
    [
        "missingness_pattern",
        "target_coverage",
    ],
    sort=True,
):

    for feature in FEATURES:

        condition_row = {
            "missingness_pattern":
                pattern,

            "target_coverage":
                float(
                    coverage
                ),

            "feature":
                feature,

            "participant_replicates":
                len(
                    x
                ),
        }


        for scale in [
            "pairwise_median",
            "iqr",
            "mad",
        ]:

            values = pd.to_numeric(
                x[
                    f"{feature}_{scale}"
                ],
                errors="coerce",
            )


            condition_row[
                f"median_normalized_error_{scale}"
            ] = (
                values.median()
            )

            condition_row[
                f"p90_normalized_error_{scale}"
            ] = (
                values.quantile(
                    0.90
                )
            )


        condition_rows.append(
            condition_row
        )


conditions = pd.DataFrame(
    condition_rows
)


conditions.to_csv(
    CONDITION_OUT,
    index=False,
)


print()
print("=" * 80)
print("NORMALIZED RELIABILITY BY SCALE")
print("=" * 80)

print(
    conditions[
        [
            "missingness_pattern",
            "target_coverage",
            "feature",
            "median_normalized_error_pairwise_median",
            "median_normalized_error_iqr",
            "median_normalized_error_mad",
            "p90_normalized_error_pairwise_median",
            "p90_normalized_error_iqr",
            "p90_normalized_error_mad",
        ]
    ]
    .sort_values(
        [
            "feature",
            "missingness_pattern",
            "target_coverage",
        ],
        ascending=[
            True,
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


# ============================================================
# CONTIGUOUS MINUS DISTRIBUTED UNDER EACH SCALE
# ============================================================

distributed = conditions[
    conditions[
        "missingness_pattern"
    ]
    == "distributed"
].copy()


contiguous = conditions[
    conditions[
        "missingness_pattern"
    ]
    == "contiguous"
].copy()


contrast = contiguous.merge(
    distributed,
    on=[
        "target_coverage",
        "feature",
    ],
    suffixes=(
        "_contiguous",
        "_distributed",
    ),
    validate="one_to_one",
)


for scale in [
    "pairwise_median",
    "iqr",
    "mad",
]:

    contrast[
        f"difference_{scale}"
    ] = (
        contrast[
            f"median_normalized_error_"
            f"{scale}_contiguous"
        ]
        - contrast[
            f"median_normalized_error_"
            f"{scale}_distributed"
        ]
    )


contrast.to_csv(
    PATTERN_OUT,
    index=False,
)


print()
print("=" * 80)
print("CONTIGUOUS MINUS DISTRIBUTED BY SCALE")
print("=" * 80)

print(
    contrast[
        [
            "target_coverage",
            "feature",
            "difference_pairwise_median",
            "difference_iqr",
            "difference_mad",
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


# ============================================================
# CONSISTENCY CHECK
# ============================================================

difference_columns = [
    "difference_pairwise_median",
    "difference_iqr",
    "difference_mad",
]


all_positive = bool(
    (
        contrast[
            difference_columns
        ]
        > 0
    )
    .all()
    .all()
)


print()
print("=" * 80)
print("ROBUSTNESS CHECK")
print("=" * 80)

print(
    "Contiguous error > distributed error "
    "for every feature × coverage × scale:",
    all_positive,
)


print()
print("=" * 80)
print("OUTPUTS")
print("=" * 80)

print(DENOMINATOR_OUT)
print(CONDITION_OUT)
print(PATTERN_OUT)

print()
print("=" * 80)
print("DONE")
print("=" * 80)
