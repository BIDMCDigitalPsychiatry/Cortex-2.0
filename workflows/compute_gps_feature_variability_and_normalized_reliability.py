"""Normalize held-out GPS feature degradation by natural within-person variability.

Purpose
-------
Cortex 2.0 should not ask only:

    How much does a feature change when data are missing?

It should also ask:

    Is that degradation error large relative to how much the feature
    naturally varies within this participant?

Natural variability is estimated ONLY from the locked, well-observed
held-out reference days.

Primary variability scale
-------------------------
For each participant and feature:

    median absolute difference across all pairs of reference days

With 7 reference days this gives 21 pairwise differences per participant.

Additional descriptive scales are also retained:
- IQR
- raw MAD around participant median
- mean pairwise absolute difference

No reliability threshold or good/bad classification is imposed here.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "/shared/Digital Clinic/"
    "cortex2_outputs/digital_clinic/"
    "feature_reliability/"
    "gps_observability"
)

REFERENCE_PATH = (
    ROOT
    / "gps_validation_reference_location_features.parquet"
)

DEGRADATION_PATH = (
    ROOT
    / "gps_validation_degradation_all_runs.parquet"
)


VARIABILITY_OUT = (
    ROOT
    / "gps_validation_reference_feature_variability_by_participant.csv"
)

NORMALIZED_DAILY_OUT = (
    ROOT
    / "gps_validation_degradation_normalized_errors.parquet"
)

PARTICIPANT_REPLICATE_OUT = (
    ROOT
    / "gps_validation_normalized_participant_replicate.csv"
)

CURVE_OUT = (
    ROOT
    / "gps_validation_normalized_reliability_curves.csv"
)

QUARTILE_OUT = (
    ROOT
    / "gps_validation_normalized_reliability_by_density_quartile.csv"
)

PATTERN_OUT = (
    ROOT
    / "gps_validation_normalized_pattern_contrast.csv"
)


FEATURES = {
    "entropy": {
        "reference_column":
            "location_entropy_nats",

        "error_column":
            "abs_entropy_error",
    },

    "dominant_fraction": {
        "reference_column":
            "dominant_place_fraction",

        "error_column":
            "abs_dominant_fraction_error",
    },

    "effective_places": {
        "reference_column":
            "effective_number_places",

        "error_column":
            "abs_effective_places_error",
    },
}


EXPECTED_PARTICIPANTS = 16
EXPECTED_REFERENCE_DAYS = 112
EXPECTED_DAYS_PER_PARTICIPANT = 7
EXPECTED_DEGRADATION_ROWS = 4480


# ============================================================
# HELPERS
# ============================================================

def pairwise_absolute_differences(
    values,
):
    """Return absolute differences for every unique pair."""

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    differences = [
        abs(
            values[i]
            - values[j]
        )
        for i, j in combinations(
            range(
                len(values)
            ),
            2,
        )
    ]

    return np.asarray(
        differences,
        dtype=float,
    )


def safe_normalize(
    numerator,
    denominator,
):
    """Normalize only when denominator is finite and > 0."""

    numerator = pd.to_numeric(
        numerator,
        errors="coerce",
    )

    denominator = pd.to_numeric(
        denominator,
        errors="coerce",
    )

    result = np.full(
        len(numerator),
        np.nan,
        dtype=float,
    )

    valid = (
        numerator.notna()
        & denominator.notna()
        & np.isfinite(
            numerator
        )
        & np.isfinite(
            denominator
        )
        & (
            denominator
            > 0
        )
    )

    result[
        valid.to_numpy()
    ] = (
        numerator[
            valid
        ].to_numpy(
            dtype=float
        )
        / denominator[
            valid
        ].to_numpy(
            dtype=float
        )
    )

    return result


# ============================================================
# LOAD
# ============================================================

reference = pd.read_parquet(
    REFERENCE_PATH
)

degradation = pd.read_parquet(
    DEGRADATION_PATH
)


reference[
    "date"
] = reference[
    "date"
].astype(str)

degradation[
    "date"
] = degradation[
    "date"
].astype(str)


print("=" * 80)
print("Cortex 2.0 GPS NATURAL VARIABILITY + NORMALIZED RELIABILITY")
print("=" * 80)

print(
    "Reference rows:",
    len(
        reference
    ),
)

print(
    "Reference participants:",
    reference[
        "participant_id"
    ].nunique(),
)

print(
    "Degradation rows:",
    f"{len(degradation):,}",
)


if len(
    reference
) != EXPECTED_REFERENCE_DAYS:

    raise RuntimeError(
        f"Expected {EXPECTED_REFERENCE_DAYS} "
        f"reference rows, found {len(reference)}."
    )


if (
    reference[
        "participant_id"
    ].nunique()
    != EXPECTED_PARTICIPANTS
):

    raise RuntimeError(
        "Unexpected number of reference participants."
    )


reference_day_counts = (
    reference
    .groupby(
        "participant_id"
    )
    .size()
)


if not (
    reference_day_counts
    == EXPECTED_DAYS_PER_PARTICIPANT
).all():

    raise RuntimeError(
        "Every participant must have exactly "
        "7 reference days."
    )


if len(
    degradation
) != EXPECTED_DEGRADATION_ROWS:

    raise RuntimeError(
        f"Expected {EXPECTED_DEGRADATION_ROWS} "
        f"degradation rows, found {len(degradation)}."
    )


# ============================================================
# NATURAL WITHIN-PERSON VARIABILITY
# ============================================================

variability_rows = []


for participant_id, x in reference.groupby(
    "participant_id",
    sort=True,
):

    participant_row = {
        "participant_id":
            participant_id,

        "sampling_density_quartile":
            x[
                "sampling_density_quartile"
            ].iloc[0],

        "reference_days":
            len(x),
    }


    for feature_name, spec in FEATURES.items():

        column = spec[
            "reference_column"
        ]


        values = pd.to_numeric(
            x[
                column
            ],
            errors="coerce",
        ).dropna().to_numpy(
            dtype=float
        )


        if len(values) != EXPECTED_DAYS_PER_PARTICIPANT:

            raise RuntimeError(
                f"{participant_id} {feature_name}: "
                "missing reference feature values."
            )


        participant_median = float(
            np.median(
                values
            )
        )


        q25 = float(
            np.quantile(
                values,
                0.25,
            )
        )

        q75 = float(
            np.quantile(
                values,
                0.75,
            )
        )


        iqr = (
            q75
            - q25
        )


        mad = float(
            np.median(
                np.abs(
                    values
                    - participant_median
                )
            )
        )


        pairwise = (
            pairwise_absolute_differences(
                values
            )
        )


        expected_pairs = (
            EXPECTED_DAYS_PER_PARTICIPANT
            * (
                EXPECTED_DAYS_PER_PARTICIPANT
                - 1
            )
            // 2
        )


        if len(
            pairwise
        ) != expected_pairs:

            raise RuntimeError(
                f"{participant_id} {feature_name}: "
                f"expected {expected_pairs} pairwise "
                f"differences, found {len(pairwise)}."
            )


        participant_row.update({
            f"{feature_name}_reference_median":
                participant_median,

            f"{feature_name}_reference_min":
                float(
                    np.min(
                        values
                    )
                ),

            f"{feature_name}_reference_max":
                float(
                    np.max(
                        values
                    )
                ),

            f"{feature_name}_iqr":
                float(
                    iqr
                ),

            f"{feature_name}_mad":
                mad,

            f"{feature_name}_pairwise_median_abs_diff":
                float(
                    np.median(
                        pairwise
                    )
                ),

            f"{feature_name}_pairwise_mean_abs_diff":
                float(
                    np.mean(
                        pairwise
                    )
                ),

            f"{feature_name}_pairwise_q75_abs_diff":
                float(
                    np.quantile(
                        pairwise,
                        0.75,
                    )
                ),

            f"{feature_name}_pairwise_max_abs_diff":
                float(
                    np.max(
                        pairwise
                    )
                ),
        })


    variability_rows.append(
        participant_row
    )


variability = pd.DataFrame(
    variability_rows
)


variability.to_csv(
    VARIABILITY_OUT,
    index=False,
)


# ============================================================
# AUDIT DENOMINATORS
# ============================================================

print()
print("=" * 80)
print("NATURAL VARIABILITY SCALE AUDIT")
print("=" * 80)


audit_rows = []


for feature_name in FEATURES:

    primary_col = (
        f"{feature_name}_"
        "pairwise_median_abs_diff"
    )

    iqr_col = (
        f"{feature_name}_iqr"
    )

    mad_col = (
        f"{feature_name}_mad"
    )


    audit_rows.append({
        "feature":
            feature_name,

        "participants":
            len(
                variability
            ),

        "zero_primary_scale":
            int(
                (
                    variability[
                        primary_col
                    ]
                    <= 0
                ).sum()
            ),

        "median_pairwise_abs_diff":
            variability[
                primary_col
            ].median(),

        "q25_pairwise_abs_diff":
            variability[
                primary_col
            ].quantile(
                0.25
            ),

        "q75_pairwise_abs_diff":
            variability[
                primary_col
            ].quantile(
                0.75
            ),

        "median_iqr":
            variability[
                iqr_col
            ].median(),

        "median_mad":
            variability[
                mad_col
            ].median(),
    })


audit = pd.DataFrame(
    audit_rows
)


print(
    audit.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# ============================================================
# PARTICIPANT-LEVEL VARIABILITY TABLE
# ============================================================

print()
print("=" * 80)
print("PARTICIPANT NATURAL VARIABILITY")
print("=" * 80)


display_columns = [
    "sampling_density_quartile",
    "participant_id",
    "entropy_pairwise_median_abs_diff",
    "entropy_iqr",
    "dominant_fraction_pairwise_median_abs_diff",
    "dominant_fraction_iqr",
    "effective_places_pairwise_median_abs_diff",
    "effective_places_iqr",
]


print(
    variability[
        display_columns
    ]
    .sort_values(
        [
            "sampling_density_quartile",
            "participant_id",
        ]
    )
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# ============================================================
# JOIN VARIABILITY TO EVERY DEGRADED PARTICIPANT-DAY
# ============================================================

normalized = degradation.merge(
    variability,
    on=[
        "participant_id",
        "sampling_density_quartile",
    ],
    how="left",
    validate="many_to_one",
)


if len(
    normalized
) != len(
    degradation
):

    raise RuntimeError(
        "Variability join changed degradation row count."
    )


# ============================================================
# NORMALIZED ERROR
# ============================================================

for feature_name, spec in FEATURES.items():

    error_column = spec[
        "error_column"
    ]


    primary_scale = (
        f"{feature_name}_"
        "pairwise_median_abs_diff"
    )

    iqr_scale = (
        f"{feature_name}_iqr"
    )

    mad_scale = (
        f"{feature_name}_mad"
    )


    normalized[
        f"{feature_name}_normalized_error"
    ] = safe_normalize(
        normalized[
            error_column
        ],
        normalized[
            primary_scale
        ],
    )


    normalized[
        f"{feature_name}_error_over_iqr"
    ] = safe_normalize(
        normalized[
            error_column
        ],
        normalized[
            iqr_scale
        ],
    )


    normalized[
        f"{feature_name}_error_over_mad"
    ] = safe_normalize(
        normalized[
            error_column
        ],
        normalized[
            mad_scale
        ],
    )


normalized.to_parquet(
    NORMALIZED_DAILY_OUT,
    index=False,
)


# ============================================================
# PARTICIPANT × REPLICATE SUMMARY
# ============================================================

participant_replicate_rows = []


for key, x in normalized.groupby(
    [
        "missingness_pattern",
        "target_coverage",
        "replicate",
        "sampling_density_quartile",
        "participant_id",
    ],
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

        "days":
            len(
                x
            ),

        "median_supported_fraction_of_day":
            x[
                "supported_fraction_of_day"
            ].median(),
    }


    for feature_name in FEATURES:

        normalized_col = (
            f"{feature_name}_"
            "normalized_error"
        )


        row.update({
            f"median_{feature_name}_normalized_error":
                x[
                    normalized_col
                ].median(),

            f"mean_{feature_name}_normalized_error":
                x[
                    normalized_col
                ].mean(),

            f"q75_{feature_name}_normalized_error":
                x[
                    normalized_col
                ].quantile(
                    0.75
                ),

            f"max_{feature_name}_normalized_error":
                x[
                    normalized_col
                ].max(),
        })


    participant_replicate_rows.append(
        row
    )


participant_replicate = pd.DataFrame(
    participant_replicate_rows
)


participant_replicate.to_csv(
    PARTICIPANT_REPLICATE_OUT,
    index=False,
)


# ============================================================
# PARTICIPANT-BALANCED NORMALIZED RELIABILITY CURVES
# ============================================================

curve_rows = []


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

    row = {
        "missingness_pattern":
            pattern,

        "target_coverage":
            float(
                coverage
            ),

        "participants":
            x[
                "participant_id"
            ].nunique(),

        "participant_replicates":
            len(
                x
            ),

        "median_supported_fraction_of_day":
            x[
                "median_supported_fraction_of_day"
            ].median(),
    }


    for feature_name in FEATURES:

        column = (
            f"median_{feature_name}_"
            "normalized_error"
        )


        row.update({
            f"q25_{feature_name}_normalized_error":
                x[
                    column
                ].quantile(
                    0.25
                ),

            f"median_{feature_name}_normalized_error":
                x[
                    column
                ].median(),

            f"q75_{feature_name}_normalized_error":
                x[
                    column
                ].quantile(
                    0.75
                ),

            f"p90_{feature_name}_normalized_error":
                x[
                    column
                ].quantile(
                    0.90
                ),
        })


    curve_rows.append(
        row
    )


curves = pd.DataFrame(
    curve_rows
)


curves.to_csv(
    CURVE_OUT,
    index=False,
)


# ============================================================
# SAMPLING-DENSITY STRATA
# ============================================================

quartile_rows = []


for (
    quartile,
    pattern,
    coverage,
), x in participant_replicate.groupby(
    [
        "sampling_density_quartile",
        "missingness_pattern",
        "target_coverage",
    ],
    sort=True,
):

    row = {
        "sampling_density_quartile":
            quartile,

        "missingness_pattern":
            pattern,

        "target_coverage":
            float(
                coverage
            ),

        "participants":
            x[
                "participant_id"
            ].nunique(),
    }


    for feature_name in FEATURES:

        column = (
            f"median_{feature_name}_"
            "normalized_error"
        )


        row.update({
            f"median_{feature_name}_normalized_error":
                x[
                    column
                ].median(),

            f"p90_{feature_name}_normalized_error":
                x[
                    column
                ].quantile(
                    0.90
                ),
        })


    quartile_rows.append(
        row
    )


quartile_summary = pd.DataFrame(
    quartile_rows
)


quartile_summary.to_csv(
    QUARTILE_OUT,
    index=False,
)


# ============================================================
# CONTIGUOUS - DISTRIBUTED NORMALIZED ERROR
# ============================================================

distributed = curves[
    curves[
        "missingness_pattern"
    ]
    == "distributed"
].copy()


contiguous = curves[
    curves[
        "missingness_pattern"
    ]
    == "contiguous"
].copy()


pattern_contrast = contiguous.merge(
    distributed,
    on=[
        "target_coverage",
    ],
    suffixes=(
        "_contiguous",
        "_distributed",
    ),
    validate="one_to_one",
)


for feature_name in FEATURES:

    pattern_contrast[
        f"{feature_name}_normalized_error_"
        "difference_contiguous_minus_distributed"
    ] = (
        pattern_contrast[
            f"median_{feature_name}_"
            "normalized_error_contiguous"
        ]
        - pattern_contrast[
            f"median_{feature_name}_"
            "normalized_error_distributed"
        ]
    )


pattern_contrast.to_csv(
    PATTERN_OUT,
    index=False,
)


# ============================================================
# PRINT NORMALIZED CURVES
# ============================================================

print()
print("=" * 80)
print("PARTICIPANT-BALANCED NORMALIZED RELIABILITY")
print("=" * 80)


curve_display = curves[
    [
        "missingness_pattern",
        "target_coverage",
        "median_supported_fraction_of_day",

        "median_entropy_normalized_error",
        "p90_entropy_normalized_error",

        "median_dominant_fraction_normalized_error",
        "p90_dominant_fraction_normalized_error",

        "median_effective_places_normalized_error",
        "p90_effective_places_normalized_error",
    ]
].copy()


print(
    curve_display
    .sort_values(
        [
            "missingness_pattern",
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
# PRINT DISTRIBUTION, NOT A RELIABILITY THRESHOLD
# ============================================================

print()
print("=" * 80)
print("NORMALIZED ERROR DISTRIBUTION")
print("=" * 80)

print(
    "Interpretation: 1.0 equals the participant's median "
    "absolute difference between two well-observed reference days."
)

print(
    "These values are descriptive only; no good/bad threshold "
    "is being imposed."
)


distribution_rows = []


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

    for feature_name in FEATURES:

        column = (
            f"median_{feature_name}_"
            "normalized_error"
        )


        values = x[
            column
        ].dropna()


        distribution_rows.append({
            "missingness_pattern":
                pattern,

            "target_coverage":
                coverage,

            "feature":
                feature_name,

            "n":
                len(
                    values
                ),

            "q10":
                values.quantile(
                    0.10
                ),

            "q25":
                values.quantile(
                    0.25
                ),

            "median":
                values.median(),

            "q75":
                values.quantile(
                    0.75
                ),

            "q90":
                values.quantile(
                    0.90
                ),
        })


distribution = pd.DataFrame(
    distribution_rows
)


print(
    distribution
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
# PATTERN CONTRAST
# ============================================================

print()
print("=" * 80)
print("NORMALIZED CONTIGUOUS MINUS DISTRIBUTED")
print("=" * 80)

print(
    pattern_contrast[
        [
            "target_coverage",
            "entropy_normalized_error_difference_contiguous_minus_distributed",
            "dominant_fraction_normalized_error_difference_contiguous_minus_distributed",
            "effective_places_normalized_error_difference_contiguous_minus_distributed",
        ]
    ]
    .sort_values(
        "target_coverage",
        ascending=False,
    )
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# ============================================================
# OUTPUTS
# ============================================================

print()
print("=" * 80)
print("OUTPUTS")
print("=" * 80)

print(VARIABILITY_OUT)
print(NORMALIZED_DAILY_OUT)
print(PARTICIPANT_REPLICATE_OUT)
print(CURVE_OUT)
print(QUARTILE_OUT)
print(PATTERN_OUT)

print()
print("=" * 80)
print("DONE")
print("=" * 80)
