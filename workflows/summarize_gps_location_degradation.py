"""Summarize Cortex 2.0 GPS location-feature degradation experiments.

Combines all controlled-missingness runs and produces:

1. integrity checks
2. participant-level reliability summaries
3. participant-balanced reliability curves
4. contiguous-vs-distributed contrasts
5. descriptive comparison of place-building algorithms

This is descriptive reliability analysis. No universal reliability threshold
is imposed here.
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

RUN_DIR = (
    ROOT
    / "degradation_runs"
)

COMBINED_OUT = (
    ROOT
    / "gps_location_degradation_all_runs.parquet"
)

PARTICIPANT_OUT = (
    ROOT
    / "gps_location_degradation_participant_summary.csv"
)

CURVE_OUT = (
    ROOT
    / "gps_location_degradation_reliability_curves.csv"
)

PATTERN_OUT = (
    ROOT
    / "gps_location_degradation_pattern_contrast.csv"
)

ALGORITHM_OUT = (
    ROOT
    / "gps_location_degradation_algorithm_summary.csv"
)


EXPECTED_COVERAGES = {
    0.90,
    0.75,
    0.50,
    0.25,
}

EXPECTED_PATTERNS = {
    "distributed",
    "contiguous",
}

EXPECTED_REPLICATES = set(
    range(5)
)

EXPECTED_MULTIPLIERS = {
    0.75,
    1.00,
    1.50,
    2.00,
}


# ============================================================
# HELPERS
# ============================================================

def safe_pearson(
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
        x.nunique()
        < 2
        or y.nunique()
        < 2
    ):
        return np.nan

    return float(
        x.corr(
            y,
            method="pearson",
        )
    )


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
        x.nunique()
        < 2
        or y.nunique()
        < 2
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


# ============================================================
# LOAD + INTEGRITY CHECK
# ============================================================

files = sorted(
    RUN_DIR.glob(
        "coverage_*_rep_*.parquet"
    )
)


print("=" * 80)
print("Cortex 2.0 GPS DEGRADATION SUMMARY")
print("=" * 80)

print(
    "Run files:",
    len(files),
)


if len(files) != 40:
    raise RuntimeError(
        f"Expected 40 run files, found {len(files)}."
    )


frames = []


for path in files:

    df = pd.read_parquet(
        path
    )


    if len(df) != 260:

        raise RuntimeError(
            f"{path.name}: expected 260 rows, "
            f"found {len(df)}."
        )


    frames.append(
        df
    )


all_runs = pd.concat(
    frames,
    ignore_index=True,
)


print(
    "Combined rows:",
    f"{len(all_runs):,}",
)


expected_rows = (
    40
    * 260
)


if len(all_runs) != expected_rows:
    raise RuntimeError(
        f"Expected {expected_rows} combined rows, "
        f"found {len(all_runs)}."
    )


# ============================================================
# CONDITION CHECK
# ============================================================

conditions = (
    all_runs[
        [
            "target_coverage",
            "missingness_pattern",
            "replicate",
        ]
    ]
    .drop_duplicates()
)


actual_coverages = set(
    conditions[
        "target_coverage"
    ]
    .astype(float)
    .round(2)
)

actual_patterns = set(
    conditions[
        "missingness_pattern"
    ]
    .astype(str)
)

actual_replicates = set(
    conditions[
        "replicate"
    ]
    .astype(int)
)


if actual_coverages != EXPECTED_COVERAGES:
    raise RuntimeError(
        f"Unexpected coverages: {actual_coverages}"
    )


if actual_patterns != EXPECTED_PATTERNS:
    raise RuntimeError(
        f"Unexpected patterns: {actual_patterns}"
    )


if actual_replicates != EXPECTED_REPLICATES:
    raise RuntimeError(
        f"Unexpected replicates: {actual_replicates}"
    )


actual_multipliers = set(
    all_runs[
        "compatibility_multiplier"
    ]
    .astype(float)
    .round(2)
)


if actual_multipliers != EXPECTED_MULTIPLIERS:
    raise RuntimeError(
        f"Unexpected multipliers: {actual_multipliers}"
    )


duplicate_key = [
    "target_coverage",
    "missingness_pattern",
    "replicate",
    "compatibility_multiplier",
    "participant_id",
    "date",
]


n_duplicates = int(
    all_runs.duplicated(
        subset=duplicate_key
    ).sum()
)


if n_duplicates:
    raise RuntimeError(
        f"Found {n_duplicates} duplicated condition/day rows."
    )


missing_reference = int(
    all_runs[
        "reference_entropy_nats"
    ]
    .isna()
    .sum()
)


if missing_reference:
    raise RuntimeError(
        f"Found {missing_reference} missing reference matches."
    )


all_runs.to_parquet(
    COMBINED_OUT,
    index=False,
)


print(
    "Unique conditions:",
    len(
        conditions
    ),
)

print(
    "Integrity check: PASS"
)


# ============================================================
# PARTICIPANT × REPLICATE RELIABILITY
# ============================================================

participant_replicate_rows = []


group_columns = [
    "compatibility_multiplier",
    "missingness_pattern",
    "target_coverage",
    "replicate",
    "participant_id",
]


for key, x in all_runs.groupby(
    group_columns,
    sort=True,
):

    (
        multiplier,
        pattern,
        coverage,
        replicate,
        participant_id,
    ) = key


    participant_replicate_rows.append({
        "compatibility_multiplier":
            float(
                multiplier
            ),

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

        "participant_id":
            participant_id,

        "days":
            len(x),

        "median_raw_fraction_retained":
            x[
                "raw_fraction_retained"
            ].median(),

        "median_supported_fraction_of_day":
            x[
                "supported_fraction_of_day"
            ].median(),

        # Entropy
        "median_abs_entropy_error":
            x[
                "abs_entropy_error"
            ].median(),

        "mean_abs_entropy_error":
            x[
                "abs_entropy_error"
            ].mean(),

        "rmse_entropy":
            float(
                np.sqrt(
                    np.mean(
                        np.square(
                            x[
                                "entropy_error"
                            ]
                        )
                    )
                )
            ),

        "median_entropy_bias":
            x[
                "entropy_error"
            ].median(),

        "entropy_pearson":
            safe_pearson(
                x[
                    "reference_entropy_nats"
                ],
                x[
                    "location_entropy_nats"
                ],
            ),

        "entropy_spearman":
            safe_spearman(
                x[
                    "reference_entropy_nats"
                ],
                x[
                    "location_entropy_nats"
                ],
            ),

        # Dominant-place fraction
        "median_abs_dominant_error":
            x[
                "abs_dominant_fraction_error"
            ].median(),

        "mean_abs_dominant_error":
            x[
                "abs_dominant_fraction_error"
            ].mean(),

        "rmse_dominant":
            float(
                np.sqrt(
                    np.mean(
                        np.square(
                            x[
                                "dominant_fraction_error"
                            ]
                        )
                    )
                )
            ),

        "median_dominant_bias":
            x[
                "dominant_fraction_error"
            ].median(),

        "dominant_pearson":
            safe_pearson(
                x[
                    "reference_dominant_fraction"
                ],
                x[
                    "dominant_place_fraction"
                ],
            ),

        "dominant_spearman":
            safe_spearman(
                x[
                    "reference_dominant_fraction"
                ],
                x[
                    "dominant_place_fraction"
                ],
            ),
    })


participant_replicate = pd.DataFrame(
    participant_replicate_rows
)


# ============================================================
# PARTICIPANT SUMMARY ACROSS 5 REPLICATES
# ============================================================

participant_rows = []


for key, x in participant_replicate.groupby(
    [
        "compatibility_multiplier",
        "missingness_pattern",
        "target_coverage",
        "participant_id",
    ],
    sort=True,
):

    (
        multiplier,
        pattern,
        coverage,
        participant_id,
    ) = key


    participant_rows.append({
        "compatibility_multiplier":
            multiplier,

        "missingness_pattern":
            pattern,

        "target_coverage":
            coverage,

        "participant_id":
            participant_id,

        "replicates":
            len(x),

        "median_supported_fraction_of_day":
            x[
                "median_supported_fraction_of_day"
            ].median(),

        "median_abs_entropy_error":
            x[
                "median_abs_entropy_error"
            ].median(),

        "p90_replicate_abs_entropy_error":
            x[
                "median_abs_entropy_error"
            ].quantile(
                0.90
            ),

        "median_entropy_bias":
            x[
                "median_entropy_bias"
            ].median(),

        "median_entropy_spearman":
            x[
                "entropy_spearman"
            ].median(),

        "median_abs_dominant_error":
            x[
                "median_abs_dominant_error"
            ].median(),

        "p90_replicate_abs_dominant_error":
            x[
                "median_abs_dominant_error"
            ].quantile(
                0.90
            ),

        "median_dominant_bias":
            x[
                "median_dominant_bias"
            ].median(),

        "median_dominant_spearman":
            x[
                "dominant_spearman"
            ].median(),
    })


participant_summary = pd.DataFrame(
    participant_rows
)


participant_summary.to_csv(
    PARTICIPANT_OUT,
    index=False,
)


# ============================================================
# PARTICIPANT-BALANCED RELIABILITY CURVES
# ============================================================

curve_rows = []


for key, x in participant_replicate.groupby(
    [
        "compatibility_multiplier",
        "missingness_pattern",
        "target_coverage",
    ],
    sort=True,
):

    (
        multiplier,
        pattern,
        coverage,
    ) = key


    curve_rows.append({
        "compatibility_multiplier":
            multiplier,

        "missingness_pattern":
            pattern,

        "target_coverage":
            coverage,

        "participant_replicates":
            len(x),

        "median_supported_fraction_of_day":
            x[
                "median_supported_fraction_of_day"
            ].median(),

        # entropy
        "median_abs_entropy_error":
            x[
                "median_abs_entropy_error"
            ].median(),

        "p90_participant_replicate_abs_entropy_error":
            x[
                "median_abs_entropy_error"
            ].quantile(
                0.90
            ),

        "median_entropy_bias":
            x[
                "median_entropy_bias"
            ].median(),

        "median_entropy_spearman":
            x[
                "entropy_spearman"
            ].median(),

        # dominant
        "median_abs_dominant_error":
            x[
                "median_abs_dominant_error"
            ].median(),

        "p90_participant_replicate_abs_dominant_error":
            x[
                "median_abs_dominant_error"
            ].quantile(
                0.90
            ),

        "median_dominant_bias":
            x[
                "median_dominant_bias"
            ].median(),

        "median_dominant_spearman":
            x[
                "dominant_spearman"
            ].median(),
    })


curves = pd.DataFrame(
    curve_rows
)


curves.to_csv(
    CURVE_OUT,
    index=False,
)


# ============================================================
# CONTIGUOUS - DISTRIBUTED CONTRAST
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


contrast = contiguous.merge(
    distributed,
    on=[
        "compatibility_multiplier",
        "target_coverage",
    ],
    suffixes=(
        "_contiguous",
        "_distributed",
    ),
    validate="one_to_one",
)


contrast[
    "entropy_error_difference_contiguous_minus_distributed"
] = (
    contrast[
        "median_abs_entropy_error_contiguous"
    ]
    - contrast[
        "median_abs_entropy_error_distributed"
    ]
)


contrast[
    "dominant_error_difference_contiguous_minus_distributed"
] = (
    contrast[
        "median_abs_dominant_error_contiguous"
    ]
    - contrast[
        "median_abs_dominant_error_distributed"
    ]
)


contrast.to_csv(
    PATTERN_OUT,
    index=False,
)


# ============================================================
# DESCRIPTIVE ALGORITHM COMPARISON
#
# Equal weighting across tested coverage levels, patterns,
# participants and replicates.
# This is NOT a final optimization criterion.
# ============================================================

algorithm_rows = []


for multiplier, x in participant_replicate.groupby(
    "compatibility_multiplier",
    sort=True,
):

    algorithm_rows.append({
        "compatibility_multiplier":
            multiplier,

        "participant_replicates":
            len(x),

        "median_abs_entropy_error_all_conditions":
            x[
                "median_abs_entropy_error"
            ].median(),

        "p90_abs_entropy_error_all_conditions":
            x[
                "median_abs_entropy_error"
            ].quantile(
                0.90
            ),

        "median_abs_dominant_error_all_conditions":
            x[
                "median_abs_dominant_error"
            ].median(),

        "p90_abs_dominant_error_all_conditions":
            x[
                "median_abs_dominant_error"
            ].quantile(
                0.90
            ),

        "median_entropy_spearman_all_conditions":
            x[
                "entropy_spearman"
            ].median(),

        "median_dominant_spearman_all_conditions":
            x[
                "dominant_spearman"
            ].median(),
    })


algorithm_summary = pd.DataFrame(
    algorithm_rows
)


algorithm_summary.to_csv(
    ALGORITHM_OUT,
    index=False,
)


# ============================================================
# PRINT COMPACT RESULTS
# ============================================================

print()
print("=" * 80)
print("PARTICIPANT-BALANCED RELIABILITY CURVES")
print("=" * 80)

display_columns = [
    "compatibility_multiplier",
    "missingness_pattern",
    "target_coverage",
    "median_supported_fraction_of_day",
    "median_abs_entropy_error",
    "median_entropy_bias",
    "median_entropy_spearman",
    "median_abs_dominant_error",
    "median_dominant_bias",
    "median_dominant_spearman",
]


print(
    curves[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print()
print("=" * 80)
print("CONTIGUOUS MINUS DISTRIBUTED ERROR")
print("=" * 80)

print(
    contrast[
        [
            "compatibility_multiplier",
            "target_coverage",
            "entropy_error_difference_contiguous_minus_distributed",
            "dominant_error_difference_contiguous_minus_distributed",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print()
print("=" * 80)
print("DESCRIPTIVE ALGORITHM COMPARISON")
print("=" * 80)

print(
    algorithm_summary.to_string(
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
print(PARTICIPANT_OUT)
print(CURVE_OUT)
print(PATTERN_OUT)
print(ALGORITHM_OUT)

print()
print("=" * 80)
print("DONE")
print("=" * 80)
