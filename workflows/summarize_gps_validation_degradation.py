"""Summarize held-out RAPHE GPS location-feature validation.

Validation design is frozen:
- 16 held-out participants
- 7 reference days/person
- 112 participant-days
- compatibility_multiplier = 2.0
- 4 target coverages
- distributed vs contiguous missingness
- 5 replicates

Produces participant-balanced validation reliability curves, sampling-density
stratum summaries, missingness-pattern contrasts, and a direct descriptive
comparison with the earlier development experiment.

No new parameter tuning or reliability threshold is introduced here.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "/shared/Digital Clinic/"
    "raphe_outputs/digital_clinic/"
    "feature_reliability/"
    "gps_observability"
)

RUN_DIR = (
    ROOT
    / "validation_degradation_runs"
)

DEV_CURVE_PATH = (
    ROOT
    / "gps_location_degradation_reliability_curves.csv"
)

COMBINED_OUT = (
    ROOT
    / "gps_validation_degradation_all_runs.parquet"
)

PARTICIPANT_REPLICATE_OUT = (
    ROOT
    / "gps_validation_degradation_participant_replicate.csv"
)

CURVE_OUT = (
    ROOT
    / "gps_validation_degradation_reliability_curves.csv"
)

QUARTILE_OUT = (
    ROOT
    / "gps_validation_degradation_quartile_summary.csv"
)

PATTERN_OUT = (
    ROOT
    / "gps_validation_degradation_pattern_contrast.csv"
)

DEV_COMPARISON_OUT = (
    ROOT
    / "gps_validation_vs_development.csv"
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

EXPECTED_PARTICIPANTS = 16
EXPECTED_DAYS_PER_PARTICIPANT = 7
EXPECTED_RUN_FILES = 40
EXPECTED_ROWS_PER_RUN = 112

FROZEN_MULTIPLIER = 2.0


# ============================================================
# HELPERS
# ============================================================

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


# ============================================================
# LOAD + INTEGRITY CHECK
# ============================================================

files = sorted(
    RUN_DIR.glob(
        "coverage_*_rep_*.parquet"
    )
)


print("=" * 80)
print("RAPHE HELD-OUT GPS VALIDATION SUMMARY")
print("=" * 80)

print(
    "Run files:",
    len(files),
)


if len(files) != EXPECTED_RUN_FILES:
    raise RuntimeError(
        f"Expected {EXPECTED_RUN_FILES} run files, "
        f"found {len(files)}."
    )


frames = []


for path in files:

    df = pd.read_parquet(
        path
    )

    if len(df) != EXPECTED_ROWS_PER_RUN:
        raise RuntimeError(
            f"{path.name}: expected "
            f"{EXPECTED_ROWS_PER_RUN} rows, "
            f"found {len(df)}."
        )

    frames.append(
        df
    )


all_runs = pd.concat(
    frames,
    ignore_index=True,
)


expected_total_rows = (
    EXPECTED_RUN_FILES
    * EXPECTED_ROWS_PER_RUN
)


if len(all_runs) != expected_total_rows:
    raise RuntimeError(
        f"Expected {expected_total_rows} rows, "
        f"found {len(all_runs)}."
    )


if not np.allclose(
    all_runs[
        "compatibility_multiplier"
    ],
    FROZEN_MULTIPLIER,
):
    raise RuntimeError(
        "Validation contains a compatibility multiplier "
        "other than frozen value 2.0."
    )


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
        f"Unexpected coverages: "
        f"{actual_coverages}"
    )


if actual_patterns != EXPECTED_PATTERNS:
    raise RuntimeError(
        f"Unexpected patterns: "
        f"{actual_patterns}"
    )


if actual_replicates != EXPECTED_REPLICATES:
    raise RuntimeError(
        f"Unexpected replicates: "
        f"{actual_replicates}"
    )


key = [
    "target_coverage",
    "missingness_pattern",
    "replicate",
    "participant_id",
    "date",
]


duplicate_rows = int(
    all_runs.duplicated(
        subset=key
    ).sum()
)


if duplicate_rows:
    raise RuntimeError(
        f"Found {duplicate_rows} duplicated "
        "condition/day rows."
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
        f"Found {missing_reference} missing "
        "reference matches."
    )


if (
    all_runs[
        "participant_id"
    ].nunique()
    != EXPECTED_PARTICIPANTS
):
    raise RuntimeError(
        "Unexpected number of validation participants."
    )


all_runs.to_parquet(
    COMBINED_OUT,
    index=False,
)


print(
    "Combined rows:",
    f"{len(all_runs):,}",
)

print(
    "Unique conditions:",
    len(
        conditions
    ),
)

print(
    "Participants:",
    all_runs[
        "participant_id"
    ].nunique(),
)

print(
    "Integrity check: PASS"
)


# ============================================================
# PARTICIPANT × REPLICATE
# ============================================================

participant_replicate_rows = []


group_columns = [
    "missingness_pattern",
    "target_coverage",
    "replicate",
    "sampling_density_quartile",
    "participant_id",
]


for key_values, x in all_runs.groupby(
    group_columns,
    sort=True,
):

    (
        pattern,
        coverage,
        replicate,
        quartile,
        participant_id,
    ) = key_values


    if len(x) != EXPECTED_DAYS_PER_PARTICIPANT:
        raise RuntimeError(
            f"{participant_id} "
            f"{pattern} "
            f"{coverage} "
            f"rep={replicate}: "
            f"expected 7 days, found {len(x)}."
        )


    participant_replicate_rows.append({
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
            len(x),

        "median_raw_fraction_retained":
            x[
                "raw_fraction_retained"
            ].median(),

        "median_supported_fraction_of_day":
            x[
                "supported_fraction_of_day"
            ].median(),

        # ----------------------------------------------------
        # Entropy
        # ----------------------------------------------------
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

        "entropy_spearman":
            safe_spearman(
                x[
                    "reference_entropy_nats"
                ],
                x[
                    "location_entropy_nats"
                ],
            ),

        # ----------------------------------------------------
        # Dominant-place fraction
        # ----------------------------------------------------
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

        "dominant_spearman":
            safe_spearman(
                x[
                    "reference_dominant_fraction"
                ],
                x[
                    "dominant_place_fraction"
                ],
            ),

        # ----------------------------------------------------
        # Effective number of places
        # ----------------------------------------------------
        "median_abs_effective_places_error":
            x[
                "abs_effective_places_error"
            ].median(),

        # ----------------------------------------------------
        # Fraction of supported time assigned to places
        # ----------------------------------------------------
        "median_abs_place_time_fraction_error":
            x[
                "place_time_fraction_error"
            ].abs().median(),
    })


participant_replicate = pd.DataFrame(
    participant_replicate_rows
)


participant_replicate.to_csv(
    PARTICIPANT_REPLICATE_OUT,
    index=False,
)


expected_participant_replicates = (
    4
    * 2
    * 5
    * 16
)


if len(
    participant_replicate
) != expected_participant_replicates:

    raise RuntimeError(
        f"Expected "
        f"{expected_participant_replicates} "
        "participant-replicate rows, "
        f"found {len(participant_replicate)}."
    )


# ============================================================
# PARTICIPANT-BALANCED VALIDATION CURVES
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

    curve_rows.append({
        "compatibility_multiplier":
            FROZEN_MULTIPLIER,

        "missingness_pattern":
            pattern,

        "target_coverage":
            coverage,

        "participants":
            x[
                "participant_id"
            ].nunique(),

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

        # effective places
        "median_abs_effective_places_error":
            x[
                "median_abs_effective_places_error"
            ].median(),

        # place-time assignment
        "median_abs_place_time_fraction_error":
            x[
                "median_abs_place_time_fraction_error"
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
# SAMPLING-DENSITY QUARTILE SUMMARY
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

    quartile_rows.append({
        "sampling_density_quartile":
            quartile,

        "missingness_pattern":
            pattern,

        "target_coverage":
            coverage,

        "participants":
            x[
                "participant_id"
            ].nunique(),

        "participant_replicates":
            len(x),

        "median_supported_fraction_of_day":
            x[
                "median_supported_fraction_of_day"
            ].median(),

        "median_abs_entropy_error":
            x[
                "median_abs_entropy_error"
            ].median(),

        "p90_abs_entropy_error":
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

        "p90_abs_dominant_error":
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


quartile_summary = pd.DataFrame(
    quartile_rows
)


quartile_summary.to_csv(
    QUARTILE_OUT,
    index=False,
)


# ============================================================
# CONTIGUOUS MINUS DISTRIBUTED
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
        "compatibility_multiplier",
        "target_coverage",
    ],
    suffixes=(
        "_contiguous",
        "_distributed",
    ),
    validate="one_to_one",
)


pattern_contrast[
    "entropy_error_difference_contiguous_minus_distributed"
] = (
    pattern_contrast[
        "median_abs_entropy_error_contiguous"
    ]
    - pattern_contrast[
        "median_abs_entropy_error_distributed"
    ]
)


pattern_contrast[
    "dominant_error_difference_contiguous_minus_distributed"
] = (
    pattern_contrast[
        "median_abs_dominant_error_contiguous"
    ]
    - pattern_contrast[
        "median_abs_dominant_error_distributed"
    ]
)


pattern_contrast.to_csv(
    PATTERN_OUT,
    index=False,
)


# ============================================================
# DEVELOPMENT VS HELD-OUT VALIDATION
# ============================================================

development = pd.read_csv(
    DEV_CURVE_PATH
)


development = development[
    np.isclose(
        development[
            "compatibility_multiplier"
        ],
        FROZEN_MULTIPLIER,
    )
].copy()


if len(development) != 8:
    raise RuntimeError(
        f"Expected 8 development curve rows "
        f"for multiplier 2.0, found "
        f"{len(development)}."
    )


development_keep = development[
    [
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
].copy()


validation_keep = curves[
    [
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
].copy()


comparison = validation_keep.merge(
    development_keep,
    on=[
        "missingness_pattern",
        "target_coverage",
    ],
    suffixes=(
        "_validation",
        "_development",
    ),
    validate="one_to_one",
)


comparison[
    "entropy_error_validation_minus_development"
] = (
    comparison[
        "median_abs_entropy_error_validation"
    ]
    - comparison[
        "median_abs_entropy_error_development"
    ]
)


comparison[
    "dominant_error_validation_minus_development"
] = (
    comparison[
        "median_abs_dominant_error_validation"
    ]
    - comparison[
        "median_abs_dominant_error_development"
    ]
)


comparison[
    "entropy_spearman_validation_minus_development"
] = (
    comparison[
        "median_entropy_spearman_validation"
    ]
    - comparison[
        "median_entropy_spearman_development"
    ]
)


comparison[
    "dominant_spearman_validation_minus_development"
] = (
    comparison[
        "median_dominant_spearman_validation"
    ]
    - comparison[
        "median_dominant_spearman_development"
    ]
)


comparison.to_csv(
    DEV_COMPARISON_OUT,
    index=False,
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 80)
print("HELD-OUT VALIDATION RELIABILITY CURVES")
print("=" * 80)

print(
    curves[
        [
            "missingness_pattern",
            "target_coverage",
            "median_supported_fraction_of_day",
            "median_abs_entropy_error",
            "p90_participant_replicate_abs_entropy_error",
            "median_entropy_bias",
            "median_entropy_spearman",
            "median_abs_dominant_error",
            "p90_participant_replicate_abs_dominant_error",
            "median_dominant_bias",
            "median_dominant_spearman",
        ]
    ]
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


print()
print("=" * 80)
print("CONTIGUOUS MINUS DISTRIBUTED ERROR")
print("=" * 80)

print(
    pattern_contrast[
        [
            "target_coverage",
            "entropy_error_difference_contiguous_minus_distributed",
            "dominant_error_difference_contiguous_minus_distributed",
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


print()
print("=" * 80)
print("VALIDATION MINUS DEVELOPMENT")
print("=" * 80)

print(
    comparison[
        [
            "missingness_pattern",
            "target_coverage",
            "median_abs_entropy_error_development",
            "median_abs_entropy_error_validation",
            "entropy_error_validation_minus_development",
            "median_abs_dominant_error_development",
            "median_abs_dominant_error_validation",
            "dominant_error_validation_minus_development",
            "median_entropy_spearman_development",
            "median_entropy_spearman_validation",
            "median_dominant_spearman_development",
            "median_dominant_spearman_validation",
        ]
    ]
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


print()
print("=" * 80)
print("SAMPLING-DENSITY QUARTILE RELIABILITY")
print("=" * 80)

print(
    quartile_summary[
        [
            "sampling_density_quartile",
            "missingness_pattern",
            "target_coverage",
            "median_abs_entropy_error",
            "median_entropy_spearman",
            "median_abs_dominant_error",
            "median_dominant_spearman",
        ]
    ]
    .sort_values(
        [
            "sampling_density_quartile",
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


print()
print("=" * 80)
print("OUTPUTS")
print("=" * 80)

print(COMBINED_OUT)
print(PARTICIPANT_REPLICATE_OUT)
print(CURVE_OUT)
print(QUARTILE_OUT)
print(PATTERN_OUT)
print(DEV_COMPARISON_OUT)

print()
print("=" * 80)
print("DONE")
print("=" * 80)
