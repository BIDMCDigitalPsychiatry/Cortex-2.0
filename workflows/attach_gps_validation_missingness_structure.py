"""Reconstruct exact validation masks and attach temporal missingness structure.

This workflow reproduces the exact deterministic masks used in the frozen
held-out GPS degradation experiment and describes each mask continuously.

Important
---------
This does NOT rerun GPS feature extraction.

It reconstructs masks using exactly the same:
- BASE_SEED
- blake2b seed construction
- participant/date/coverage/pattern/replicate key
- 5-minute masking bins
- make_missing_bin_mask() implementation

The goal is to replace coarse labels such as "distributed" and "contiguous"
with explicit descriptors of temporal missingness geometry.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from cortex2_engine.features.gps_degradation import (
    make_missing_bin_mask,
)

from cortex2_engine.features.missingness_structure import (
    compute_missingness_structure,
)


# ============================================================
# FROZEN VALIDATION PARAMETERS
# ============================================================

BIN_SIZE_MS = 5 * 60_000
BASE_SEED = 20261020

EXPECTED_ROWS = 4480
EXPECTED_PARTICIPANTS = 16
EXPECTED_CONDITIONS = 40


# ============================================================
# PATHS
# ============================================================

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

REFERENCE_PATH = (
    ROOT
    / "gps_validation_reference_location_features.parquet"
)

LOCKED_DAYS_PATH = (
    ROOT
    / "gps_validation_locked_days.parquet"
)

OUT_PATH = (
    ROOT
    / "gps_validation_degradation_with_missingness_structure.parquet"
)

SUMMARY_OUT = (
    ROOT
    / "gps_validation_missingness_structure_summary.csv"
)

DESCRIPTOR_RANGE_OUT = (
    ROOT
    / "gps_validation_missingness_descriptor_ranges.csv"
)


# ============================================================
# EXACT SEED FUNCTION FROM FROZEN VALIDATION WORKFLOW
# ============================================================

def stable_seed(*parts) -> int:
    """Stable deterministic seed independent of Python hash state."""

    text = "|".join(
        str(part)
        for part in parts
    )

    digest = hashlib.blake2b(
        text.encode("utf-8"),
        digest_size=8,
    ).digest()

    return (
        int.from_bytes(
            digest,
            byteorder="little",
            signed=False,
        )
        % (2 ** 32)
    )


# ============================================================
# WINDOW COLUMN DISCOVERY
# ============================================================

WINDOW_COLUMN_PAIRS = [
    (
        "day_start_ms",
        "day_end_ms",
    ),
    (
        "window_start_ms",
        "window_end_ms",
    ),
    (
        "start_ms",
        "end_ms",
    ),
]


def find_window_columns(
    df: pd.DataFrame,
):
    for start_col, end_col in WINDOW_COLUMN_PAIRS:
        if (
            start_col in df.columns
            and end_col in df.columns
        ):
            return (
                start_col,
                end_col,
            )

    return None


# ============================================================
# LOAD
# ============================================================

df = pd.read_parquet(
    NORMALIZED_PATH
)

reference = pd.read_parquet(
    REFERENCE_PATH
)

locked_days = pd.read_parquet(
    LOCKED_DAYS_PATH
)


df["date"] = (
    df["date"]
    .astype(str)
)

reference["date"] = (
    reference["date"]
    .astype(str)
)

locked_days["date"] = (
    locked_days["date"]
    .astype(str)
)


print("=" * 80)
print("Cortex 2.0 GPS VALIDATION MISSINGNESS-STRUCTURE RECONSTRUCTION")
print("=" * 80)

print(
    "Input rows:",
    f"{len(df):,}",
)

print(
    "Participants:",
    df[
        "participant_id"
    ].nunique(),
)


if len(df) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_ROWS} rows, "
        f"found {len(df)}."
    )


if (
    df[
        "participant_id"
    ].nunique()
    != EXPECTED_PARTICIPANTS
):
    raise RuntimeError(
        "Unexpected validation participant count."
    )


conditions = (
    df[
        [
            "target_coverage",
            "missingness_pattern",
            "replicate",
        ]
    ]
    .drop_duplicates()
)


if len(conditions) != EXPECTED_CONDITIONS:
    raise RuntimeError(
        f"Expected {EXPECTED_CONDITIONS} conditions, "
        f"found {len(conditions)}."
    )


# ============================================================
# GET EXACT FROZEN VALIDATION DAY WINDOWS
# ============================================================

required_locked_columns = {
    "participant_id",
    "date",
    "window_start_ms",
    "window_end_ms",
}

missing_locked_columns = (
    required_locked_columns
    - set(
        locked_days.columns
    )
)

if missing_locked_columns:
    raise RuntimeError(
        "Locked validation-day table is missing "
        f"required columns: {sorted(missing_locked_columns)}"
    )


window_table = (
    locked_days[
        [
            "participant_id",
            "date",
            "window_start_ms",
            "window_end_ms",
        ]
    ]
    .drop_duplicates()
    .rename(
        columns={
            "window_start_ms":
                "validation_window_start_ms",

            "window_end_ms":
                "validation_window_end_ms",
        }
    )
)


if len(window_table) != 112:
    raise RuntimeError(
        f"Expected 112 locked validation windows, "
        f"found {len(window_table)}."
    )


if window_table.duplicated(
    subset=[
        "participant_id",
        "date",
    ]
).any():
    raise RuntimeError(
        "Locked validation windows contain duplicate "
        "participant/date keys."
    )


df = df.merge(
    window_table,
    on=[
        "participant_id",
        "date",
    ],
    how="left",
    validate="many_to_one",
)


start_col = (
    "validation_window_start_ms"
)

end_col = (
    "validation_window_end_ms"
)


if (
    df[
        start_col
    ].isna().any()
    or df[
        end_col
    ].isna().any()
):
    raise RuntimeError(
        "Some degradation rows could not be matched "
        "to their frozen validation-day windows."
    )


window_duration = (
    df[
        end_col
    ]
    - df[
        start_col
    ]
)


if not (
    window_duration
    == 86_400_000
).all():
    raise RuntimeError(
        "Expected all locked validation windows "
        "to be exactly 24 hours."
    )


# ============================================================
# RECONSTRUCT EXACT MASKS
# ============================================================

structure_rows = []


for row in df[
    [
        "participant_id",
        "date",
        "target_coverage",
        "missingness_pattern",
        "replicate",
        start_col,
        end_col,
    ]
].itertuples(
    index=False
):

    participant_id = str(
        row.participant_id
    )

    date = str(
        row.date
    )

    coverage = float(
        row.target_coverage
    )

    pattern = str(
        row.missingness_pattern
    )

    replicate = int(
        row.replicate
    )

    start_ms = int(
        getattr(
            row,
            start_col
        )
    )

    end_ms = int(
        getattr(
            row,
            end_col
        )
    )


    # EXACT frozen validation seed.
    seed = stable_seed(
        BASE_SEED,
        "validation",
        participant_id,
        date,
        coverage,
        pattern,
        replicate,
    )


    missing_mask, mask_info = (
        make_missing_bin_mask(
            window_start_ms=start_ms,
            window_end_ms=end_ms,
            target_coverage=coverage,
            pattern=pattern,
            bin_size_ms=BIN_SIZE_MS,
            seed=seed,
        )
    )


    structure = (
        compute_missingness_structure(
            missing_mask,
            bin_size_ms=BIN_SIZE_MS,
        )
    )


    if not np.isclose(
        structure[
            "coverage"
        ],
        mask_info[
            "planned_coverage"
        ],
        atol=1e-12,
        rtol=0,
    ):
        raise RuntimeError(
            f"Coverage reconstruction mismatch: "
            f"{participant_id} {date} "
            f"{coverage} {pattern} rep={replicate}"
        )


    structure_rows.append({
        "participant_id":
            participant_id,

        "date":
            date,

        "target_coverage":
            coverage,

        "missingness_pattern":
            pattern,

        "replicate":
            replicate,

        "mask_seed":
            seed,

        "mask_bin_size_ms":
            BIN_SIZE_MS,

        "mask_planned_coverage":
            mask_info[
                "planned_coverage"
            ],

        **structure,
    })


structure_df = pd.DataFrame(
    structure_rows
)


# ============================================================
# INTEGRITY
# ============================================================

join_key = [
    "participant_id",
    "date",
    "target_coverage",
    "missingness_pattern",
    "replicate",
]


if structure_df.duplicated(
    subset=join_key
).any():
    raise RuntimeError(
        "Duplicate reconstructed mask keys."
    )


if len(
    structure_df
) != len(
    df
):
    raise RuntimeError(
        "Unexpected reconstructed mask row count."
    )


result = df.merge(
    structure_df,
    on=join_key,
    how="left",
    validate="one_to_one",
)


if len(
    result
) != EXPECTED_ROWS:
    raise RuntimeError(
        "Structure merge changed row count."
    )


# ============================================================
# SEMANTIC CHECKS
# ============================================================

contiguous = result[
    result[
        "missingness_pattern"
    ]
    == "contiguous"
]


distributed = result[
    result[
        "missingness_pattern"
    ]
    == "distributed"
]


if not (
    contiguous[
        "missing_run_count"
    ]
    == 1
).all():
    raise RuntimeError(
        "Contiguous masks must contain exactly one missing run."
    )


if not np.allclose(
    contiguous[
        "longest_missing_fraction_of_missing"
    ],
    1.0,
):
    raise RuntimeError(
        "Contiguous masks should place all missing time "
        "inside one run."
    )


if not np.allclose(
    contiguous[
        "missing_run_concentration"
    ],
    1.0,
):
    raise RuntimeError(
        "Contiguous masks should have concentration = 1."
    )


# ============================================================
# SAVE FULL TABLE
# ============================================================

result.to_parquet(
    OUT_PATH,
    index=False,
)


# ============================================================
# CONDITION SUMMARY
# ============================================================

summary_columns = [
    "mask_planned_coverage",
    "missing_fraction",
    "missing_run_count",
    "longest_missing_bins",
    "longest_missing_ms",
    "mean_missing_run_bins",
    "median_missing_run_bins",
    "q90_missing_run_bins",
    "longest_missing_fraction_of_missing",
    "missing_run_concentration",
    "missing_fragmentation",
]


summary_rows = []


for (
    pattern,
    coverage,
), x in result.groupby(
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

        "rows":
            len(
                x
            ),

        "participants":
            x[
                "participant_id"
            ].nunique(),
    }


    for column in summary_columns:

        values = pd.to_numeric(
            x[
                column
            ],
            errors="coerce",
        )


        row.update({
            f"{column}_min":
                values.min(),

            f"{column}_median":
                values.median(),

            f"{column}_max":
                values.max(),
        })


    summary_rows.append(
        row
    )


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    SUMMARY_OUT,
    index=False,
)


# ============================================================
# DESCRIPTOR RANGE / SUPPORT AUDIT
# ============================================================

descriptor_rows = []


for pattern, x in result.groupby(
    "missingness_pattern",
    sort=True,
):

    for descriptor in [
        "missing_run_count",
        "longest_missing_bins",
        "longest_missing_fraction_of_missing",
        "missing_run_concentration",
        "missing_fragmentation",
    ]:

        values = pd.to_numeric(
            x[
                descriptor
            ],
            errors="coerce",
        )


        descriptor_rows.append({
            "missingness_pattern":
                pattern,

            "descriptor":
                descriptor,

            "n":
                values.notna().sum(),

            "min":
                values.min(),

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

            "max":
                values.max(),
        })


descriptor_ranges = pd.DataFrame(
    descriptor_rows
)


descriptor_ranges.to_csv(
    DESCRIPTOR_RANGE_OUT,
    index=False,
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 80)
print("RECONSTRUCTION INTEGRITY")
print("=" * 80)

print(
    "Rows reconstructed:",
    f"{len(result):,}",
)

print(
    "Conditions:",
    len(
        conditions
    ),
)

print(
    "Unique n_bins:",
    sorted(
        result[
            "n_bins"
        ]
        .unique()
        .tolist()
    ),
)

print(
    "Contiguous one-run check: PASS"
)

print(
    "Exact planned-coverage check: PASS"
)


print()
print("=" * 80)
print("MISSINGNESS STRUCTURE BY CONDITION")
print("=" * 80)

display = (
    result
    .groupby(
        [
            "missingness_pattern",
            "target_coverage",
        ]
    )
    .agg(
        planned_coverage=(
            "mask_planned_coverage",
            "median",
        ),

        missing_runs_median=(
            "missing_run_count",
            "median",
        ),

        missing_runs_min=(
            "missing_run_count",
            "min",
        ),

        missing_runs_max=(
            "missing_run_count",
            "max",
        ),

        longest_gap_bins_median=(
            "longest_missing_bins",
            "median",
        ),

        longest_gap_hours_median=(
            "longest_missing_ms",
            lambda x:
                np.median(
                    x
                )
                / 3_600_000,
        ),

        longest_missing_share_median=(
            "longest_missing_fraction_of_missing",
            "median",
        ),

        concentration_median=(
            "missing_run_concentration",
            "median",
        ),

        fragmentation_median=(
            "missing_fragmentation",
            "median",
        ),
    )
    .reset_index()
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
)


print(
    display.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print()
print("=" * 80)
print("DESCRIPTOR SUPPORT BY SIMULATION PATTERN")
print("=" * 80)

print(
    descriptor_ranges.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print()
print("=" * 80)
print("OUTPUTS")
print("=" * 80)

print(OUT_PATH)
print(SUMMARY_OUT)
print(DESCRIPTOR_RANGE_OUT)

print()
print("=" * 80)
print("DONE")
print("=" * 80)
