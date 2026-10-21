"""Audit intermediate temporal-missingness geometries before feature rebuilding.

No GPS features are computed here.

The purpose is to verify that controlled k-run masks fill the structural
gap between the already validated extremes:

- one contiguous missing block
- randomly distributed missing bins

Candidate intermediate geometries:
    2, 4, 8, 16 missing runs

The design is not considered frozen until this audit is inspected.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from raphe.features.missingness_simulation import (
    make_balanced_k_run_missing_bin_mask,
)

from raphe.features.missingness_structure import (
    compute_missingness_structure,
)


BIN_SIZE_MS = 5 * 60_000

PROVISIONAL_BASE_SEED = 20261021

COVERAGES = [
    0.90,
    0.75,
    0.50,
    0.25,
]

RUN_COUNTS = [
    2,
    4,
    8,
    16,
]

REPLICATES = range(
    5
)


ROOT = Path(
    "/shared/Digital Clinic/"
    "raphe_outputs/digital_clinic/"
    "feature_reliability/"
    "gps_observability"
)

LOCKED_DAYS_PATH = (
    ROOT
    / "gps_validation_locked_days.parquet"
)

EXISTING_PATH = (
    ROOT
    / "gps_validation_degradation_with_missingness_structure.parquet"
)

AUDIT_OUT = (
    ROOT
    / "gps_intermediate_missingness_design_audit.parquet"
)

SUMMARY_OUT = (
    ROOT
    / "gps_intermediate_missingness_design_summary.csv"
)

COMBINED_OUT = (
    ROOT
    / "gps_missingness_design_combined_support.csv"
)


def stable_seed(
    *parts,
) -> int:

    text = "|".join(
        str(
            part
        )
        for part in parts
    )


    digest = hashlib.blake2b(
        text.encode(
            "utf-8"
        ),
        digest_size=8,
    ).digest()


    return (
        int.from_bytes(
            digest,
            byteorder="little",
            signed=False,
        )
        % (
            2 ** 32
        )
    )


locked = pd.read_parquet(
    LOCKED_DAYS_PATH
)


locked[
    "date"
] = locked[
    "date"
].astype(
    str
)


if len(
    locked
) != 112:

    raise RuntimeError(
        f"Expected 112 locked days, "
        f"found {len(locked)}."
    )


if not (
    (
        locked[
            "window_end_ms"
        ]
        - locked[
            "window_start_ms"
        ]
    )
    == 86_400_000
).all():

    raise RuntimeError(
        "Expected every locked day to span exactly 24 hours."
    )


rows = []


for day in locked.itertuples(
    index=False
):

    participant_id = str(
        day.participant_id
    )

    date = str(
        day.date
    )

    start_ms = int(
        day.window_start_ms
    )

    end_ms = int(
        day.window_end_ms
    )


    for coverage in COVERAGES:

        for n_runs in RUN_COUNTS:

            for replicate in REPLICATES:

                seed = stable_seed(
                    PROVISIONAL_BASE_SEED,
                    "intermediate_missingness",
                    participant_id,
                    date,
                    coverage,
                    n_runs,
                    replicate,
                )


                mask, mask_info = (
                    make_balanced_k_run_missing_bin_mask(
                        window_start_ms=start_ms,
                        window_end_ms=end_ms,
                        target_coverage=coverage,
                        n_runs=n_runs,
                        bin_size_ms=BIN_SIZE_MS,
                        seed=seed,
                    )
                )


                structure = (
                    compute_missingness_structure(
                        mask,
                        bin_size_ms=BIN_SIZE_MS,
                    )
                )


                if (
                    structure[
                        "missing_run_count"
                    ]
                    != n_runs
                ):
                    raise RuntimeError(
                        "Exact run-count check failed."
                    )


                rows.append({
                    "participant_id":
                        participant_id,

                    "date":
                        date,

                    "sampling_density_quartile":
                        day.sampling_density_quartile,

                    "target_coverage":
                        coverage,

                    "n_runs_design":
                        n_runs,

                    "replicate":
                        replicate,

                    "seed":
                        seed,

                    "planned_coverage":
                        mask_info[
                            "planned_coverage"
                        ],

                    **structure,
                })


audit = pd.DataFrame(
    rows
)


expected_rows = (
    112
    * len(
        COVERAGES
    )
    * len(
        RUN_COUNTS
    )
    * len(
        list(
            REPLICATES
        )
    )
)


if len(
    audit
) != expected_rows:

    raise RuntimeError(
        f"Expected {expected_rows:,} rows, "
        f"found {len(audit):,}."
    )


audit.to_parquet(
    AUDIT_OUT,
    index=False,
)


# ============================================================
# INTERMEDIATE DESIGN SUMMARY
# ============================================================

summary = (
    audit
    .groupby(
        [
            "target_coverage",
            "n_runs_design",
        ]
    )
    .agg(
        rows=(
            "participant_id",
            "size",
        ),

        planned_coverage=(
            "planned_coverage",
            "median",
        ),

        missing_runs=(
            "missing_run_count",
            "median",
        ),

        longest_gap_hours=(
            "longest_missing_ms",
            lambda x:
                np.median(
                    x
                )
                / 3_600_000,
        ),

        longest_missing_share=(
            "longest_missing_fraction_of_missing",
            "median",
        ),

        concentration=(
            "missing_run_concentration",
            "median",
        ),

        fragmentation=(
            "missing_fragmentation",
            "median",
        ),
    )
    .reset_index()
)


summary.to_csv(
    SUMMARY_OUT,
    index=False,
)


# ============================================================
# COMBINE WITH EXISTING EXTREMES
# ============================================================

existing = pd.read_parquet(
    EXISTING_PATH
)


existing_summary = (
    existing
    .groupby(
        [
            "missingness_pattern",
            "target_coverage",
        ]
    )
    .agg(
        longest_gap_hours=(
            "longest_missing_ms",
            lambda x:
                np.median(
                    x
                )
                / 3_600_000,
        ),

        longest_missing_share=(
            "longest_missing_fraction_of_missing",
            "median",
        ),

        concentration=(
            "missing_run_concentration",
            "median",
        ),

        fragmentation=(
            "missing_fragmentation",
            "median",
        ),
    )
    .reset_index()
)


existing_summary[
    "design"
] = existing_summary[
    "missingness_pattern"
].map({
    "contiguous":
        "k1_contiguous",

    "distributed":
        "random_distributed",
})


existing_summary = (
    existing_summary[
        [
            "target_coverage",
            "design",
            "longest_gap_hours",
            "longest_missing_share",
            "concentration",
            "fragmentation",
        ]
    ]
)


intermediate_summary = (
    summary.copy()
)


intermediate_summary[
    "design"
] = (
    "k"
    + intermediate_summary[
        "n_runs_design"
    ].astype(
        str
    )
)


intermediate_summary = (
    intermediate_summary[
        [
            "target_coverage",
            "design",
            "longest_gap_hours",
            "longest_missing_share",
            "concentration",
            "fragmentation",
        ]
    ]
)


combined = pd.concat(
    [
        existing_summary,
        intermediate_summary,
    ],
    ignore_index=True,
)


design_order = {
    "k1_contiguous":
        1,

    "k2":
        2,

    "k4":
        4,

    "k8":
        8,

    "k16":
        16,

    "random_distributed":
        999,
}


combined[
    "_order"
] = combined[
    "design"
].map(
    design_order
)


combined = (
    combined
    .sort_values(
        [
            "target_coverage",
            "_order",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .drop(
        columns=[
            "_order"
        ]
    )
)


combined.to_csv(
    COMBINED_OUT,
    index=False,
)


print("=" * 80)
print("RAPHE INTERMEDIATE MISSINGNESS DESIGN AUDIT")
print("=" * 80)

print(
    "Audit rows:",
    f"{len(audit):,}",
)

print(
    "Participants:",
    audit[
        "participant_id"
    ].nunique(),
)

print(
    "Exact-run-count checks: PASS"
)


print()
print("=" * 80)
print("INTERMEDIATE GEOMETRY SUMMARY")
print("=" * 80)

print(
    summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print()
print("=" * 80)
print("COMBINED STRUCTURAL SUPPORT")
print("=" * 80)

print(
    combined.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print()
print("=" * 80)
print("OUTPUTS")
print("=" * 80)

print(AUDIT_OUT)
print(SUMMARY_OUT)
print(COMBINED_OUT)

print()
print("=" * 80)
print("DONE")
print("=" * 80)
