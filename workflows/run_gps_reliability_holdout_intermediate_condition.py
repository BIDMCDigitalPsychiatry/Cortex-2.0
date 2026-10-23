"""Run one RAPHE intermediate GPS missingness-geometry condition.

This experiment extends the frozen held-out GPS degradation validation with
controlled intermediate temporal missingness geometries.

Frozen feature pipeline:
- same 16 validation participants / 112 reference days
- same raw GPS cache
- same 60-second continuity policy
- same 5-minute masking bins
- same compatibility_multiplier = 2.0
- same spatial semantics, anchor microsegments, participant-level place
  construction, and daily location features
- same frozen complete-data reference

Only the missingness geometry changes.

Intermediate geometries:
    k = 2, 4, 8, 16 balanced missing runs

The original contiguous and random-distributed runs are NOT regenerated here.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from raphe.features.gps_degradation import (
    apply_missing_bin_mask_to_points,
)
from raphe.features.missingness_simulation import (
    make_balanced_k_run_missing_bin_mask,
)
from raphe.features.missingness_structure import (
    compute_missingness_structure,
)
from raphe.features.gps_intervals import (
    build_gps_intervals,
)
from raphe.features.gps_spatial_semantics import (
    add_gps_spatial_semantics,
)
from raphe.features.gps_anchor_episodes import (
    build_anchor_compatible_location_episodes,
)
from raphe.features.gps_place_candidates import (
    build_place_candidates,
)
from raphe.features.gps_location_features import (
    compute_daily_location_features,
)


# ============================================================
# FROZEN DESIGN
# ============================================================

MAX_GAP_MS = 60_000
BIN_SIZE_MS = 5 * 60_000
COMPATIBILITY_MULTIPLIER = 2.0

# Frozen from the intermediate-geometry design audit.
BASE_SEED = 20261021

ALLOWED_COVERAGES = {
    0.90,
    0.75,
    0.50,
    0.25,
}

ALLOWED_RUN_COUNTS = {
    2,
    4,
    8,
    16,
}

ALLOWED_REPLICATES = set(
    range(5)
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "/shared/Digital Clinic/"
    "raphe_outputs/digital_clinic/"
    "feature_reliability/"
    "gps_observability"
)

RAW_PATH = (
    ROOT
    / "gps_reliability_holdout_raw_points.parquet"
)

REFERENCE_PATH = (
    ROOT
    / "gps_reliability_holdout_reference_location_features.parquet"
)

RUN_DIR = (
    ROOT
    / "reliability_holdout_intermediate_missingness_runs"
)

RUN_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DETERMINISTIC SEED
# ============================================================

def stable_seed(
    *parts,
) -> int:
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
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--coverage",
    type=float,
    required=True,
)

parser.add_argument(
    "--n-runs",
    type=int,
    required=True,
)

parser.add_argument(
    "--replicate",
    type=int,
    required=True,
)

args = parser.parse_args()


coverage = float(
    args.coverage
)

n_runs = int(
    args.n_runs
)

replicate = int(
    args.replicate
)


if coverage not in ALLOWED_COVERAGES:
    raise ValueError(
        "coverage must be one of "
        "0.90, 0.75, 0.50, 0.25"
    )


if n_runs not in ALLOWED_RUN_COUNTS:
    raise ValueError(
        "n-runs must be one of "
        "2, 4, 8, 16"
    )


if replicate not in ALLOWED_REPLICATES:
    raise ValueError(
        "replicate must be 0, 1, 2, 3, or 4"
    )


coverage_label = (
    str(
        int(
            round(
                coverage
                * 100
            )
        )
    )
    .zfill(3)
)


OUT_PATH = (
    RUN_DIR
    / (
        f"coverage_{coverage_label}_"
        f"k{n_runs:02d}_"
        f"rep_{replicate:02d}.parquet"
    )
)


print("=" * 80)
print("RAPHE INDEPENDENT HOLDOUT INTERMEDIATE GPS MISSINGNESS CONDITION")
print("=" * 80)

print(
    "Coverage:",
    coverage,
)

print(
    "Missing runs:",
    n_runs,
)

print(
    "Replicate:",
    replicate,
)

print(
    "Frozen compatibility multiplier:",
    COMPATIBILITY_MULTIPLIER,
)

print(
    "Output:",
    OUT_PATH,
)


# ============================================================
# LOAD
# ============================================================

raw = pd.read_parquet(
    RAW_PATH
)

reference = pd.read_parquet(
    REFERENCE_PATH
)


raw["date"] = (
    raw["date"]
    .astype(str)
)

reference["date"] = (
    reference["date"]
    .astype(str)
)


if (
    reference[
        "compatibility_multiplier"
    ]
    .nunique()
    != 1
    or not np.isclose(
        reference[
            "compatibility_multiplier"
        ].iloc[0],
        COMPATIBILITY_MULTIPLIER,
    )
):
    raise RuntimeError(
        "Reference dataset does not contain exactly "
        "the frozen compatibility multiplier 2.0."
    )


if (
    raw[
        "participant_id"
    ].nunique()
    != 16
):
    raise RuntimeError(
        "Expected 16 validation participants "
        "in raw cache."
    )


raw_day_keys = (
    raw[
        [
            "participant_id",
            "date",
        ]
    ]
    .drop_duplicates()
)


if len(raw_day_keys) != 112:
    raise RuntimeError(
        f"Expected 112 validation participant-days, "
        f"found {len(raw_day_keys)}."
    )


# ============================================================
# DEGRADE RAW GPS + REBUILD MICROSEGMENTS
# ============================================================

participant_microsegments = {}

day_metadata = {}


for (
    participant_id,
    uid,
), participant_raw in raw.groupby(
    [
        "participant_id",
        "mindlamp_uid",
    ],
    sort=True,
):

    microsegment_frames = []


    for date, day_raw in participant_raw.groupby(
        "date",
        sort=True,
    ):

        start_values = (
            day_raw[
                "day_start_ms"
            ]
            .dropna()
            .unique()
        )

        end_values = (
            day_raw[
                "day_end_ms"
            ]
            .dropna()
            .unique()
        )

        quartile_values = (
            day_raw[
                "sampling_density_quartile"
            ]
            .dropna()
            .unique()
        )


        if len(start_values) != 1:
            raise RuntimeError(
                f"{participant_id} {date}: "
                "expected one day_start_ms."
            )


        if len(end_values) != 1:
            raise RuntimeError(
                f"{participant_id} {date}: "
                "expected one day_end_ms."
            )


        if len(quartile_values) != 1:
            raise RuntimeError(
                f"{participant_id} {date}: "
                "expected one sampling density quartile."
            )


        start_ms = int(
            start_values[0]
        )

        end_ms = int(
            end_values[0]
        )


        if (
            end_ms
            - start_ms
            != 86_400_000
        ):
            raise RuntimeError(
                f"{participant_id} {date}: "
                "expected a 24-hour validation window."
            )


        seed = stable_seed(
            BASE_SEED,
            "reliability_holdout_intermediate_missingness",
            participant_id,
            date,
            coverage,
            n_runs,
            replicate,
        )


        missing_mask, mask_info = (
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
                missing_mask,
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
                f"{participant_id} {date}: "
                "reconstructed run count does not "
                "match requested design."
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
                f"{participant_id} {date}: "
                "mask coverage mismatch."
            )


        degraded_raw = (
            apply_missing_bin_mask_to_points(
                day_raw,
                window_start_ms=start_ms,
                window_end_ms=end_ms,
                bin_size_ms=BIN_SIZE_MS,
                missing_mask=missing_mask,
            )
        )


        intervals = build_gps_intervals(
            degraded_raw[
                [
                    "timestamp",
                    "latitude",
                    "longitude",
                    "accuracy",
                ]
            ],
            window_start_ms=start_ms,
            window_end_ms=end_ms,
            max_gap_ms=MAX_GAP_MS,
        )


        supported = (
            intervals[
                intervals[
                    "supported"
                ]
            ]
            .copy()
        )


        supported_duration_ms = float(
            supported[
                "duration_ms"
            ].sum()
        )


        window_duration_ms = float(
            end_ms
            - start_ms
        )


        raw_fraction_retained = (
            len(degraded_raw)
            / len(day_raw)
            if len(day_raw)
            else np.nan
        )


        day_metadata[
            (
                participant_id,
                str(date),
            )
        ] = {
            "mindlamp_uid":
                uid,

            "sampling_density_quartile":
                str(
                    quartile_values[0]
                ),

            "mask_seed":
                seed,

            "planned_coverage":
                float(
                    mask_info[
                        "planned_coverage"
                    ]
                ),

            "n_runs_design":
                n_runs,

            "missing_run_count":
                int(
                    structure[
                        "missing_run_count"
                    ]
                ),

            "longest_missing_bins":
                int(
                    structure[
                        "longest_missing_bins"
                    ]
                ),

            "longest_missing_ms":
                int(
                    structure[
                        "longest_missing_ms"
                    ]
                ),

            "longest_missing_fraction_of_missing":
                float(
                    structure[
                        "longest_missing_fraction_of_missing"
                    ]
                ),

            "missing_run_concentration":
                float(
                    structure[
                        "missing_run_concentration"
                    ]
                ),

            "missing_fragmentation":
                float(
                    structure[
                        "missing_fragmentation"
                    ]
                ),

            "raw_rows_original":
                len(
                    day_raw
                ),

            "raw_rows_retained":
                len(
                    degraded_raw
                ),

            "raw_fraction_retained":
                raw_fraction_retained,

            "supported_duration_ms":
                supported_duration_ms,

            "supported_fraction_of_day":
                (
                    supported_duration_ms
                    / window_duration_ms
                    if window_duration_ms > 0
                    else np.nan
                ),

            "n_supported_edges":
                len(
                    supported
                ),
        }


        if supported.empty:
            continue


        spatial = (
            add_gps_spatial_semantics(
                supported
            )
        )


        microsegments = (
            build_anchor_compatible_location_episodes(
                spatial
            )
        )


        if microsegments.empty:
            continue


        microsegments[
            "participant_id"
        ] = participant_id

        microsegments[
            "mindlamp_uid"
        ] = uid

        microsegments[
            "date"
        ] = str(
            date
        )


        microsegment_frames.append(
            microsegments
        )


    if microsegment_frames:

        participant_segments = pd.concat(
            microsegment_frames,
            ignore_index=True,
        )

    else:

        participant_segments = pd.DataFrame()


    participant_microsegments[
        participant_id
    ] = participant_segments


    participant_meta = [
        value
        for (
            pid,
            _,
        ), value
        in day_metadata.items()
        if pid == participant_id
    ]


    median_supported = float(
        np.median(
            [
                x[
                    "supported_fraction_of_day"
                ]
                for x in participant_meta
            ]
        )
    )


    print(
        f"{participant_id}: "
        f"days={len(participant_meta)} "
        f"microsegments={len(participant_segments):,} "
        f"median_supported={median_supported:.4f}"
    )


# ============================================================
# REBUILD FROZEN PARTICIPANT-LEVEL PLACES + FEATURES
# ============================================================

feature_rows = []


for participant_id, microsegments in (
    participant_microsegments.items()
):

    dates = sorted(
        raw.loc[
            raw[
                "participant_id"
            ]
            == participant_id,
            "date",
        ]
        .unique()
        .tolist()
    )


    if microsegments.empty:

        assignments = pd.DataFrame(
            columns=[
                "date",
                "place_candidate_id",
                "duration_ms",
            ]
        )

    else:

        assignments, _ = (
            build_place_candidates(
                microsegments,
                compatibility_multiplier=(
                    COMPATIBILITY_MULTIPLIER
                ),
            )
        )


    for date in dates:

        meta = day_metadata[
            (
                participant_id,
                str(date),
            )
        ]


        if assignments.empty:

            day_assignments = pd.DataFrame(
                columns=[
                    "place_candidate_id",
                    "duration_ms",
                ]
            )

        else:

            day_assignments = assignments[
                assignments[
                    "date"
                ].astype(str)
                == str(date)
            ]


        result = (
            compute_daily_location_features(
                day_assignments,
                supported_duration_ms=(
                    meta[
                        "supported_duration_ms"
                    ]
                ),
            )
        )


        result.update({
            "participant_id":
                participant_id,

            "mindlamp_uid":
                meta[
                    "mindlamp_uid"
                ],

            "sampling_density_quartile":
                meta[
                    "sampling_density_quartile"
                ],

            "date":
                str(
                    date
                ),

            "compatibility_multiplier":
                COMPATIBILITY_MULTIPLIER,

            "target_coverage":
                coverage,

            "missingness_design":
                f"k{n_runs}",

            "n_runs_design":
                n_runs,

            "replicate":
                replicate,

            "mask_seed":
                meta[
                    "mask_seed"
                ],

            "planned_coverage":
                meta[
                    "planned_coverage"
                ],

            "missing_run_count":
                meta[
                    "missing_run_count"
                ],

            "longest_missing_bins":
                meta[
                    "longest_missing_bins"
                ],

            "longest_missing_ms":
                meta[
                    "longest_missing_ms"
                ],

            "longest_missing_fraction_of_missing":
                meta[
                    "longest_missing_fraction_of_missing"
                ],

            "missing_run_concentration":
                meta[
                    "missing_run_concentration"
                ],

            "missing_fragmentation":
                meta[
                    "missing_fragmentation"
                ],

            "raw_fraction_retained":
                meta[
                    "raw_fraction_retained"
                ],

            "supported_fraction_of_day":
                meta[
                    "supported_fraction_of_day"
                ],

            "n_raw_original":
                meta[
                    "raw_rows_original"
                ],

            "n_raw_retained":
                meta[
                    "raw_rows_retained"
                ],

            "n_supported_edges":
                meta[
                    "n_supported_edges"
                ],
        })


        feature_rows.append(
            result
        )


degraded = pd.DataFrame(
    feature_rows
)


if len(degraded) != 112:
    raise RuntimeError(
        f"Expected 112 degraded feature rows, "
        f"found {len(degraded)}."
    )


# ============================================================
# JOIN FROZEN COMPLETE-DATA REFERENCE
# ============================================================

reference_keep = (
    reference[
        [
            "participant_id",
            "date",
            "location_entropy_nats",
            "effective_number_places",
            "dominant_place_fraction",
            "place_time_fraction_of_supported",
        ]
    ]
    .rename(
        columns={
            "location_entropy_nats":
                "reference_entropy_nats",

            "effective_number_places":
                "reference_effective_places",

            "dominant_place_fraction":
                "reference_dominant_fraction",

            "place_time_fraction_of_supported":
                "reference_place_time_fraction",
        }
    )
)


comparison = degraded.merge(
    reference_keep,
    on=[
        "participant_id",
        "date",
    ],
    how="left",
    validate="one_to_one",
)


comparison[
    "entropy_error"
] = (
    comparison[
        "location_entropy_nats"
    ]
    - comparison[
        "reference_entropy_nats"
    ]
)

comparison[
    "abs_entropy_error"
] = comparison[
    "entropy_error"
].abs()


comparison[
    "effective_places_error"
] = (
    comparison[
        "effective_number_places"
    ]
    - comparison[
        "reference_effective_places"
    ]
)

comparison[
    "abs_effective_places_error"
] = comparison[
    "effective_places_error"
].abs()


comparison[
    "dominant_fraction_error"
] = (
    comparison[
        "dominant_place_fraction"
    ]
    - comparison[
        "reference_dominant_fraction"
    ]
)

comparison[
    "abs_dominant_fraction_error"
] = comparison[
    "dominant_fraction_error"
].abs()


comparison[
    "place_time_fraction_error"
] = (
    comparison[
        "place_time_fraction_of_supported"
    ]
    - comparison[
        "reference_place_time_fraction"
    ]
)


missing_reference = int(
    comparison[
        "reference_entropy_nats"
    ]
    .isna()
    .sum()
)


if missing_reference:
    raise RuntimeError(
        f"Missing reference matches: "
        f"{missing_reference}"
    )


# ============================================================
# FINAL INTEGRITY
# ============================================================

if (
    comparison[
        "participant_id"
    ].nunique()
    != 16
):
    raise RuntimeError(
        "Expected 16 participants."
    )


if not (
    comparison[
        "missing_run_count"
    ]
    == n_runs
).all():
    raise RuntimeError(
        "Final output contains an unexpected "
        "missing-run count."
    )


if not np.allclose(
    comparison[
        "planned_coverage"
    ],
    comparison[
        "target_coverage"
    ],
    atol=0.004,
    rtol=0,
):
    raise RuntimeError(
        "Planned coverage differs unexpectedly "
        "from target coverage."
    )


comparison.to_parquet(
    OUT_PATH,
    index=False,
)


# ============================================================
# COMPACT REPORT
# ============================================================

median_supported = float(
    comparison[
        "supported_fraction_of_day"
    ].median()
)

median_entropy_error = float(
    comparison[
        "abs_entropy_error"
    ].median()
)

median_dominant_error = float(
    comparison[
        "abs_dominant_fraction_error"
    ].median()
)

median_effective_error = float(
    comparison[
        "abs_effective_places_error"
    ].median()
)

median_longest_gap_hours = float(
    comparison[
        "longest_missing_ms"
    ].median()
    / 3_600_000
)

median_concentration = float(
    comparison[
        "missing_run_concentration"
    ].median()
)


print()
print("=" * 80)
print("RUN COMPLETE")
print("=" * 80)

print(
    "Rows:",
    len(
        comparison
    ),
)

print(
    "Participants:",
    comparison[
        "participant_id"
    ].nunique(),
)

print(
    "Missing reference matches:",
    missing_reference,
)

print(
    "Missing runs:",
    n_runs,
)

print(
    "Median longest missing gap (hours):",
    f"{median_longest_gap_hours:.4f}",
)

print(
    "Median missing-run concentration:",
    f"{median_concentration:.4f}",
)

print(
    "Median supported coverage:",
    f"{median_supported:.4f}",
)

print(
    "Median absolute entropy error:",
    f"{median_entropy_error:.4f}",
)

print(
    "Median absolute dominant-place error:",
    f"{median_dominant_error:.4f}",
)

print(
    "Median absolute effective-places error:",
    f"{median_effective_error:.4f}",
)

print(
    "Output:",
    OUT_PATH,
)

print()
print("=" * 80)
print("DONE")
print("=" * 80)
