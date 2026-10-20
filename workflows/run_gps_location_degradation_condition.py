"""Run one RAPHE GPS location degradation condition.

Example
-------
python workflows/run_gps_location_degradation_condition.py \
    --coverage 0.75 \
    --pattern distributed \
    --replicate 0

Missingness is imposed on raw GPS and the complete RAPHE location pipeline
is rebuilt.

Outputs one parquet file per condition so experiments are resumable.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from raphe.features.gps_degradation import (
    apply_missing_bin_mask_to_points,
    make_missing_bin_mask,
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


DAY_MS = 86_400_000
MAX_GAP_MS = 60_000
BIN_SIZE_MS = 5 * 60_000

BASE_SEED = 20261019

MULTIPLIERS = [
    0.75,
    1.00,
    1.50,
    2.00,
]


ROOT = Path(
    "/shared/Digital Clinic/"
    "raphe_outputs/digital_clinic/"
    "feature_reliability/"
    "gps_observability"
)

RAW_PATH = (
    ROOT
    / "gps_reference_raw_points.parquet"
)

REFERENCE_PATH = (
    ROOT
    / "gps_reference_location_features_rebuilt_from_raw.parquet"
)

RUN_DIR = (
    ROOT
    / "degradation_runs"
)

RUN_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def stable_seed(
    *parts,
) -> int:

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


parser = argparse.ArgumentParser()

parser.add_argument(
    "--coverage",
    type=float,
    required=True,
)

parser.add_argument(
    "--pattern",
    choices=[
        "distributed",
        "contiguous",
    ],
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

pattern = str(
    args.pattern
)

replicate = int(
    args.replicate
)


if coverage not in {
    0.90,
    0.75,
    0.50,
    0.25,
}:
    raise ValueError(
        "coverage must be one of "
        "0.90, 0.75, 0.50, 0.25"
    )


coverage_label = str(
    int(
        round(
            coverage * 100
        )
    )
).zfill(3)


OUT_PATH = (
    RUN_DIR
    / (
        f"coverage_{coverage_label}_"
        f"{pattern}_"
        f"rep_{replicate:02d}.parquet"
    )
)


print("=" * 80)
print("RAPHE GPS LOCATION DEGRADATION CONDITION")
print("=" * 80)

print(
    "Coverage:",
    coverage,
)

print(
    "Pattern:",
    pattern,
)

print(
    "Replicate:",
    replicate,
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


# ============================================================
# REBUILD DEGRADED MICROSEGMENTS
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

    print()
    print(
        participant_id,
        uid,
    )


    microsegment_frames = []


    for date, day_raw in participant_raw.groupby(
        "date",
        sort=True,
    ):

        day_start_values = (
            day_raw[
                "day_start_ms"
            ]
            .dropna()
            .unique()
        )


        if len(
            day_start_values
        ) != 1:

            raise RuntimeError(
                f"{participant_id} {date}: "
                "expected one day_start_ms"
            )


        day_start_ms = int(
            day_start_values[0]
        )

        day_end_ms = (
            day_start_ms
            + DAY_MS
        )


        seed = stable_seed(
            BASE_SEED,
            participant_id,
            date,
            coverage,
            pattern,
            replicate,
        )


        missing_mask, mask_info = (
            make_missing_bin_mask(
                window_start_ms=(
                    day_start_ms
                ),
                window_end_ms=(
                    day_end_ms
                ),
                target_coverage=(
                    coverage
                ),
                pattern=pattern,
                bin_size_ms=(
                    BIN_SIZE_MS
                ),
                seed=seed,
            )
        )


        degraded_raw = (
            apply_missing_bin_mask_to_points(
                day_raw,
                window_start_ms=(
                    day_start_ms
                ),
                window_end_ms=(
                    day_end_ms
                ),
                bin_size_ms=(
                    BIN_SIZE_MS
                ),
                missing_mask=(
                    missing_mask
                ),
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
            window_start_ms=(
                day_start_ms
            ),
            window_end_ms=(
                day_end_ms
            ),
            max_gap_ms=(
                MAX_GAP_MS
            ),
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


        raw_fraction_retained = (
            len(
                degraded_raw
            )
            / len(
                day_raw
            )
            if len(
                day_raw
            )
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

            "planned_coverage":
                float(
                    mask_info[
                        "planned_coverage"
                    ]
                ),

            "raw_fraction_retained":
                raw_fraction_retained,

            "supported_duration_ms":
                supported_duration_ms,

            "supported_fraction_of_day":
                (
                    supported_duration_ms
                    / DAY_MS
                ),

            "n_raw_original":
                len(
                    day_raw
                ),

            "n_raw_retained":
                len(
                    degraded_raw
                ),

            "n_supported_edges":
                len(
                    supported
                ),
        }


        if supported.empty:
            continue


        supported = (
            add_gps_spatial_semantics(
                supported
            )
        )


        microsegments = (
            build_anchor_compatible_location_episodes(
                supported
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


    relevant_meta = [
        value
        for (
            pid,
            _
        ), value
        in day_metadata.items()
        if pid == participant_id
    ]


    print(
        "  days:",
        len(
            relevant_meta
        ),
    )

    print(
        "  microsegments:",
        f"{len(participant_segments):,}",
    )

    median_supported_coverage = float(
        np.median(
            [
                x["supported_fraction_of_day"]
                for x in relevant_meta
            ]
        )
    )

    print(
        "  median supported coverage:",
        f"{median_supported_coverage:.4f}",
    )


# ============================================================
# BUILD PLACES + DAILY FEATURES
# ============================================================

rows = []


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


    for multiplier in MULTIPLIERS:

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
                        multiplier
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
                "compatibility_multiplier":
                    float(
                        multiplier
                    ),

                "participant_id":
                    participant_id,

                "mindlamp_uid":
                    meta[
                        "mindlamp_uid"
                    ],

                "date":
                    str(
                        date
                    ),

                "target_coverage":
                    coverage,

                "missingness_pattern":
                    pattern,

                "replicate":
                    replicate,

                "planned_coverage":
                    meta[
                        "planned_coverage"
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
                        "n_raw_original"
                    ],

                "n_raw_retained":
                    meta[
                        "n_raw_retained"
                    ],

                "n_supported_edges":
                    meta[
                        "n_supported_edges"
                    ],
            })


            rows.append(
                result
            )


degraded = pd.DataFrame(
    rows
)


# ============================================================
# REFERENCE COMPARISON
# ============================================================

reference_keep = (
    reference[
        [
            "compatibility_multiplier",
            "participant_id",
            "date",
            "location_entropy_nats",
            "dominant_place_fraction",
            "effective_number_places",
        ]
    ]
    .rename(
        columns={
            "location_entropy_nats":
                "reference_entropy_nats",

            "dominant_place_fraction":
                "reference_dominant_fraction",

            "effective_number_places":
                "reference_effective_places",
        }
    )
)


comparison = degraded.merge(
    reference_keep,
    on=[
        "compatibility_multiplier",
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


comparison.to_parquet(
    OUT_PATH,
    index=False,
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
    "Missing reference matches:",
    int(
        comparison[
            "reference_entropy_nats"
        ]
        .isna()
        .sum()
    ),
)

median_supported_coverage = float(
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

print(
    "Median supported coverage:",
    f"{median_supported_coverage:.4f}",
)

print(
    "Median entropy absolute error:",
    f"{median_entropy_error:.4f}",
)

print(
    "Median dominant-place absolute error:",
    f"{median_dominant_error:.4f}",
)

print()
print(
    OUT_PATH
)

print()
print("=" * 80)
print("DONE")
print("=" * 80)
