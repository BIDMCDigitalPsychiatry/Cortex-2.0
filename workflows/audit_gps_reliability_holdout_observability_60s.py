"""Audit true 60-second Cortex 2.0 GPS observability for held-out candidates.

The existing 10-minute daily DQ was used only to prefilter candidate days.
This workflow goes back to raw GPS timestamps and computes Cortex 2.0 temporal
observability using the already-tested 60-second continuity policy.

No feature values are computed here.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cortex2_engine.features.gps_observability import (
    compute_gps_temporal_observability,
)


MAX_GAP_MS = 60_000
DAY_MS = 86_400_000


ROOT = Path(
    "/shared/Digital Clinic/"
    "cortex2_outputs/digital_clinic/"
    "feature_reliability/"
    "gps_observability"
)

CANDIDATE_PATH = (
    ROOT
    / "gps_reliability_holdout_candidate_days.parquet"
)

INVENTORY_PATH = Path(
    "/shared/Digital Clinic/"
    "cortex2_outputs/digital_clinic/"
    "inventory/source_files/gps.parquet"
)

DAILY_OUT = (
    ROOT
    / "gps_reliability_holdout_candidate_observability_60s.parquet"
)

PARTICIPANT_OUT = (
    ROOT
    / "gps_reliability_holdout_candidate_observability_60s_participants.csv"
)


# ============================================================
# LOAD
# ============================================================

candidates = pd.read_parquet(
    CANDIDATE_PATH
)

inventory = pd.read_parquet(
    INVENTORY_PATH
)


candidates["date"] = (
    candidates["date"]
    .astype(str)
)


print("=" * 80)
print("Cortex 2.0 HELD-OUT GPS 60-SECOND OBSERVABILITY AUDIT")
print("=" * 80)

print(
    "Candidate participants:",
    candidates["participant_id"].nunique(),
)

print(
    "Candidate days:",
    len(candidates),
)


# ============================================================
# PROCESS PARTICIPANTS
# ============================================================

daily_rows = []


for (
    participant_id,
    uid,
), participant_days in candidates.groupby(
    [
        "participant_id",
        "source_participant_id",
    ],
    sort=True,
):

    participant_days = (
        participant_days
        .sort_values(
            "start_ms"
        )
        .copy()
    )


    overall_start = int(
        participant_days[
            "start_ms"
        ].min()
    )

    overall_end = int(
        participant_days[
            "end_ms"
        ].max()
    )


    source_files = inventory[
        (
            inventory[
                "source_participant_id"
            ].astype(str)
            == str(uid)
        )
        & (
            pd.to_numeric(
                inventory[
                    "max_timestamp_ms"
                ],
                errors="coerce",
            )
            >= overall_start
        )
        & (
            pd.to_numeric(
                inventory[
                    "min_timestamp_ms"
                ],
                errors="coerce",
            )
            < overall_end
        )
    ].copy()


    timestamp_chunks = []

    failed_files = []


    for _, source in source_files.iterrows():

        path = Path(
            source[
                "source_file"
            ]
        )


        try:

            x = pd.read_csv(
                path,
                usecols=[
                    "timestamp",
                ],
            )

            ts = pd.to_numeric(
                x[
                    "timestamp"
                ],
                errors="coerce",
            )

            ts = ts[
                ts.notna()
            ].astype(
                np.int64
            )


            ts = ts[
                (ts >= overall_start)
                & (ts < overall_end)
            ]


            if len(ts):

                timestamp_chunks.append(
                    ts.to_numpy(
                        dtype=np.int64
                    )
                )


        except Exception as exc:

            failed_files.append(
                (
                    str(path),
                    repr(exc),
                )
            )


    if timestamp_chunks:

        participant_timestamps = np.concatenate(
            timestamp_chunks
        )

    else:

        participant_timestamps = np.array(
            [],
            dtype=np.int64,
        )


    print()
    print("-" * 80)

    print(
        participant_id,
        uid,
        f"days={len(participant_days)}",
        f"files={len(source_files)}",
        f"failed={len(failed_files)}",
    )


    participant_observabilities = []


    for _, day in participant_days.iterrows():

        start_ms = int(
            day[
                "start_ms"
            ]
        )

        end_ms = int(
            day[
                "end_ms"
            ]
        )


        in_day = participant_timestamps[
            (participant_timestamps >= start_ms)
            & (participant_timestamps < end_ms)
        ]


        result = (
            compute_gps_temporal_observability(
                in_day,
                window_start_ms=start_ms,
                window_end_ms=end_ms,
                max_gap_ms=MAX_GAP_MS,
            )
        )


        if "observability" not in result:

            raise RuntimeError(
                "GPS observability primitive did not "
                "return 'observability'. Returned keys: "
                f"{list(result.keys())}"
            )


        observability = float(
            result[
                "observability"
            ]
        )


        participant_observabilities.append(
            observability
        )


        row = {
            "participant_id":
                participant_id,

            "source_participant_id":
                uid,

            "episode_id":
                day[
                    "episode_id"
                ],

            "date":
                str(
                    day[
                        "date"
                    ]
                ),

            "sampling_density_quartile":
                day[
                    "sampling_density_quartile"
                ],

            "participant_median_n_observations":
                day[
                    "participant_median_n_observations"
                ],

            "coarse_n_observations":
                int(
                    day[
                        "n_observations"
                    ]
                ),

            "start_ms":
                start_ms,

            "end_ms":
                end_ms,

            "window_hours":
                (
                    end_ms
                    - start_ms
                )
                / 3_600_000.0,

            "raw_timestamp_rows":
                int(
                    len(
                        in_day
                    )
                ),

            "unique_timestamps":
                int(
                    np.unique(
                        in_day
                    ).size
                ),

            "source_files_considered":
                int(
                    len(
                        source_files
                    )
                ),

            "failed_source_files":
                int(
                    len(
                        failed_files
                    )
                ),

            "max_gap_ms":
                MAX_GAP_MS,
        }


        for key, value in result.items():

            row[
                key
            ] = value


        daily_rows.append(
            row
        )


    print(
        "  median observability:",
        f"{np.median(participant_observabilities):.4f}",
    )

    print(
        "  >=0.99 days:",
        int(
            np.sum(
                np.asarray(
                    participant_observabilities
                )
                >= 0.99
            )
        ),
    )


daily = pd.DataFrame(
    daily_rows
)


if len(daily) != len(candidates):

    raise RuntimeError(
        f"Expected {len(candidates)} daily rows, "
        f"got {len(daily)}."
    )


daily.to_parquet(
    DAILY_OUT,
    index=False,
)


# ============================================================
# PARTICIPANT SUMMARY
# ============================================================

participant_rows = []


for (
    quartile,
    participant_id,
), x in daily.groupby(
    [
        "sampling_density_quartile",
        "participant_id",
    ],
    sort=True,
):

    participant_rows.append({
        "sampling_density_quartile":
            quartile,

        "participant_id":
            participant_id,

        "candidate_days":
            len(x),

        "median_observability":
            x[
                "observability"
            ].median(),

        "minimum_observability":
            x[
                "observability"
            ].min(),

        "q25_observability":
            x[
                "observability"
            ].quantile(
                0.25
            ),

        "days_ge_099":
            int(
                (
                    x[
                        "observability"
                    ]
                    >= 0.99
                ).sum()
            ),

        "days_ge_095":
            int(
                (
                    x[
                        "observability"
                    ]
                    >= 0.95
                ).sum()
            ),

        "days_ge_090":
            int(
                (
                    x[
                        "observability"
                    ]
                    >= 0.90
                ).sum()
            ),

        "median_unique_timestamps":
            x[
                "unique_timestamps"
            ].median(),

        "failed_source_files":
            int(
                x[
                    "failed_source_files"
                ].max()
            ),
    })


participant_summary = pd.DataFrame(
    participant_rows
)


participant_summary.to_csv(
    PARTICIPANT_OUT,
    index=False,
)


# ============================================================
# OVERALL DISTRIBUTION
# ============================================================

print()
print("=" * 80)
print("60-SECOND OBSERVABILITY DISTRIBUTION")
print("=" * 80)

quantiles = [
    0.00,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    1.00,
]


for q in quantiles:

    value = daily[
        "observability"
    ].quantile(
        q
    )

    print(
        f"q{int(q * 100):02d}: "
        f"{value:.4f}"
    )


# ============================================================
# REFERENCE-DAY ELIGIBILITY COUNTS
# ============================================================

print()
print("=" * 80)
print("REFERENCE-DAY ELIGIBILITY")
print("=" * 80)

for threshold in [
    0.99,
    0.95,
    0.90,
    0.75,
]:

    x = daily[
        daily[
            "observability"
        ]
        >= threshold
    ]


    counts = (
        x.groupby(
            "participant_id"
        )
        .size()
    )


    participants_ge_7 = int(
        (
            counts
            >= 7
        ).sum()
    )


    print(
        f">= {threshold:.2f}: "
        f"{len(x)} days, "
        f"{x['participant_id'].nunique()} participants, "
        f"{participants_ge_7} participants with >=7 days"
    )


# ============================================================
# SAMPLING-STRATUM SUMMARY
# ============================================================

print()
print("=" * 80)
print("OBSERVABILITY BY SAMPLING-DENSITY QUARTILE")
print("=" * 80)

quartile_rows = []


for quartile, x in daily.groupby(
    "sampling_density_quartile",
    sort=True,
):

    participant_counts_099 = (
        x[
            x[
                "observability"
            ]
            >= 0.99
        ]
        .groupby(
            "participant_id"
        )
        .size()
    )


    quartile_rows.append({
        "sampling_density_quartile":
            quartile,

        "candidate_days":
            len(x),

        "participants":
            x[
                "participant_id"
            ].nunique(),

        "median_observability":
            x[
                "observability"
            ].median(),

        "q25_observability":
            x[
                "observability"
            ].quantile(
                0.25
            ),

        "days_ge_099":
            int(
                (
                    x[
                        "observability"
                    ]
                    >= 0.99
                ).sum()
            ),

        "participants_with_ge7_days_ge099":
            int(
                (
                    participant_counts_099
                    >= 7
                ).sum()
            ),
    })


quartile_summary = pd.DataFrame(
    quartile_rows
)


print(
    quartile_summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# ============================================================
# PARTICIPANT SUMMARY
# ============================================================

print()
print("=" * 80)
print("PARTICIPANT OBSERVABILITY SUMMARY")
print("=" * 80)

print(
    participant_summary[
        [
            "sampling_density_quartile",
            "participant_id",
            "candidate_days",
            "median_observability",
            "minimum_observability",
            "days_ge_099",
            "days_ge_095",
            "median_unique_timestamps",
            "failed_source_files",
        ]
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
# WINDOW CHECK
# ============================================================

print()
print("=" * 80)
print("WINDOW CHECK")
print("=" * 80)

print(
    "24-hour windows:",
    int(
        np.isclose(
            daily[
                "window_hours"
            ],
            24.0,
        ).sum()
    ),
    "/",
    len(daily),
)

print(
    "Failed source-file reads:",
    int(
        daily[
            "failed_source_files"
        ].max()
    ),
    "(maximum for any participant)",
)


print()
print("=" * 80)
print("OUTPUTS")
print("=" * 80)

print(DAILY_OUT)
print(PARTICIPANT_OUT)

print()
print("=" * 80)
print("DONE")
print("=" * 80)
