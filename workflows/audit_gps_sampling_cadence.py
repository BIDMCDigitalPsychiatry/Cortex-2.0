"""Audit empirical GPS sampling cadence for Cortex 2.0.

This workflow does NOT use Cortex feature functions.

It characterizes the raw temporal structure of GPS observations and evaluates
Cortex 2.0 temporal observability across a sensitivity grid of continuity
tolerances.

The continuity tolerances are NOT proposed reliability thresholds.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cortex2_engine.features.gps_observability import (
    compute_gps_temporal_observability,
)


DAY_MS = 86_400_000

INVENTORY_PATH = Path(
    "/shared/Digital Clinic/"
    "cortex2_outputs/digital_clinic/"
    "inventory/source_files/gps.parquet"
)

DAILY_ROOT = Path(
    "/shared/Digital Clinic/"
    "cortex_longitudinal/data/"
    "benchmark_5_full/"
    "participant_daily"
)

OUT_DIR = Path(
    "/shared/Digital Clinic/"
    "cortex2_outputs/digital_clinic/"
    "feature_reliability/"
    "gps_observability"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DAILY_OUT = (
    OUT_DIR
    / "gps_sampling_cadence_benchmark5_daily.parquet"
)

SENSITIVITY_OUT = (
    OUT_DIR
    / "gps_continuity_sensitivity_benchmark5.csv"
)

GAP_OUT = (
    OUT_DIR
    / "gps_gap_distribution_benchmark5.csv"
)

PARTICIPANT_OUT = (
    OUT_DIR
    / "gps_sampling_cadence_benchmark5_participants.csv"
)


CONTINUITY_MINUTES = [
    1,
    2,
    5,
    10,
    15,
    30,
    60,
]


# ============================================================
# LOAD BENCHMARK DAY WINDOWS
# ============================================================

daily_frames = []

for path in sorted(
    DAILY_ROOT.glob("*.csv")
):
    df = pd.read_csv(
        path
    )

    keep = [
        "participant_id",
        "mindlamp_uid",
        "timestamp",
        "date",
        "gps_data_quality",
    ]

    daily_frames.append(
        df[keep].copy()
    )


if not daily_frames:
    raise RuntimeError(
        f"No benchmark daily files found in {DAILY_ROOT}"
    )


days = pd.concat(
    daily_frames,
    ignore_index=True,
)


days["timestamp"] = pd.to_numeric(
    days["timestamp"],
    errors="coerce",
)

days["gps_data_quality"] = pd.to_numeric(
    days["gps_data_quality"],
    errors="coerce",
)

days = (
    days.dropna(
        subset=[
            "mindlamp_uid",
            "timestamp",
        ]
    )
    .copy()
)

days["timestamp"] = (
    days["timestamp"]
    .astype("int64")
)


# ============================================================
# LOAD SOURCE INVENTORY
# ============================================================

inventory = pd.read_parquet(
    INVENTORY_PATH
)

inventory = (
    inventory[
        inventory["scan_status"]
        == "ok"
    ]
    .copy()
)


benchmark_uids = set(
    days["mindlamp_uid"]
    .astype(str)
)


inventory = inventory[
    inventory["source_participant_id"]
    .astype(str)
    .isin(
        benchmark_uids
    )
].copy()


print("=" * 80)
print("Cortex 2.0 GPS SAMPLING-CADENCE AUDIT")
print("=" * 80)

print(
    "Benchmark participants:",
    days["mindlamp_uid"].nunique(),
)

print(
    "Benchmark participant-days:",
    len(days),
)

print(
    "GPS source files:",
    len(inventory),
)


# ============================================================
# PROCESS PARTICIPANT BY PARTICIPANT
# ============================================================

daily_rows = []
participant_rows = []
all_gap_arrays = []


for uid, day_group in days.groupby(
    "mindlamp_uid",
    sort=True,
):

    participant_id = (
        day_group[
            "participant_id"
        ]
        .iloc[0]
    )

    inv = inventory[
        inventory[
            "source_participant_id"
        ]
        .astype(str)
        == str(uid)
    ].copy()


    print()
    print("-" * 80)

    print(
        participant_id,
        uid,
        f"days={len(day_group)}",
        f"files={len(inv)}",
    )


    timestamp_parts = []

    n_rows_read = 0
    n_invalid = 0
    n_read_failures = 0


    for source_file in sorted(
        inv["source_file"]
        .astype(str)
        .unique()
    ):

        try:
            x = pd.read_csv(
                source_file,
                usecols=lambda c:
                    c == "timestamp",
            )

        except Exception as e:
            n_read_failures += 1

            print(
                "  read failure:",
                Path(source_file).name,
                type(e).__name__,
            )

            continue


        if "timestamp" not in x.columns:
            n_read_failures += 1
            continue


        n_rows_read += len(x)

        ts = pd.to_numeric(
            x["timestamp"],
            errors="coerce",
        )

        n_invalid += int(
            ts.isna().sum()
        )

        ts = (
            ts.dropna()
            .astype("int64")
            .to_numpy()
        )

        if len(ts):
            timestamp_parts.append(
                ts
            )


    if timestamp_parts:
        raw_timestamps = np.concatenate(
            timestamp_parts
        )
    else:
        raw_timestamps = np.array(
            [],
            dtype=np.int64,
        )


    n_valid_before_dedup = len(
        raw_timestamps
    )

    unique_timestamps = np.unique(
        raw_timestamps
    )

    n_duplicates = (
        n_valid_before_dedup
        - len(
            unique_timestamps
        )
    )


    participant_gap_arrays = []


    for day in day_group.itertuples(
        index=False
    ):

        start = int(
            day.timestamp
        )

        end = (
            start
            + DAY_MS
        )


        left = np.searchsorted(
            unique_timestamps,
            start,
            side="left",
        )

        right = np.searchsorted(
            unique_timestamps,
            end,
            side="left",
        )


        day_ts = unique_timestamps[
            left:right
        ]


        if len(day_ts) >= 2:
            gaps = np.diff(
                day_ts
            )

            gaps = gaps[
                gaps > 0
            ]

        else:
            gaps = np.array(
                [],
                dtype=np.int64,
            )


        if len(gaps):
            participant_gap_arrays.append(
                gaps
            )

            all_gap_arrays.append(
                gaps
            )


        row = {
            "participant_id":
                participant_id,

            "mindlamp_uid":
                uid,

            "date":
                day.date,

            "timestamp":
                start,

            "cortex_gps_data_quality":
                day.gps_data_quality,

            "n_gps_observations":
                len(day_ts),

            "n_inter_observation_gaps":
                len(gaps),

            "median_gap_ms":
                (
                    float(
                        np.median(
                            gaps
                        )
                    )
                    if len(gaps)
                    else np.nan
                ),

            "p90_gap_ms":
                (
                    float(
                        np.quantile(
                            gaps,
                            0.90,
                        )
                    )
                    if len(gaps)
                    else np.nan
                ),

            "p99_gap_ms":
                (
                    float(
                        np.quantile(
                            gaps,
                            0.99,
                        )
                    )
                    if len(gaps)
                    else np.nan
                ),
        }


        for minutes in CONTINUITY_MINUTES:

            result = (
                compute_gps_temporal_observability(
                    day_ts,
                    window_start_ms=start,
                    window_end_ms=end,
                    max_gap_ms=(
                        minutes
                        * 60_000
                    ),
                )
            )


            prefix = (
                f"gap_{minutes}m"
            )

            row[
                f"{prefix}_observability"
            ] = result[
                "observability"
            ]

            row[
                f"{prefix}_observed_duration_ms"
            ] = result[
                "observed_duration_ms"
            ]

            row[
                f"{prefix}_longest_unobserved_gap_ms"
            ] = result[
                "longest_unobserved_gap_ms"
            ]

            row[
                f"{prefix}_n_segments"
            ] = result[
                "n_observable_segments"
            ]


        daily_rows.append(
            row
        )


    if participant_gap_arrays:

        participant_gaps = np.concatenate(
            participant_gap_arrays
        )

        participant_rows.append({
            "participant_id":
                participant_id,

            "mindlamp_uid":
                uid,

            "n_source_files":
                len(inv),

            "n_rows_read":
                n_rows_read,

            "n_invalid_timestamps":
                n_invalid,

            "n_unique_timestamps":
                len(
                    unique_timestamps
                ),

            "n_duplicate_timestamps":
                n_duplicates,

            "n_read_failures":
                n_read_failures,

            "n_within_day_gaps":
                len(
                    participant_gaps
                ),

            "median_gap_seconds":
                np.median(
                    participant_gaps
                )
                / 1000,

            "p90_gap_seconds":
                np.quantile(
                    participant_gaps,
                    0.90,
                )
                / 1000,

            "p95_gap_seconds":
                np.quantile(
                    participant_gaps,
                    0.95,
                )
                / 1000,

            "p99_gap_seconds":
                np.quantile(
                    participant_gaps,
                    0.99,
                )
                / 1000,
        })

    else:

        participant_rows.append({
            "participant_id":
                participant_id,

            "mindlamp_uid":
                uid,

            "n_source_files":
                len(inv),

            "n_rows_read":
                n_rows_read,

            "n_invalid_timestamps":
                n_invalid,

            "n_unique_timestamps":
                len(
                    unique_timestamps
                ),

            "n_duplicate_timestamps":
                n_duplicates,

            "n_read_failures":
                n_read_failures,

            "n_within_day_gaps":
                0,

            "median_gap_seconds":
                np.nan,

            "p90_gap_seconds":
                np.nan,

            "p95_gap_seconds":
                np.nan,

            "p99_gap_seconds":
                np.nan,
        })


# ============================================================
# SAVE DAILY RESULTS
# ============================================================

daily_result = pd.DataFrame(
    daily_rows
)

daily_result.to_parquet(
    DAILY_OUT,
    index=False,
)


participant_result = pd.DataFrame(
    participant_rows
)

participant_result.to_csv(
    PARTICIPANT_OUT,
    index=False,
)


# ============================================================
# GLOBAL GAP DISTRIBUTION
# ============================================================

if all_gap_arrays:
    all_gaps = np.concatenate(
        all_gap_arrays
    )
else:
    all_gaps = np.array(
        [],
        dtype=np.int64,
    )


gap_rows = []

for q in [
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
]:

    gap_rows.append({
        "quantile":
            q,

        "gap_seconds":
            (
                np.quantile(
                    all_gaps,
                    q,
                )
                / 1000
                if len(all_gaps)
                else np.nan
            ),

        "gap_minutes":
            (
                np.quantile(
                    all_gaps,
                    q,
                )
                / 60_000
                if len(all_gaps)
                else np.nan
            ),
    })


gap_result = pd.DataFrame(
    gap_rows
)

gap_result.to_csv(
    GAP_OUT,
    index=False,
)


# ============================================================
# CONTINUITY SENSITIVITY
# ============================================================

sensitivity_rows = []


for minutes in CONTINUITY_MINUTES:

    obs_col = (
        f"gap_{minutes}m_observability"
    )

    longest_col = (
        f"gap_{minutes}m_longest_unobserved_gap_ms"
    )


    for subset_name, subset in [
        (
            "all_days",
            daily_result,
        ),
        (
            "cortex_dq_1_days",
            daily_result[
                daily_result[
                    "cortex_gps_data_quality"
                ]
                == 1.0
            ],
        ),
    ]:

        obs = subset[
            obs_col
        ].dropna()


        sensitivity_rows.append({
            "subset":
                subset_name,

            "max_gap_minutes":
                minutes,

            "days":
                len(subset),

            "participants":
                subset[
                    "participant_id"
                ].nunique(),

            "median_observability":
                (
                    obs.median()
                    if len(obs)
                    else np.nan
                ),

            "p25_observability":
                (
                    obs.quantile(
                        0.25
                    )
                    if len(obs)
                    else np.nan
                ),

            "p75_observability":
                (
                    obs.quantile(
                        0.75
                    )
                    if len(obs)
                    else np.nan
                ),

            "median_longest_gap_hours":
                (
                    subset[
                        longest_col
                    ]
                    .median()
                    / 3_600_000
                    if len(subset)
                    else np.nan
                ),

            "median_observations":
                (
                    subset[
                        "n_gps_observations"
                    ]
                    .median()
                    if len(subset)
                    else np.nan
                ),
        })


sensitivity_result = pd.DataFrame(
    sensitivity_rows
)

sensitivity_result.to_csv(
    SENSITIVITY_OUT,
    index=False,
)


# ============================================================
# COMPACT OUTPUT
# ============================================================

print()
print("=" * 80)
print("PARTICIPANT SAMPLING SUMMARY")
print("=" * 80)

print(
    participant_result[
        [
            "participant_id",
            "n_source_files",
            "n_rows_read",
            "n_unique_timestamps",
            "n_duplicate_timestamps",
            "median_gap_seconds",
            "p90_gap_seconds",
            "p99_gap_seconds",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:.2f}",
    )
)


print()
print("=" * 80)
print("GLOBAL WITHIN-DAY GAP QUANTILES")
print("=" * 80)

print(
    gap_result.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print()
print("=" * 80)
print("CONTINUITY-TOLERANCE SENSITIVITY")
print("=" * 80)

print(
    sensitivity_result.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print()
print("=" * 80)
print("OUTPUTS")
print("=" * 80)

print(
    DAILY_OUT
)

print(
    PARTICIPANT_OUT
)

print(
    GAP_OUT
)

print(
    SENSITIVITY_OUT
)

print()
print("=" * 80)
print("DONE")
print("=" * 80)
