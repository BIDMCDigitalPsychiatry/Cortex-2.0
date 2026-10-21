"""Cache raw GPS for the locked held-out RAPHE validation cohort.

Reads only the source files overlapping the 112 locked validation days.
No feature values are computed here.
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

LOCKED_DAYS_PATH = (
    ROOT
    / "gps_validation_locked_days.parquet"
)

INVENTORY_PATH = Path(
    "/shared/Digital Clinic/"
    "raphe_outputs/digital_clinic/"
    "inventory/source_files/gps.parquet"
)

RAW_OUT = (
    ROOT
    / "gps_validation_raw_points.parquet"
)

SUMMARY_OUT = (
    ROOT
    / "gps_validation_raw_points_summary.csv"
)


# ============================================================
# LOAD
# ============================================================

days = pd.read_parquet(
    LOCKED_DAYS_PATH
)

inventory = pd.read_parquet(
    INVENTORY_PATH
)

days["date"] = (
    days["date"]
    .astype(str)
)


print("=" * 80)
print("RAPHE HELD-OUT VALIDATION RAW GPS CACHE")
print("=" * 80)

print(
    "Participants:",
    days["participant_id"].nunique(),
)

print(
    "Reference days:",
    len(days),
)


# ============================================================
# CACHE
# ============================================================

cached_frames = []
summary_rows = []


for (
    participant_id,
    uid,
), participant_days in days.groupby(
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


    print()
    print("-" * 80)

    print(
        participant_id,
        uid,
        f"days={len(participant_days)}",
        f"files={len(source_files)}",
    )


    participant_chunks = []
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
                    "latitude",
                    "longitude",
                    "accuracy",
                ],
            )

        except Exception as exc:

            failed_files.append(
                (
                    str(path),
                    repr(exc),
                )
            )

            continue


        x["timestamp"] = pd.to_numeric(
            x["timestamp"],
            errors="coerce",
        )

        x["latitude"] = pd.to_numeric(
            x["latitude"],
            errors="coerce",
        )

        x["longitude"] = pd.to_numeric(
            x["longitude"],
            errors="coerce",
        )

        x["accuracy"] = pd.to_numeric(
            x["accuracy"],
            errors="coerce",
        )


        x = x[
            x["timestamp"].notna()
        ].copy()


        x["timestamp"] = (
            x["timestamp"]
            .astype(np.int64)
        )


        x = x[
            (x["timestamp"] >= overall_start)
            & (x["timestamp"] < overall_end)
        ]


        if len(x):

            participant_chunks.append(
                x
            )


    if participant_chunks:

        participant_raw = pd.concat(
            participant_chunks,
            ignore_index=True,
        )

    else:

        participant_raw = pd.DataFrame(
            columns=[
                "timestamp",
                "latitude",
                "longitude",
                "accuracy",
            ]
        )


    participant_cached = []


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


        x = participant_raw[
            (participant_raw["timestamp"] >= start_ms)
            & (participant_raw["timestamp"] < end_ms)
        ].copy()


        x["participant_id"] = (
            participant_id
        )

        x["mindlamp_uid"] = (
            str(uid)
        )

        x["episode_id"] = (
            day[
                "episode_id"
            ]
        )

        x["date"] = (
            str(
                day[
                    "date"
                ]
            )
        )

        x["day_start_ms"] = (
            start_ms
        )

        x["day_end_ms"] = (
            end_ms
        )

        x["validation_day_number"] = int(
            day[
                "validation_day_number"
            ]
        )

        x["sampling_density_quartile"] = (
            day[
                "sampling_density_quartile"
            ]
        )

        x["reference_observability"] = float(
            day[
                "observability"
            ]
        )


        participant_cached.append(
            x
        )


    if participant_cached:

        participant_cached = pd.concat(
            participant_cached,
            ignore_index=True,
        )

    else:

        participant_cached = pd.DataFrame()


    cached_frames.append(
        participant_cached
    )


    raw_rows = len(
        participant_cached
    )

    unique_timestamps = (
        participant_cached[
            "timestamp"
        ].nunique()
        if raw_rows
        else 0
    )

    duplicate_timestamp_rows = (
        raw_rows
        - unique_timestamps
    )


    invalid_coordinates = int(
        (
            participant_cached[
                "latitude"
            ].isna()
            | participant_cached[
                "longitude"
            ].isna()
        ).sum()
    ) if raw_rows else 0


    missing_accuracy = int(
        participant_cached[
            "accuracy"
        ].isna().sum()
    ) if raw_rows else 0


    summary_rows.append({
        "participant_id":
            participant_id,

        "mindlamp_uid":
            str(uid),

        "sampling_density_quartile":
            participant_days[
                "sampling_density_quartile"
            ].iloc[0],

        "reference_days":
            len(
                participant_days
            ),

        "source_files":
            len(
                source_files
            ),

        "failed_files":
            len(
                failed_files
            ),

        "raw_rows":
            raw_rows,

        "unique_day_timestamps":
            unique_timestamps,

        "duplicate_timestamp_rows":
            duplicate_timestamp_rows,

        "invalid_coordinate_rows":
            invalid_coordinates,

        "missing_accuracy_rows":
            missing_accuracy,
    })


    print(
        "  cached rows:",
        f"{raw_rows:,}",
    )

    print(
        "  unique timestamps:",
        f"{unique_timestamps:,}",
    )

    print(
        "  duplicate rows:",
        f"{duplicate_timestamp_rows:,}",
    )


# ============================================================
# SAVE
# ============================================================

cached = pd.concat(
    cached_frames,
    ignore_index=True,
)

summary = pd.DataFrame(
    summary_rows
)


cached.to_parquet(
    RAW_OUT,
    index=False,
)

summary.to_csv(
    SUMMARY_OUT,
    index=False,
)


# ============================================================
# VALIDATION
# ============================================================

actual_days = (
    cached[
        [
            "participant_id",
            "date",
        ]
    ]
    .drop_duplicates()
    .shape[0]
)


if actual_days != 112:

    raise RuntimeError(
        f"Expected raw GPS for 112 participant-days, "
        f"found {actual_days}."
    )


if summary[
    "failed_files"
].sum() != 0:

    raise RuntimeError(
        "One or more source files failed to read."
    )


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 80)
print("VALIDATION RAW GPS SUMMARY")
print("=" * 80)

print(
    summary.to_string(
        index=False,
    )
)


print()
print("=" * 80)
print("TOTAL")
print("=" * 80)

print(
    "Rows:",
    f"{len(cached):,}",
)

print(
    "Participants:",
    cached[
        "participant_id"
    ].nunique(),
)

print(
    "Participant-days:",
    actual_days,
)

print(
    "Duplicate timestamp rows:",
    f"{summary['duplicate_timestamp_rows'].sum():,}",
)

print(
    "Invalid coordinate rows:",
    f"{summary['invalid_coordinate_rows'].sum():,}",
)

print(
    "Missing accuracy rows:",
    f"{summary['missing_accuracy_rows'].sum():,}",
)


print()
print("=" * 80)
print("OUTPUTS")
print("=" * 80)

print(RAW_OUT)
print(SUMMARY_OUT)

print()
print("=" * 80)
print("DONE")
print("=" * 80)
