"""Cache raw GPS for the Cortex 2.0 weekly clinical-window analysis.

Reads only GPS source files overlapping participant-days needed for
the 7-day windows preceding weekly depression/anxiety assessments.

No GPS features are computed here.
"""

from pathlib import Path

import numpy as np
import pandas as pd


DC = Path("/shared/Digital Clinic")

ROOT = (
    DC
    / "cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical"
)

REQUESTED_DAYS_PATH = (
    ROOT
    / "gps_clinical_weekly_requested_days.parquet"
)

INVENTORY_PATH = (
    DC
    / "cortex2_outputs/digital_clinic/"
    "inventory/source_files/gps.parquet"
)

if not INVENTORY_PATH.exists():
    INVENTORY_PATH = (
        DC
        / "raphe_outputs/digital_clinic/"
        "inventory/source_files/gps.parquet"
    )

RAW_OUT = (
    ROOT
    / "gps_clinical_weekly_raw_points.parquet"
)

SUMMARY_OUT = (
    ROOT
    / "gps_clinical_weekly_raw_points_summary.csv"
)


# ============================================================
# LOAD + DEDUPLICATE REQUESTED DAYS
# ============================================================

requested = pd.read_parquet(
    REQUESTED_DAYS_PATH
)

inventory = pd.read_parquet(
    INVENTORY_PATH
)

requested["date"] = (
    pd.to_datetime(
        requested["date"]
    )
    .dt.strftime("%Y-%m-%d")
)

# The same GPS day can belong to multiple weekly clinical windows.
# Cache it only once.
days = (
    requested[
        [
            "participant_id",
            "mindlamp_uid",
            "date",
            "start_ms",
            "end_ms",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "participant_id",
            "date",
        ]
    )
    .reset_index(drop=True)
)

if days[
    [
        "participant_id",
        "date",
    ]
].duplicated().any():
    raise RuntimeError(
        "Duplicate participant-day rows remain after deduplication."
    )


print("=" * 80)
print("Cortex 2.0 CLINICAL-WINDOW RAW GPS CACHE")
print("=" * 80)

print(
    "Participants:",
    days["participant_id"].nunique(),
)

print(
    "Unique participant-days:",
    len(days),
)

print(
    "Date range:",
    days["date"].min(),
    "→",
    days["date"].max(),
)

print(
    "GPS inventory:",
    INVENTORY_PATH,
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
        "mindlamp_uid",
    ],
    sort=True,
):

    participant_days = (
        participant_days
        .sort_values("start_ms")
        .copy()
    )

    overall_start = int(
        participant_days["start_ms"].min()
    )

    overall_end = int(
        participant_days["end_ms"].max()
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
            source["source_file"]
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

        for col in [
            "timestamp",
            "latitude",
            "longitude",
            "accuracy",
        ]:
            x[col] = pd.to_numeric(
                x[col],
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
            participant_chunks.append(x)

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

    days_with_raw = 0
    days_without_raw = 0

    for _, day in participant_days.iterrows():

        start_ms = int(
            day["start_ms"]
        )

        end_ms = int(
            day["end_ms"]
        )

        x = participant_raw[
            (participant_raw["timestamp"] >= start_ms)
            & (participant_raw["timestamp"] < end_ms)
        ].copy()

        if len(x):
            days_with_raw += 1
        else:
            days_without_raw += 1

        x["participant_id"] = (
            participant_id
        )

        x["mindlamp_uid"] = (
            str(uid)
        )

        x["date"] = (
            str(day["date"])
        )

        x["day_start_ms"] = (
            start_ms
        )

        x["day_end_ms"] = (
            end_ms
        )

        participant_cached.append(x)

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

    invalid_coordinates = (
        int(
            (
                participant_cached[
                    "latitude"
                ].isna()
                | participant_cached[
                    "longitude"
                ].isna()
            ).sum()
        )
        if raw_rows
        else 0
    )

    missing_accuracy = (
        int(
            participant_cached[
                "accuracy"
            ].isna().sum()
        )
        if raw_rows
        else 0
    )

    summary_rows.append(
        {
            "participant_id":
                participant_id,

            "mindlamp_uid":
                str(uid),

            "requested_days":
                len(participant_days),

            "days_with_raw":
                days_with_raw,

            "days_without_raw":
                days_without_raw,

            "source_files":
                len(source_files),

            "failed_files":
                len(failed_files),

            "raw_rows":
                raw_rows,

            "unique_timestamps":
                unique_timestamps,

            "duplicate_timestamp_rows":
                duplicate_timestamp_rows,

            "invalid_coordinate_rows":
                invalid_coordinates,

            "missing_accuracy_rows":
                missing_accuracy,
        }
    )

    print(
        "  cached rows:",
        f"{raw_rows:,}",
    )

    print(
        "  days with raw:",
        days_with_raw,
        "/",
        len(participant_days),
    )

    print(
        "  failed files:",
        len(failed_files),
    )


# ============================================================
# SAVE
# ============================================================

nonempty = [
    x
    for x in cached_frames
    if not x.empty
]

if nonempty:
    cached = pd.concat(
        nonempty,
        ignore_index=True,
    )
else:
    cached = pd.DataFrame()

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
# VALIDATION + REPORT
# ============================================================

if summary[
    "failed_files"
].sum() != 0:
    raise RuntimeError(
        "One or more GPS source files failed to read."
    )

print()
print("=" * 80)
print("CLINICAL-WINDOW RAW GPS SUMMARY")
print("=" * 80)

print(
    "Requested participant-days:",
    len(days),
)

print(
    "Days with raw GPS:",
    int(
        summary[
            "days_with_raw"
        ].sum()
    ),
)

print(
    "Days without raw GPS:",
    int(
        summary[
            "days_without_raw"
        ].sum()
    ),
)

print(
    "Total raw rows:",
    f"{len(cached):,}",
)

print(
    "Participants cached:",
    cached[
        "participant_id"
    ].nunique()
    if len(cached)
    else 0,
)

print(
    "Invalid coordinate rows:",
    int(
        summary[
            "invalid_coordinate_rows"
        ].sum()
    ),
)

print(
    "Missing accuracy rows:",
    int(
        summary[
            "missing_accuracy_rows"
        ].sum()
    ),
)

print(
    "Failed source files:",
    int(
        summary[
            "failed_files"
        ].sum()
    ),
)

print(
    "\nSaved raw cache:",
    RAW_OUT,
)

print(
    "Saved summary:",
    SUMMARY_OUT,
)
