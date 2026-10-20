"""Cache raw GPS observations for the high-observability reference days.

The reference set is the existing 65 benchmark days with Cortex GPS DQ == 1.

Cortex is used ONLY to identify these high-observation reference days.
No Cortex feature functions are used.

The cached raw observations will be the starting point for RAPHE controlled
missingness experiments.
"""

from pathlib import Path

import numpy as np
import pandas as pd


DAY_MS = 86_400_000


INVENTORY_PATH = Path(
    "/shared/Digital Clinic/"
    "raphe_outputs/digital_clinic/"
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
    "raphe_outputs/digital_clinic/"
    "feature_reliability/"
    "gps_observability"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_PATH = (
    OUT_DIR
    / "gps_reference_raw_points.parquet"
)

SUMMARY_PATH = (
    OUT_DIR
    / "gps_reference_raw_points_summary.csv"
)


# ============================================================
# REFERENCE DAYS
# ============================================================

daily_frames = []


for path in sorted(
    DAILY_ROOT.glob("*.csv")
):

    df = pd.read_csv(
        path
    )

    required = [
        "participant_id",
        "mindlamp_uid",
        "timestamp",
        "date",
        "gps_data_quality",
    ]

    daily_frames.append(
        df[
            required
        ].copy()
    )


if not daily_frames:
    raise RuntimeError(
        "No benchmark participant_daily files found."
    )


daily = pd.concat(
    daily_frames,
    ignore_index=True,
)


daily["timestamp"] = pd.to_numeric(
    daily["timestamp"],
    errors="coerce",
)

daily["gps_data_quality"] = pd.to_numeric(
    daily["gps_data_quality"],
    errors="coerce",
)


reference = (
    daily[
        daily[
            "gps_data_quality"
        ]
        == 1.0
    ]
    .dropna(
        subset=[
            "timestamp",
            "mindlamp_uid",
        ]
    )
    .copy()
)


reference["timestamp"] = (
    reference[
        "timestamp"
    ]
    .astype(
        "int64"
    )
)


print("=" * 80)
print("RAPHE RAW GPS REFERENCE CACHE")
print("=" * 80)

print(
    "Reference days:",
    len(reference),
)

print(
    "Participants:",
    reference[
        "participant_id"
    ].nunique(),
)


# ============================================================
# SOURCE INVENTORY
# ============================================================

inventory = pd.read_parquet(
    INVENTORY_PATH
)


inventory = inventory[
    inventory[
        "scan_status"
    ]
    == "ok"
].copy()


for column in [
    "min_timestamp_ms",
    "max_timestamp_ms",
]:

    inventory[column] = pd.to_numeric(
        inventory[column],
        errors="coerce",
    )


# ============================================================
# LOAD RAW GPS ONCE PER PARTICIPANT
# ============================================================

reference_frames = []
summary_rows = []


for uid, refs in reference.groupby(
    "mindlamp_uid",
    sort=True,
):

    participant_id = (
        refs[
            "participant_id"
        ]
        .iloc[0]
    )


    refs = refs.sort_values(
        "timestamp"
    )


    earliest = int(
        refs[
            "timestamp"
        ].min()
    )

    latest = int(
        refs[
            "timestamp"
        ].max()
        + DAY_MS
    )


    participant_inventory = inventory[
        inventory[
            "source_participant_id"
        ]
        .astype(
            str
        )
        == str(
            uid
        )
    ].copy()


    participant_inventory = (
        participant_inventory[
            (
                participant_inventory[
                    "max_timestamp_ms"
                ]
                >= earliest
            )
            & (
                participant_inventory[
                    "min_timestamp_ms"
                ]
                < latest
            )
        ]
    )


    print()
    print("-" * 80)
    print(
        participant_id,
        uid,
        f"days={len(refs)}",
        f"files={len(participant_inventory)}",
    )


    raw_parts = []

    failed_files = 0


    for source_file in (
        participant_inventory[
            "source_file"
        ]
        .astype(
            str
        )
        .tolist()
    ):

        try:

            x = pd.read_csv(
                source_file,
                usecols=lambda c:
                    c in {
                        "timestamp",
                        "latitude",
                        "longitude",
                        "accuracy",
                    },
            )

        except Exception as e:

            failed_files += 1

            print(
                "  read failure:",
                Path(
                    source_file
                ).name,
                type(
                    e
                ).__name__,
            )

            continue


        required_columns = {
            "timestamp",
            "latitude",
            "longitude",
        }


        if not required_columns.issubset(
            x.columns
        ):

            failed_files += 1
            continue


        if (
            "accuracy"
            not in x.columns
        ):

            x[
                "accuracy"
            ] = np.nan


        raw_parts.append(
            x[
                [
                    "timestamp",
                    "latitude",
                    "longitude",
                    "accuracy",
                ]
            ]
        )


    if not raw_parts:

        print(
            "  ERROR: no raw GPS loaded"
        )

        continue


    raw = pd.concat(
        raw_parts,
        ignore_index=True,
    )


    for column in [
        "timestamp",
        "latitude",
        "longitude",
        "accuracy",
    ]:

        raw[column] = pd.to_numeric(
            raw[column],
            errors="coerce",
        )


    raw = raw[
        raw[
            "timestamp"
        ].notna()
    ].copy()


    raw[
        "timestamp"
    ] = (
        raw[
            "timestamp"
        ]
        .astype(
            "int64"
        )
    )


    participant_rows = []


    for ref in refs.itertuples(
        index=False
    ):

        start = int(
            ref.timestamp
        )

        end = (
            start
            + DAY_MS
        )


        day = raw[
            (
                raw[
                    "timestamp"
                ]
                >= start
            )
            & (
                raw[
                    "timestamp"
                ]
                < end
            )
        ].copy()


        if day.empty:
            continue


        day[
            "participant_id"
        ] = participant_id

        day[
            "mindlamp_uid"
        ] = uid

        day[
            "date"
        ] = str(
            ref.date
        )

        day[
            "day_start_ms"
        ] = start


        participant_rows.append(
            day
        )


    if not participant_rows:

        continue


    participant_reference = pd.concat(
        participant_rows,
        ignore_index=True,
    )


    unique_timestamps = (
        participant_reference[
            [
                "date",
                "timestamp",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )


    duplicates = (
        len(
            participant_reference
        )
        - unique_timestamps
    )


    invalid_coordinate_rows = int(
        (
            participant_reference[
                "latitude"
            ].isna()
            | participant_reference[
                "longitude"
            ].isna()
            | ~np.isfinite(
                participant_reference[
                    "latitude"
                ]
            )
            | ~np.isfinite(
                participant_reference[
                    "longitude"
                ]
            )
        )
        .sum()
    )


    summary_rows.append({
        "participant_id":
            participant_id,

        "mindlamp_uid":
            uid,

        "reference_days":
            participant_reference[
                "date"
            ].nunique(),

        "source_files":
            len(
                participant_inventory
            ),

        "failed_files":
            failed_files,

        "raw_rows":
            len(
                participant_reference
            ),

        "unique_day_timestamps":
            unique_timestamps,

        "duplicate_timestamp_rows":
            duplicates,

        "invalid_coordinate_rows":
            invalid_coordinate_rows,

        "missing_accuracy_rows":
            int(
                participant_reference[
                    "accuracy"
                ]
                .isna()
                .sum()
            ),
    })


    print(
        "  cached rows:",
        f"{len(participant_reference):,}",
    )

    print(
        "  unique timestamps:",
        f"{unique_timestamps:,}",
    )

    print(
        "  duplicate rows:",
        f"{duplicates:,}",
    )


    reference_frames.append(
        participant_reference
    )


# ============================================================
# SAVE
# ============================================================

if not reference_frames:
    raise RuntimeError(
        "No raw reference GPS observations generated."
    )


reference_raw = pd.concat(
    reference_frames,
    ignore_index=True,
)


reference_raw = reference_raw[
    [
        "participant_id",
        "mindlamp_uid",
        "date",
        "day_start_ms",
        "timestamp",
        "latitude",
        "longitude",
        "accuracy",
    ]
]


reference_raw.to_parquet(
    OUT_PATH,
    index=False,
)


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    SUMMARY_PATH,
    index=False,
)


print()
print("=" * 80)
print("REFERENCE RAW GPS SUMMARY")
print("=" * 80)

print(
    summary.to_string(
        index=False
    )
)


print()
print("=" * 80)
print("TOTAL")
print("=" * 80)

print(
    "Rows:",
    f"{len(reference_raw):,}",
)

print(
    "Participant-days:",
    reference_raw[
        [
            "participant_id",
            "date",
        ]
    ]
    .drop_duplicates()
    .shape[0],
)

print()
print("=" * 80)
print("OUTPUTS")
print("=" * 80)

print(
    OUT_PATH
)

print(
    SUMMARY_PATH
)

print()
print("=" * 80)
print("DONE")
print("=" * 80)
