"""Cache raw GPS observations for Digital Clinic daily EMA days.

Uses the same raw-source loading strategy as the validated weekly
clinical GPS cache:
inventory manifest -> overlapping source CSV files -> requested days.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

DC = Path("/shared/Digital Clinic")

EMA_PATH = (
    DC
    / "cortex2_outputs/digital_clinic/"
      "clinical/daily_ema/"
      "daily_ema_gps_participants.csv"
)

MAP_PATH = (
    DC
    / "raphe_outputs/digital_clinic/"
      "inventory/raphe_participant_map.csv"
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

OUT_DIR = (
    DC
    / "cortex2_outputs/digital_clinic/"
      "clinical/daily_ema"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RAW_OUT = (
    OUT_DIR
    / "daily_ema_gps_raw_points.parquet"
)

SUMMARY_OUT = (
    OUT_DIR
    / "daily_ema_gps_raw_coverage.csv"
)


# ============================================================
# LOAD EMA + PARTICIPANT MAP + INVENTORY
# ============================================================

ema = pd.read_csv(
    EMA_PATH,
    low_memory=False,
)

participant_map = pd.read_csv(
    MAP_PATH,
    dtype=str,
)

inventory = pd.read_parquet(
    INVENTORY_PATH
)


ema["date"] = pd.to_datetime(
    ema["date"],
    errors="coerce",
).dt.strftime("%Y-%m-%d")


# ============================================================
# UNIQUE REQUESTED PARTICIPANT-DAYS
# ============================================================

days = (
    ema[
        [
            "cortex_participant_id",
            "date",
        ]
    ]
    .dropna()
    .drop_duplicates()
    .rename(
        columns={
            "cortex_participant_id":
                "participant_id"
        }
    )
)

days = days.merge(
    participant_map[
        [
            "participant_id",
            "source_participant_id",
        ]
    ],
    on="participant_id",
    how="left",
)

if days[
    "source_participant_id"
].isna().any():

    missing = (
        days.loc[
            days[
                "source_participant_id"
            ].isna(),
            "participant_id",
        ]
        .unique()
    )

    raise RuntimeError(
        f"Missing participant mappings: {missing}"
    )


# UTC calendar-day windows.

date_ts = pd.to_datetime(
    days["date"],
    utc=True,
)

days[
    "start_ms"
] = (
    date_ts.astype("int64")
    // 1_000_000
)

days[
    "end_ms"
] = (
    (
        date_ts
        + pd.Timedelta(days=1)
    )
    .astype("int64")
    // 1_000_000
)

days = (
    days.sort_values(
        [
            "participant_id",
            "date",
        ]
    )
    .reset_index(
        drop=True
    )
)


if days[
    [
        "participant_id",
        "date",
    ]
].duplicated().any():

    raise RuntimeError(
        "Duplicate participant-day rows remain."
    )


print("=" * 100)
print("CORTEX 2.0 DAILY EMA RAW GPS CACHE")
print("=" * 100)

print(
    "Participants:",
    days[
        "participant_id"
    ].nunique(),
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
            ]
            .astype(str)
            .eq(
                str(uid)
            )
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


    # ========================================================
    # READ RAW CSV SOURCE FILES
    # ========================================================

    for _, source in (
        source_files.iterrows()
    ):

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
            x[
                "timestamp"
            ].notna()
        ].copy()

        x[
            "timestamp"
        ] = x[
            "timestamp"
        ].astype(
            np.int64
        )


        # Keep only the broad time range needed for this participant.
        x = x[
            (
                x[
                    "timestamp"
                ]
                >= overall_start
            )
            & (
                x[
                    "timestamp"
                ]
                < overall_end
            )
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

        # Protect against duplicated rows coming from overlapping
        # or duplicated source files.
        participant_raw = (
            participant_raw
            .drop_duplicates(
                subset=[
                    "timestamp",
                    "latitude",
                    "longitude",
                    "accuracy",
                ]
            )
            .sort_values(
                "timestamp"
            )
            .reset_index(
                drop=True
            )
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


    # ========================================================
    # CUT INTO REQUESTED EMA DAYS
    # ========================================================

    participant_cached = []

    days_with_raw = 0
    days_without_raw = 0


    for _, day in (
        participant_days.iterrows()
    ):

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
            (
                participant_raw[
                    "timestamp"
                ]
                >= start_ms
            )
            & (
                participant_raw[
                    "timestamp"
                ]
                < end_ms
            )
        ].copy()


        if len(x):

            days_with_raw += 1

        else:

            days_without_raw += 1


        x[
            "participant_id"
        ] = participant_id

        x[
            "mindlamp_uid"
        ] = str(uid)

        x[
            "date"
        ] = str(
            day[
                "date"
            ]
        )

        x[
            "day_start_ms"
        ] = start_ms

        x[
            "day_end_ms"
        ] = end_ms


        if len(x):

            participant_cached.append(
                x
            )


        summary_rows.append(
            {
                "participant_id":
                    participant_id,

                "mindlamp_uid":
                    str(uid),

                "date":
                    str(
                        day[
                            "date"
                        ]
                    ),

                "day_start_ms":
                    start_ms,

                "day_end_ms":
                    end_ms,

                "n_gps_points":
                    len(x),

                "has_raw_gps":
                    bool(
                        len(x)
                    ),
            }
        )


    if participant_cached:

        participant_cached = (
            pd.concat(
                participant_cached,
                ignore_index=True,
            )
        )

        cached_frames.append(
            participant_cached
        )


    print(
        f"raw_rows={len(participant_raw):,} | "
        f"days_with_raw={days_with_raw} | "
        f"days_without_raw={days_without_raw} | "
        f"failed_files={len(failed_files)}"
    )


    if failed_files:

        print(
            "  First failed file:",
            failed_files[0],
        )


# ============================================================
# COMBINE
# ============================================================

if cached_frames:

    raw = pd.concat(
        cached_frames,
        ignore_index=True,
    )

    raw = (
        raw.sort_values(
            [
                "participant_id",
                "timestamp",
            ]
        )
        .reset_index(
            drop=True
        )
    )

else:

    raw = pd.DataFrame(
        columns=[
            "timestamp",
            "latitude",
            "longitude",
            "accuracy",
            "participant_id",
            "mindlamp_uid",
            "date",
            "day_start_ms",
            "day_end_ms",
        ]
    )


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# SAVE
# ============================================================

raw.to_parquet(
    RAW_OUT,
    index=False,
)

summary.to_csv(
    SUMMARY_OUT,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 100)
print("RESULT")
print("=" * 100)

print(
    "Raw GPS rows retained:",
    f"{len(raw):,}",
)

print(
    "Requested participant-days:",
    len(summary),
)

print(
    "Days with raw GPS:",
    int(
        summary[
            "has_raw_gps"
        ].sum()
    ),
)

print(
    "Days without raw GPS:",
    int(
        (
            ~summary[
                "has_raw_gps"
            ]
        ).sum()
    ),
)

print(
    "Raw GPS day availability:",
    f"{summary['has_raw_gps'].mean():.1%}",
)

print(
    "Participants with >=1 GPS day:",
    summary.loc[
        summary[
            "has_raw_gps"
        ],
        "participant_id",
    ].nunique(),
)

if summary[
    "has_raw_gps"
].any():

    observed = summary[
        summary[
            "has_raw_gps"
        ]
    ]

    print()
    print(
        "GPS points/day among available days:"
    )

    print(
        observed[
            "n_gps_points"
        ]
        .describe(
            percentiles=[
                .25,
                .5,
                .75,
            ]
        )
        .to_string()
    )


print()
print("Saved:")
print(" ", RAW_OUT)
print(" ", SUMMARY_OUT)
