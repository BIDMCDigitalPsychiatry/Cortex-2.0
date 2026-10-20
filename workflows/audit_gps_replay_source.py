from pathlib import Path

import pandas as pd


GPS_PATH = Path(
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


# ============================================================
# LOAD GPS
# ============================================================

gps = pd.read_parquet(
    GPS_PATH
)


print("=" * 80)
print("GPS REPLAY SOURCE AUDIT")
print("=" * 80)

print(
    "Rows:",
    len(gps),
)

print(
    "Columns:",
    list(gps.columns),
)

print()

print(
    gps.head(3).to_string(
        index=False
    )
)


# ============================================================
# LOAD BENCHMARK DAILY TABLES
# ============================================================

daily_frames = []

for path in sorted(
    DAILY_ROOT.glob("*.csv")
):

    df = pd.read_csv(
        path
    )

    daily_frames.append(
        df
    )


daily = pd.concat(
    daily_frames,
    ignore_index=True,
)

daily["date"] = pd.to_datetime(
    daily["date"]
)


reference = daily[
    daily["gps_data_quality"] == 1.0
].copy()


print()
print("=" * 80)
print("DQ=1 REFERENCE DAYS")
print("=" * 80)

print(
    "Days:",
    len(reference),
)

print(
    "Participants:",
    reference["participant_id"].nunique(),
)

print(
    reference[
        [
            "participant_id",
            "mindlamp_uid",
            "date",
        ]
    ]
    .groupby(
        [
            "participant_id",
            "mindlamp_uid",
        ]
    )
    .size()
    .rename("reference_days")
    .reset_index()
    .to_string(
        index=False
    )
)


# ============================================================
# IDENTIFY LIKELY ID / TIME COLUMNS
# ============================================================

print()
print("=" * 80)
print("COLUMN DIAGNOSTICS")
print("=" * 80)

for col in gps.columns:

    lc = col.lower()

    if any(
        key in lc
        for key in [
            "participant",
            "uid",
            "lamp",
            "timestamp",
            "time",
            "date",
            "lat",
            "lon",
            "accuracy",
        ]
    ):

        s = gps[col]

        print(
            f"{col:30s} "
            f"dtype={str(s.dtype):20s} "
            f"non_null={int(s.notna().sum()):10d} "
            f"unique={int(s.nunique(dropna=True)):10d}"
        )


# ============================================================
# TIMESTAMP DIAGNOSTIC
# ============================================================

timestamp_candidates = [
    c
    for c in gps.columns
    if c.lower()
    in {
        "timestamp",
        "timestamp_ms",
        "time",
        "datetime",
    }
]


if timestamp_candidates:

    ts_col = (
        "timestamp"
        if "timestamp" in timestamp_candidates
        else timestamp_candidates[0]
    )

    ts = pd.to_numeric(
        gps[ts_col],
        errors="coerce",
    )

    print()
    print("=" * 80)
    print("TIMESTAMP RANGE")
    print("=" * 80)

    print(
        "Column:",
        ts_col,
    )

    print(
        "Min:",
        ts.min(),
    )

    print(
        "Max:",
        ts.max(),
    )

    if ts.dropna().median() > 1e11:

        converted = pd.to_datetime(
            ts,
            unit="ms",
            utc=True,
            errors="coerce",
        )

        print(
            "UTC min:",
            converted.min(),
        )

        print(
            "UTC max:",
            converted.max(),
        )


# ============================================================
# CHECK BENCHMARK UID OVERLAP
# ============================================================

uid_candidates = [
    c
    for c in gps.columns
    if (
        "uid" in c.lower()
        or "lamp" in c.lower()
    )
]


benchmark_uids = set(
    reference[
        "mindlamp_uid"
    ]
    .astype(str)
)


print()
print("=" * 80)
print("BENCHMARK UID OVERLAP")
print("=" * 80)

for col in uid_candidates:

    values = set(
        gps[col]
        .dropna()
        .astype(str)
    )

    overlap = (
        benchmark_uids
        & values
    )

    print(
        f"{col}: "
        f"{len(overlap)}/{len(benchmark_uids)} benchmark UIDs matched"
    )

    if overlap:
        print(
            "  matched:",
            sorted(overlap)
        )


# ============================================================
# GPS STRUCTURE
# ============================================================

required = {
    "timestamp",
    "latitude",
    "longitude",
}

print()
print("=" * 80)
print("CORE GPS FIELDS")
print("=" * 80)

print(
    "Required fields present:",
    required.issubset(
        set(gps.columns)
    )
)

for col in [
    "timestamp",
    "latitude",
    "longitude",
    "accuracy",
    "altitude",
]:

    if col in gps.columns:

        print(
            f"{col}: "
            f"missing={gps[col].isna().sum():,}"
        )


print()
print("=" * 80)
print("DONE")
print("=" * 80)
