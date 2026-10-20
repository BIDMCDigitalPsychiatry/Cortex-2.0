"""Audit RAPHE GPS edge behavior on well-observed benchmark days.

No Cortex feature functions are used.

The existing Cortex GPS DQ=1 days are used only as a convenient,
previously identified high-observation reference subset.

The purpose is to characterize empirical distributions of:

- supported edge duration
- edge displacement
- implied speed
- reported positional accuracy

before RAPHE defines stationary/movement semantics.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from raphe.features.gps_intervals import (
    build_gps_intervals,
)


DAY_MS = 86_400_000
MAX_GAP_MS = 60_000


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


EDGE_OUT = (
    OUT_DIR
    / "gps_reference_supported_edges.parquet"
)

PARTICIPANT_OUT = (
    OUT_DIR
    / "gps_reference_edge_summary_participants.csv"
)

QUANTILE_OUT = (
    OUT_DIR
    / "gps_reference_edge_quantiles.csv"
)

SPEED_BAND_OUT = (
    OUT_DIR
    / "gps_reference_speed_bands.csv"
)


# ============================================================
# LOAD HIGH-OBSERVATION REFERENCE DAYS
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
        daily["gps_data_quality"]
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
    reference["timestamp"]
    .astype("int64")
)


print("=" * 80)
print("RAPHE GPS REFERENCE-EDGE AUDIT")
print("=" * 80)

print(
    "Reference days:",
    len(reference),
)

print(
    "Participants:",
    reference["participant_id"].nunique(),
)

print(
    "Continuity tolerance:",
    f"{MAX_GAP_MS / 1000:.0f} seconds",
)


# ============================================================
# SOURCE INVENTORY
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


# ============================================================
# PROCESS EACH PARTICIPANT
# ============================================================

edge_frames = []


for uid, refs in reference.groupby(
    "mindlamp_uid",
    sort=True,
):

    participant_id = (
        refs["participant_id"]
        .iloc[0]
    )


    print()
    print("-" * 80)
    print(
        participant_id,
        uid,
        f"reference_days={len(refs)}",
    )


    inv = inventory[
        inventory["source_participant_id"]
        .astype(str)
        == str(uid)
    ].copy()


    selected_files = set()


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


        overlap = inv[
            (
                inv["max_timestamp_ms"]
                >= start
            )
            & (
                inv["min_timestamp_ms"]
                < end
            )
        ]


        selected_files.update(
            overlap["source_file"]
            .astype(str)
            .tolist()
        )


    raw_parts = []
    failed_files = 0


    for source_file in sorted(
        selected_files
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
                Path(source_file).name,
                type(e).__name__,
            )

            continue


        required = {
            "timestamp",
            "latitude",
            "longitude",
        }

        if not required.issubset(
            x.columns
        ):
            failed_files += 1
            continue


        if "accuracy" not in x.columns:
            x["accuracy"] = np.nan


        raw_parts.append(
            x
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


    for col in [
        "timestamp",
        "latitude",
        "longitude",
        "accuracy",
    ]:

        raw[col] = pd.to_numeric(
            raw[col],
            errors="coerce",
        )


    raw = (
        raw.dropna(
            subset=[
                "timestamp",
                "latitude",
                "longitude",
            ]
        )
        .copy()
    )


    raw["timestamp"] = (
        raw["timestamp"]
        .astype("int64")
    )


    print(
        "  raw rows:",
        f"{len(raw):,}",
        f"files={len(selected_files)}",
        f"failed={failed_files}",
    )


    participant_edge_count = 0


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


        day_gps = raw[
            (
                raw["timestamp"]
                >= start
            )
            & (
                raw["timestamp"]
                < end
            )
        ].copy()


        edges = build_gps_intervals(
            day_gps,
            window_start_ms=start,
            window_end_ms=end,
            max_gap_ms=MAX_GAP_MS,
        )


        if edges.empty:
            continue


        # For behavioral semantics we only inspect
        # temporally supported trajectory edges.
        edges = (
            edges[
                edges["supported"]
            ]
            .copy()
        )


        if edges.empty:
            continue


        edges["participant_id"] = (
            participant_id
        )

        edges["mindlamp_uid"] = (
            uid
        )

        edges["date"] = (
            ref.date
        )

        edges["distance_m"] = (
            edges["distance_km"]
            * 1000
        )


        # Raw accuracy metadata only.
        # Do not yet use it as an exclusion threshold.
        edges["max_endpoint_accuracy"] = (
            edges[
                [
                    "start_accuracy",
                    "end_accuracy",
                ]
            ]
            .max(
                axis=1,
                skipna=True,
            )
        )


        participant_edge_count += len(
            edges
        )

        edge_frames.append(
            edges
        )


    print(
        "  supported edges:",
        f"{participant_edge_count:,}",
    )


# ============================================================
# COMBINE
# ============================================================

if not edge_frames:
    raise RuntimeError(
        "No supported reference edges were generated."
    )


edges = pd.concat(
    edge_frames,
    ignore_index=True,
)


edges.to_parquet(
    EDGE_OUT,
    index=False,
)


# ============================================================
# CLEAN NUMERIC DOMAIN FOR SUMMARIES
# ============================================================

valid = edges[
    edges["distance_m"].notna()
    & edges["speed_kmh"].notna()
    & np.isfinite(
        edges["distance_m"]
    )
    & np.isfinite(
        edges["speed_kmh"]
    )
].copy()


# ============================================================
# QUANTILES
# ============================================================

metrics = {
    "gap_seconds":
        valid["gap_ms"]
        / 1000,

    "distance_m":
        valid["distance_m"],

    "speed_kmh":
        valid["speed_kmh"],

    "accuracy_raw":
        pd.concat(
            [
                valid["start_accuracy"],
                valid["end_accuracy"],
            ],
            ignore_index=True,
        )
        .dropna(),
}


quantile_rows = []


for metric, values in metrics.items():

    values = (
        pd.Series(values)
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )


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
        0.999,
    ]:

        quantile_rows.append({
            "metric":
                metric,

            "quantile":
                q,

            "value":
                (
                    values.quantile(
                        q
                    )
                    if len(values)
                    else np.nan
                ),

            "n":
                len(values),
        })


quantiles = pd.DataFrame(
    quantile_rows
)


quantiles.to_csv(
    QUANTILE_OUT,
    index=False,
)


# ============================================================
# PARTICIPANT-BALANCED SUMMARY
# ============================================================

participant_rows = []


for participant_id, x in valid.groupby(
    "participant_id",
    sort=True,
):

    accuracy = (
        pd.concat(
            [
                x["start_accuracy"],
                x["end_accuracy"],
            ]
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )


    participant_rows.append({
        "participant_id":
            participant_id,

        "reference_days":
            x["date"].nunique(),

        "supported_edges":
            len(x),

        "median_gap_seconds":
            (
                x["gap_ms"].median()
                / 1000
            ),

        "median_distance_m":
            x["distance_m"].median(),

        "p90_distance_m":
            x["distance_m"].quantile(
                0.90
            ),

        "p99_distance_m":
            x["distance_m"].quantile(
                0.99
            ),

        "median_speed_kmh":
            x["speed_kmh"].median(),

        "p90_speed_kmh":
            x["speed_kmh"].quantile(
                0.90
            ),

        "p99_speed_kmh":
            x["speed_kmh"].quantile(
                0.99
            ),

        "median_accuracy_raw":
            (
                accuracy.median()
                if len(accuracy)
                else np.nan
            ),

        "p90_accuracy_raw":
            (
                accuracy.quantile(
                    0.90
                )
                if len(accuracy)
                else np.nan
            ),

        "p99_accuracy_raw":
            (
                accuracy.quantile(
                    0.99
                )
                if len(accuracy)
                else np.nan
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
# DESCRIPTIVE SPEED BANDS
# ============================================================

# These are descriptive bins only.
# They are NOT RAPHE movement thresholds.

speed_bins = [
    -np.inf,
    0.1,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    50.0,
    100.0,
    200.0,
    np.inf,
]

speed_labels = [
    "<0.1",
    "0.1-0.5",
    "0.5-1",
    "1-2",
    "2-5",
    "5-10",
    "10-20",
    "20-50",
    "50-100",
    "100-200",
    ">=200",
]


valid["speed_band_kmh"] = pd.cut(
    valid["speed_kmh"],
    bins=speed_bins,
    labels=speed_labels,
    right=False,
)


band_rows = []


for participant_id, x in valid.groupby(
    "participant_id",
    sort=True,
):

    counts = (
        x["speed_band_kmh"]
        .value_counts(
            sort=False
        )
    )


    total = len(x)


    for band in speed_labels:

        n = int(
            counts.get(
                band,
                0,
            )
        )

        band_rows.append({
            "participant_id":
                participant_id,

            "speed_band_kmh":
                band,

            "n_edges":
                n,

            "fraction":
                (
                    n / total
                    if total
                    else np.nan
                ),
        })


speed_bands = pd.DataFrame(
    band_rows
)


speed_bands.to_csv(
    SPEED_BAND_OUT,
    index=False,
)


# ============================================================
# COMPACT PRINT
# ============================================================

print()
print("=" * 80)
print("PARTICIPANT EDGE SUMMARY")
print("=" * 80)

print(
    participant_summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.3f}",
    )
)


print()
print("=" * 80)
print("POOLED QUANTILES")
print("=" * 80)

pivot = (
    quantiles.pivot(
        index="quantile",
        columns="metric",
        values="value",
    )
    .reset_index()
)


print(
    pivot.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print()
print("=" * 80)
print("SPEED BAND FRACTIONS")
print("=" * 80)

band_pivot = (
    speed_bands.pivot(
        index="speed_band_kmh",
        columns="participant_id",
        values="fraction",
    )
)


print(
    band_pivot.to_string(
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print()
print("=" * 80)
print("OUTPUTS")
print("=" * 80)

print(EDGE_OUT)
print(PARTICIPANT_OUT)
print(QUANTILE_OUT)
print(SPEED_BAND_OUT)

print()
print("=" * 80)
print("DONE")
print("=" * 80)
