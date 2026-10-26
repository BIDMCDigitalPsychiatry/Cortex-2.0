"""Audit GPS displacement relative to reported positional accuracy.

Uses Cortex 2.0-native supported edges only.

No Cortex feature functions are used.

This analysis does not classify movement. It asks whether displacement
between consecutive GPS observations exceeds the combined reported
endpoint accuracy.

For endpoints with accuracy radii a1 and a2:

    displacement_ratio = distance_m / (a1 + a2)

A ratio <= 1 means the endpoint accuracy circles overlap geometrically.
This is descriptive evidence only; reported accuracy is not treated as a
formal statistical confidence interval.
"""

from pathlib import Path

import numpy as np
import pandas as pd


EDGE_PATH = Path(
    "/shared/Digital Clinic/"
    "cortex2_outputs/digital_clinic/"
    "feature_reliability/"
    "gps_observability/"
    "gps_reference_supported_edges.parquet"
)

OUT_DIR = EDGE_PATH.parent

DETAIL_OUT = (
    OUT_DIR
    / "gps_reference_displacement_accuracy.parquet"
)

PARTICIPANT_OUT = (
    OUT_DIR
    / "gps_reference_displacement_accuracy_participants.csv"
)

SPEED_OUT = (
    OUT_DIR
    / "gps_reference_displacement_accuracy_speedbands.csv"
)


edges = pd.read_parquet(
    EDGE_PATH
)


for col in [
    "distance_km",
    "speed_kmh",
    "start_accuracy",
    "end_accuracy",
    "gap_ms",
]:
    edges[col] = pd.to_numeric(
        edges[col],
        errors="coerce",
    )


edges["distance_m"] = (
    edges["distance_km"]
    * 1000
)

edges["gap_seconds"] = (
    edges["gap_ms"]
    / 1000
)


# ============================================================
# VALID ACCURACY
# ============================================================

valid_accuracy = (
    edges["start_accuracy"].notna()
    & edges["end_accuracy"].notna()
    & np.isfinite(
        edges["start_accuracy"]
    )
    & np.isfinite(
        edges["end_accuracy"]
    )
    & (
        edges["start_accuracy"]
        > 0
    )
    & (
        edges["end_accuracy"]
        > 0
    )
)


edges["combined_accuracy_m"] = np.nan

edges.loc[
    valid_accuracy,
    "combined_accuracy_m",
] = (
    edges.loc[
        valid_accuracy,
        "start_accuracy",
    ]
    + edges.loc[
        valid_accuracy,
        "end_accuracy",
    ]
)


edges["displacement_accuracy_ratio"] = (
    edges["distance_m"]
    / edges["combined_accuracy_m"]
)


edges["accuracy_regions_overlap"] = (
    edges["displacement_accuracy_ratio"]
    <= 1
)


# ============================================================
# DESCRIPTIVE RATIO BANDS
# ============================================================

ratio_bins = [
    -np.inf,
    0.25,
    0.50,
    1.00,
    2.00,
    5.00,
    10.00,
    np.inf,
]

ratio_labels = [
    "<0.25",
    "0.25-0.5",
    "0.5-1",
    "1-2",
    "2-5",
    "5-10",
    ">=10",
]


edges["ratio_band"] = pd.cut(
    edges["displacement_accuracy_ratio"],
    bins=ratio_bins,
    labels=ratio_labels,
    right=False,
)


speed_bins = [
    -np.inf,
    0.1,
    0.5,
    1,
    2,
    5,
    10,
    20,
    50,
    100,
    200,
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


edges["speed_band_kmh"] = pd.cut(
    edges["speed_kmh"],
    bins=speed_bins,
    labels=speed_labels,
    right=False,
)


edges.to_parquet(
    DETAIL_OUT,
    index=False,
)


# ============================================================
# PARTICIPANT SUMMARY
# ============================================================

participant_rows = []


for participant_id, x in edges.groupby(
    "participant_id",
    sort=True,
):

    usable = x[
        x["displacement_accuracy_ratio"]
        .notna()
        & np.isfinite(
            x["displacement_accuracy_ratio"]
        )
    ].copy()


    participant_rows.append({
        "participant_id":
            participant_id,

        "edges":
            len(x),

        "accuracy_usable_n":
            len(usable),

        "accuracy_usable_fraction":
            (
                len(usable)
                / len(x)
                if len(x)
                else np.nan
            ),

        "overlap_fraction":
            (
                (
                    usable[
                        "displacement_accuracy_ratio"
                    ]
                    <= 1
                )
                .mean()
                if len(usable)
                else np.nan
            ),

        "ratio_gt_1_fraction":
            (
                (
                    usable[
                        "displacement_accuracy_ratio"
                    ]
                    > 1
                )
                .mean()
                if len(usable)
                else np.nan
            ),

        "ratio_gt_2_fraction":
            (
                (
                    usable[
                        "displacement_accuracy_ratio"
                    ]
                    > 2
                )
                .mean()
                if len(usable)
                else np.nan
            ),

        "ratio_gt_5_fraction":
            (
                (
                    usable[
                        "displacement_accuracy_ratio"
                    ]
                    > 5
                )
                .mean()
                if len(usable)
                else np.nan
            ),

        "median_ratio":
            (
                usable[
                    "displacement_accuracy_ratio"
                ]
                .median()
                if len(usable)
                else np.nan
            ),

        "p90_ratio":
            (
                usable[
                    "displacement_accuracy_ratio"
                ]
                .quantile(
                    0.90
                )
                if len(usable)
                else np.nan
            ),

        "p99_ratio":
            (
                usable[
                    "displacement_accuracy_ratio"
                ]
                .quantile(
                    0.99
                )
                if len(usable)
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
# SPEED × ACCURACY-EVIDENCE TABLE
# ============================================================

speed_rows = []


for participant_id, participant in edges.groupby(
    "participant_id",
    sort=True,
):

    for speed_band in speed_labels:

        x = participant[
            participant[
                "speed_band_kmh"
            ]
            .astype(str)
            == speed_band
        ].copy()


        usable = x[
            x[
                "displacement_accuracy_ratio"
            ]
            .notna()
            & np.isfinite(
                x[
                    "displacement_accuracy_ratio"
                ]
            )
        ]


        speed_rows.append({
            "participant_id":
                participant_id,

            "speed_band_kmh":
                speed_band,

            "edges":
                len(x),

            "accuracy_usable_n":
                len(usable),

            "median_distance_m":
                (
                    x[
                        "distance_m"
                    ]
                    .median()
                    if len(x)
                    else np.nan
                ),

            "median_gap_seconds":
                (
                    x[
                        "gap_seconds"
                    ]
                    .median()
                    if len(x)
                    else np.nan
                ),

            "median_accuracy_ratio":
                (
                    usable[
                        "displacement_accuracy_ratio"
                    ]
                    .median()
                    if len(usable)
                    else np.nan
                ),

            "ratio_gt_1_fraction":
                (
                    (
                        usable[
                            "displacement_accuracy_ratio"
                        ]
                        > 1
                    )
                    .mean()
                    if len(usable)
                    else np.nan
                ),

            "ratio_gt_2_fraction":
                (
                    (
                        usable[
                            "displacement_accuracy_ratio"
                        ]
                        > 2
                    )
                    .mean()
                    if len(usable)
                    else np.nan
                ),
        })


speed_summary = pd.DataFrame(
    speed_rows
)

speed_summary.to_csv(
    SPEED_OUT,
    index=False,
)


# ============================================================
# PRINT COMPACT RESULTS
# ============================================================

print("=" * 80)
print("DISPLACEMENT VS REPORTED ACCURACY")
print("=" * 80)

print(
    participant_summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print()
print("=" * 80)
print("SPEED × SPATIAL-EVIDENCE SUMMARY")
print("=" * 80)

display = speed_summary[
    speed_summary["edges"]
    > 0
].copy()


print(
    display.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print()
print("=" * 80)
print("RATIO DISTRIBUTION")
print("=" * 80)

ratio_table = (
    edges[
        edges["ratio_band"]
        .notna()
    ]
    .groupby(
        [
            "participant_id",
            "ratio_band",
        ],
        observed=True,
    )
    .size()
    .rename(
        "n"
    )
    .reset_index()
)


ratio_table[
    "fraction"
] = (
    ratio_table[
        "n"
    ]
    / ratio_table.groupby(
        "participant_id"
    )[
        "n"
    ]
    .transform(
        "sum"
    )
)


print(
    ratio_table.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print()
print("=" * 80)
print("OUTPUTS")
print("=" * 80)

print(DETAIL_OUT)
print(PARTICIPANT_OUT)
print(SPEED_OUT)

print()
print("=" * 80)
print("DONE")
print("=" * 80)
