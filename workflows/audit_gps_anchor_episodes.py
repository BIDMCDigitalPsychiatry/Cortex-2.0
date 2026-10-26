"""Audit fixed-anchor GPS location-compatible episodes.

This tests whether requiring every observation to remain compatible with
one fixed uncertainty-aware spatial anchor prevents the transitive chaining
observed in sequential pairwise overlap episodes.

No Cortex feature functions are used.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cortex2_engine.features.gps_anchor_episodes import (
    build_anchor_compatible_location_episodes,
)

from cortex2_engine.features.gps_spatial_semantics import (
    add_gps_spatial_semantics,
)


EDGE_PATH = Path(
    "/shared/Digital Clinic/"
    "cortex2_outputs/digital_clinic/"
    "feature_reliability/"
    "gps_observability/"
    "gps_reference_supported_edges.parquet"
)

OUT_DIR = EDGE_PATH.parent

EPISODE_OUT = (
    OUT_DIR
    / "gps_reference_anchor_compatible_episodes.parquet"
)

PARTICIPANT_OUT = (
    OUT_DIR
    / "gps_reference_anchor_episode_participants.csv"
)

QUANTILE_OUT = (
    OUT_DIR
    / "gps_reference_anchor_episode_quantiles.csv"
)

DURATION_OUT = (
    OUT_DIR
    / "gps_reference_anchor_episode_duration_bands.csv"
)


edges = pd.read_parquet(
    EDGE_PATH
)

edges = add_gps_spatial_semantics(
    edges
)


print("=" * 80)
print("Cortex 2.0 FIXED-ANCHOR LOCATION EPISODE AUDIT")
print("=" * 80)

print(
    "Edges:",
    f"{len(edges):,}",
)

print(
    "Accuracy-overlap edges:",
    f"{(edges['spatial_status'] == 'accuracy_overlap').sum():,}",
)


episode_frames = []


for (
    participant_id,
    uid,
    date,
), day_edges in edges.groupby(
    [
        "participant_id",
        "mindlamp_uid",
        "date",
    ],
    sort=True,
):

    day_edges = (
        day_edges
        .sort_values(
            [
                "start_ms",
                "end_ms",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    episodes = (
        build_anchor_compatible_location_episodes(
            day_edges
        )
    )

    if episodes.empty:
        continue

    episodes[
        "participant_id"
    ] = participant_id

    episodes[
        "mindlamp_uid"
    ] = uid

    episodes[
        "date"
    ] = date

    episode_frames.append(
        episodes
    )


if not episode_frames:
    raise RuntimeError(
        "No anchor-compatible episodes generated."
    )


episodes = pd.concat(
    episode_frames,
    ignore_index=True,
)


episodes[
    "duration_seconds"
] = (
    episodes[
        "duration_ms"
    ]
    / 1000
)


episodes.to_parquet(
    EPISODE_OUT,
    index=False,
)


# ============================================================
# PARTICIPANT SUMMARY
# ============================================================

participant_rows = []


for participant_id, x in episodes.groupby(
    "participant_id",
    sort=True,
):

    daily_counts = (
        x.groupby(
            "date"
        )
        .size()
    )

    participant_rows.append({
        "participant_id":
            participant_id,

        "reference_days":
            x[
                "date"
            ].nunique(),

        "episodes":
            len(x),

        "episodes_per_day_median":
            daily_counts.median(),

        "median_duration_seconds":
            x[
                "duration_seconds"
            ].median(),

        "p90_duration_seconds":
            x[
                "duration_seconds"
            ].quantile(
                0.90
            ),

        "p99_duration_seconds":
            x[
                "duration_seconds"
            ].quantile(
                0.99
            ),

        "median_anchor_accuracy_m":
            x[
                "anchor_accuracy_m"
            ].median(),

        "p90_anchor_accuracy_m":
            x[
                "anchor_accuracy_m"
            ].quantile(
                0.90
            ),

        "median_max_anchor_distance_m":
            x[
                "max_anchor_distance_m"
            ].median(),

        "p90_max_anchor_distance_m":
            x[
                "max_anchor_distance_m"
            ].quantile(
                0.90
            ),

        "p99_max_anchor_distance_m":
            x[
                "max_anchor_distance_m"
            ].quantile(
                0.99
            ),

        "median_path_distance_m":
            x[
                "path_distance_m"
            ].median(),

        "p99_path_distance_m":
            x[
                "path_distance_m"
            ].quantile(
                0.99
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
# POOLED QUANTILES
# ============================================================

metrics = {
    "duration_seconds":
        episodes[
            "duration_seconds"
        ],

    "n_edges":
        episodes[
            "n_edges"
        ],

    "anchor_accuracy_m":
        episodes[
            "anchor_accuracy_m"
        ],

    "max_anchor_distance_m":
        episodes[
            "max_anchor_distance_m"
        ],

    "max_anchor_ratio":
        episodes[
            "max_anchor_ratio"
        ],

    "path_distance_m":
        episodes[
            "path_distance_m"
        ],
}


rows = []


for metric, values in metrics.items():

    values = (
        pd.to_numeric(
            values,
            errors="coerce",
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

    for q in [
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
    ]:

        rows.append({
            "metric":
                metric,
            "quantile":
                q,
            "value":
                values.quantile(
                    q
                ),
        })


quantiles = pd.DataFrame(
    rows
)

quantiles.to_csv(
    QUANTILE_OUT,
    index=False,
)


# ============================================================
# DURATION BANDS
# ============================================================

bins = [
    -np.inf,
    5,
    30,
    120,
    600,
    1800,
    7200,
    np.inf,
]

labels = [
    "<5 sec",
    "5-30 sec",
    "30 sec-2 min",
    "2-10 min",
    "10-30 min",
    "30 min-2 h",
    ">=2 h",
]


episodes[
    "duration_band"
] = pd.cut(
    episodes[
        "duration_seconds"
    ],
    bins=bins,
    labels=labels,
    right=False,
)


band_rows = []


for participant_id, x in episodes.groupby(
    "participant_id",
    sort=True,
):

    total_time = (
        x[
            "duration_ms"
        ].sum()
    )

    for label in labels:

        y = x[
            x[
                "duration_band"
            ]
            == label
        ]

        band_rows.append({
            "participant_id":
                participant_id,

            "duration_band":
                label,

            "episode_n":
                len(y),

            "episode_fraction":
                (
                    len(y)
                    / len(x)
                ),

            "time_fraction":
                (
                    y[
                        "duration_ms"
                    ].sum()
                    / total_time
                    if total_time > 0
                    else np.nan
                ),
        })


duration_bands = pd.DataFrame(
    band_rows
)

duration_bands.to_csv(
    DURATION_OUT,
    index=False,
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 80)
print("PARTICIPANT ANCHOR-EPISODE SUMMARY")
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
print("POOLED ANCHOR-EPISODE QUANTILES")
print("=" * 80)

print(
    quantiles.pivot(
        index="quantile",
        columns="metric",
        values="value",
    )
    .reset_index()
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:.3f}",
    )
)


print()
print("=" * 80)
print("ANCHOR-EPISODE DURATION BANDS")
print("=" * 80)

print(
    duration_bands.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print()
print("=" * 80)
print("OUTPUTS")
print("=" * 80)

print(EPISODE_OUT)
print(PARTICIPANT_OUT)
print(QUANTILE_OUT)
print(DURATION_OUT)

print()
print("=" * 80)
print("DONE")
print("=" * 80)
