"""Characterize Cortex 2.0 location-compatible GPS episodes.

Uses the previously generated high-observability reference edges.

No Cortex feature functions are used.

The goal is to characterize duration and spatial spread before defining
cross-episode location identity, home inference, or location entropy.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cortex2_engine.features.gps_location_episodes import (
    build_location_compatible_episodes,
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
    / "gps_reference_location_compatible_episodes.parquet"
)

PARTICIPANT_OUT = (
    OUT_DIR
    / "gps_reference_location_episode_participants.csv"
)

QUANTILE_OUT = (
    OUT_DIR
    / "gps_reference_location_episode_quantiles.csv"
)

DURATION_BAND_OUT = (
    OUT_DIR
    / "gps_reference_location_episode_duration_bands.csv"
)

DAILY_OUT = (
    OUT_DIR
    / "gps_reference_location_episode_daily.csv"
)


# ============================================================
# LOAD EDGES
# ============================================================

edges = pd.read_parquet(
    EDGE_PATH
)

print("=" * 80)
print("Cortex 2.0 LOCATION-COMPATIBLE EPISODE AUDIT")
print("=" * 80)

print(
    "Supported edges:",
    f"{len(edges):,}",
)

print(
    "Participants:",
    edges[
        "participant_id"
    ].nunique(),
)

print(
    "Participant-days:",
    edges[
        [
            "participant_id",
            "date",
        ]
    ]
    .drop_duplicates()
    .shape[0],
)


# ============================================================
# ADD SPATIAL SEMANTICS
# ============================================================

edges = add_gps_spatial_semantics(
    edges
)


status_counts = (
    edges[
        "spatial_status"
    ]
    .value_counts()
)


print()
print("Spatial status:")

for status, count in status_counts.items():
    print(
        f"  {status}: {count:,} "
        f"({count / len(edges):.2%})"
    )


# ============================================================
# BUILD EPISODES BY PARTICIPANT-DAY
# ============================================================

episode_frames = []
daily_rows = []


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
        day_edges.sort_values(
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
        build_location_compatible_episodes(
            day_edges
        )
    )


    compatible_edge_duration_ms = (
        day_edges.loc[
            day_edges[
                "spatial_status"
            ]
            == "accuracy_overlap",
            "duration_ms",
        ]
        .sum()
    )


    supported_duration_ms = (
        day_edges[
            "duration_ms"
        ]
        .sum()
    )


    episode_duration_ms = (
        episodes[
            "duration_ms"
        ]
        .sum()
        if len(
            episodes
        )
        else 0
    )


    daily_rows.append({
        "participant_id":
            participant_id,

        "mindlamp_uid":
            uid,

        "date":
            date,

        "supported_edges":
            len(
                day_edges
            ),

        "supported_duration_ms":
            supported_duration_ms,

        "location_compatible_edges":
            int(
                (
                    day_edges[
                        "spatial_status"
                    ]
                    == "accuracy_overlap"
                )
                .sum()
            ),

        "location_compatible_duration_ms":
            compatible_edge_duration_ms,

        "location_compatible_duration_fraction":
            (
                compatible_edge_duration_ms
                / supported_duration_ms
                if supported_duration_ms > 0
                else np.nan
            ),

        "n_location_episodes":
            len(
                episodes
            ),

        "episode_duration_ms":
            episode_duration_ms,
    })


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
        "No location-compatible episodes generated."
    )


episodes = pd.concat(
    episode_frames,
    ignore_index=True,
)

daily = pd.DataFrame(
    daily_rows
)


episodes.to_parquet(
    EPISODE_OUT,
    index=False,
)

daily.to_csv(
    DAILY_OUT,
    index=False,
)


# ============================================================
# DERIVED UNITS
# ============================================================

episodes[
    "duration_seconds"
] = (
    episodes[
        "duration_ms"
    ]
    / 1000
)

episodes[
    "duration_minutes"
] = (
    episodes[
        "duration_ms"
    ]
    / 60_000
)


# ============================================================
# PARTICIPANT SUMMARY
# ============================================================

participant_rows = []


for participant_id, x in episodes.groupby(
    "participant_id",
    sort=True,
):

    d = daily[
        daily[
            "participant_id"
        ]
        == participant_id
    ]


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
            (
                d[
                    "n_location_episodes"
                ]
                .median()
            ),

        "median_duration_seconds":
            x[
                "duration_seconds"
            ]
            .median(),

        "p90_duration_seconds":
            x[
                "duration_seconds"
            ]
            .quantile(
                0.90
            ),

        "p99_duration_seconds":
            x[
                "duration_seconds"
            ]
            .quantile(
                0.99
            ),

        "median_radius_m":
            x[
                "max_radius_from_representative_m"
            ]
            .median(),

        "p90_radius_m":
            x[
                "max_radius_from_representative_m"
            ]
            .quantile(
                0.90
            ),

        "p99_radius_m":
            x[
                "max_radius_from_representative_m"
            ]
            .quantile(
                0.99
            ),

        "median_path_distance_m":
            x[
                "path_distance_m"
            ]
            .median(),

        "median_compatible_time_fraction":
            d[
                "location_compatible_duration_fraction"
            ]
            .median(),
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

    "max_radius_m":
        episodes[
            "max_radius_from_representative_m"
        ],

    "path_distance_m":
        episodes[
            "path_distance_m"
        ],

    "median_accuracy_m":
        episodes[
            "median_endpoint_accuracy_m"
        ],
}


quantile_rows = []


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

        quantile_rows.append({
            "metric":
                metric,

            "quantile":
                q,

            "value":
                values.quantile(
                    q
                ),

            "n":
                len(
                    values
                ),
        })


quantiles = pd.DataFrame(
    quantile_rows
)

quantiles.to_csv(
    QUANTILE_OUT,
    index=False,
)


# ============================================================
# DURATION BANDS
# ============================================================

duration_bins = [
    -np.inf,
    5,
    30,
    120,
    600,
    1800,
    7200,
    np.inf,
]

duration_labels = [
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
    bins=duration_bins,
    labels=duration_labels,
    right=False,
)


band_rows = []


for participant_id, x in episodes.groupby(
    "participant_id",
    sort=True,
):

    counts = (
        x[
            "duration_band"
        ]
        .value_counts(
            sort=False
        )
    )


    duration_by_band = (
        x.groupby(
            "duration_band",
            observed=True,
        )[
            "duration_ms"
        ]
        .sum()
    )


    total_duration = (
        x[
            "duration_ms"
        ]
        .sum()
    )


    for band in duration_labels:

        band_rows.append({
            "participant_id":
                participant_id,

            "duration_band":
                band,

            "episode_n":
                int(
                    counts.get(
                        band,
                        0,
                    )
                ),

            "episode_fraction":
                (
                    counts.get(
                        band,
                        0,
                    )
                    / len(x)
                ),

            "time_fraction":
                (
                    duration_by_band.get(
                        band,
                        0,
                    )
                    / total_duration
                    if total_duration
                    else np.nan
                ),
        })


duration_bands = pd.DataFrame(
    band_rows
)

duration_bands.to_csv(
    DURATION_BAND_OUT,
    index=False,
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 80)
print("PARTICIPANT EPISODE SUMMARY")
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
print("POOLED EPISODE QUANTILES")
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
print("EPISODE DURATION BANDS")
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
print(DURATION_BAND_OUT)
print(DAILY_OUT)

print()
print("=" * 80)
print("DONE")
print("=" * 80)
