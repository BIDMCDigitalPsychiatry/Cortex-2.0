"""Anchor-compatible GPS location episodes.

Sequential pairwise spatial compatibility can chain over large distances:

    A compatible with B
    B compatible with C

does not imply:

    A compatible with C

RAPHE therefore defines an anchor-compatible episode as a temporally
contiguous run of accuracy-overlap edges in which every newly observed
endpoint remains compatible with one fixed episode anchor.

For anchor accuracy a0 and observation accuracy ai:

    distance(anchor, observation) <= a0 + ai

This introduces no fixed distance threshold.

The resulting episodes remain sensor-evidence constructs. They are not
automatically interpreted as home, stationary behavior, visits, or
significant locations.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from raphe.features.gps_intervals import (
    haversine_km,
)

from raphe.features.gps_spatial_semantics import (
    SPATIAL_STATUS_ACCURACY_OVERLAP,
)


OUTPUT_COLUMNS = [
    "episode_start_ms",
    "episode_end_ms",
    "duration_ms",
    "n_edges",
    "n_observations",
    "anchor_latitude",
    "anchor_longitude",
    "anchor_accuracy_m",
    "representative_latitude",
    "representative_longitude",
    "path_distance_m",
    "max_edge_distance_m",
    "max_anchor_distance_m",
    "max_anchor_ratio",
    "max_endpoint_accuracy_m",
]


def _valid_number(
    value,
) -> bool:
    try:
        return math.isfinite(
            float(value)
        )
    except (
        TypeError,
        ValueError,
    ):
        return False


def build_anchor_compatible_location_episodes(
    edges: pd.DataFrame,
) -> pd.DataFrame:
    """Build spatially coherent GPS episodes using a fixed anchor.

    Parameters
    ----------
    edges
        GPS edge dataframe after spatial semantics have been added.

    Returns
    -------
    pandas.DataFrame
        One row per anchor-compatible episode.

    Notes
    -----
    An edge is eligible only when:

    - it is temporally supported, and
    - spatial_status == "accuracy_overlap".

    The first observation of an episode becomes its fixed anchor.

    A subsequent edge can remain in the episode only when its new endpoint
    is also compatible with that anchor:

        anchor_distance <= anchor_accuracy + endpoint_accuracy

    Any temporal discontinuity, unsupported edge, displacement-evidence edge,
    or failure of anchor compatibility terminates the current episode.
    """

    required = {
        "start_ms",
        "end_ms",
        "duration_ms",
        "supported",
        "spatial_status",
        "start_latitude",
        "start_longitude",
        "end_latitude",
        "end_longitude",
        "distance_km",
        "start_accuracy",
        "end_accuracy",
    }

    missing = (
        required
        - set(
            edges.columns
        )
    )

    if missing:
        raise ValueError(
            "Missing required GPS edge columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    if edges.empty:
        return pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

    x = edges.copy()

    numeric_columns = [
        "start_ms",
        "end_ms",
        "duration_ms",
        "start_latitude",
        "start_longitude",
        "end_latitude",
        "end_longitude",
        "distance_km",
        "start_accuracy",
        "end_accuracy",
    ]

    for column in numeric_columns:
        x[column] = pd.to_numeric(
            x[column],
            errors="coerce",
        )

    x = (
        x.dropna(
            subset=[
                "start_ms",
                "end_ms",
            ]
        )
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

    rows = []

    current = None
    previous_end_ms = None


    def finish_current():
        nonlocal current

        if current is None:
            return

        if (
            current[
                "weighted_duration_ms"
            ]
            > 0
        ):
            representative_latitude = (
                current[
                    "weighted_latitude_sum"
                ]
                / current[
                    "weighted_duration_ms"
                ]
            )

            representative_longitude = (
                current[
                    "weighted_longitude_sum"
                ]
                / current[
                    "weighted_duration_ms"
                ]
            )

        else:
            representative_latitude = np.nan
            representative_longitude = np.nan

        rows.append({
            "episode_start_ms":
                int(
                    current[
                        "episode_start_ms"
                    ]
                ),

            "episode_end_ms":
                int(
                    current[
                        "episode_end_ms"
                    ]
                ),

            "duration_ms":
                int(
                    current[
                        "duration_ms"
                    ]
                ),

            "n_edges":
                int(
                    current[
                        "n_edges"
                    ]
                ),

            "n_observations":
                int(
                    current[
                        "n_edges"
                    ]
                    + 1
                ),

            "anchor_latitude":
                current[
                    "anchor_latitude"
                ],

            "anchor_longitude":
                current[
                    "anchor_longitude"
                ],

            "anchor_accuracy_m":
                current[
                    "anchor_accuracy_m"
                ],

            "representative_latitude":
                representative_latitude,

            "representative_longitude":
                representative_longitude,

            "path_distance_m":
                current[
                    "path_distance_m"
                ],

            "max_edge_distance_m":
                current[
                    "max_edge_distance_m"
                ],

            "max_anchor_distance_m":
                current[
                    "max_anchor_distance_m"
                ],

            "max_anchor_ratio":
                current[
                    "max_anchor_ratio"
                ],

            "max_endpoint_accuracy_m":
                current[
                    "max_endpoint_accuracy_m"
                ],
        })

        current = None


    def start_episode(
        row,
    ):
        start_lat = float(
            row.start_latitude
        )

        start_lon = float(
            row.start_longitude
        )

        end_lat = float(
            row.end_latitude
        )

        end_lon = float(
            row.end_longitude
        )

        anchor_accuracy = float(
            row.start_accuracy
        )

        end_accuracy = float(
            row.end_accuracy
        )

        edge_distance_m = (
            float(
                row.distance_km
            )
            * 1000
        )

        anchor_distance_m = (
            haversine_km(
                start_lat,
                start_lon,
                end_lat,
                end_lon,
            )
            * 1000
        )

        denominator = (
            anchor_accuracy
            + end_accuracy
        )

        anchor_ratio = (
            anchor_distance_m
            / denominator
            if denominator > 0
            else np.nan
        )

        duration_ms = float(
            row.duration_ms
        )

        midpoint_lat = (
            start_lat
            + end_lat
        ) / 2

        midpoint_lon = (
            start_lon
            + end_lon
        ) / 2

        return {
            "episode_start_ms":
                int(
                    row.start_ms
                ),

            "episode_end_ms":
                int(
                    row.end_ms
                ),

            "duration_ms":
                duration_ms,

            "n_edges":
                1,

            "anchor_latitude":
                start_lat,

            "anchor_longitude":
                start_lon,

            "anchor_accuracy_m":
                anchor_accuracy,

            "weighted_latitude_sum":
                (
                    midpoint_lat
                    * duration_ms
                ),

            "weighted_longitude_sum":
                (
                    midpoint_lon
                    * duration_ms
                ),

            "weighted_duration_ms":
                duration_ms,

            "path_distance_m":
                edge_distance_m,

            "max_edge_distance_m":
                edge_distance_m,

            "max_anchor_distance_m":
                anchor_distance_m,

            "max_anchor_ratio":
                anchor_ratio,

            "max_endpoint_accuracy_m":
                max(
                    anchor_accuracy,
                    end_accuracy,
                ),
        }


    for row in x.itertuples(
        index=False
    ):

        supported = bool(
            row.supported
        )

        eligible = (
            supported
            and row.spatial_status
            == SPATIAL_STATUS_ACCURACY_OVERLAP
            and _valid_number(
                row.start_latitude
            )
            and _valid_number(
                row.start_longitude
            )
            and _valid_number(
                row.end_latitude
            )
            and _valid_number(
                row.end_longitude
            )
            and _valid_number(
                row.start_accuracy
            )
            and _valid_number(
                row.end_accuracy
            )
            and float(
                row.start_accuracy
            )
            > 0
            and float(
                row.end_accuracy
            )
            > 0
            and _valid_number(
                row.distance_km
            )
            and _valid_number(
                row.duration_ms
            )
            and float(
                row.duration_ms
            )
            > 0
        )

        if not eligible:
            finish_current()
            previous_end_ms = int(
                row.end_ms
            )
            continue

        start_ms = int(
            row.start_ms
        )

        end_ms = int(
            row.end_ms
        )

        if (
            current is None
            or previous_end_ms is None
            or start_ms
            != previous_end_ms
        ):
            finish_current()

            current = start_episode(
                row
            )

            previous_end_ms = end_ms
            continue


        anchor_distance_m = (
            haversine_km(
                current[
                    "anchor_latitude"
                ],
                current[
                    "anchor_longitude"
                ],
                float(
                    row.end_latitude
                ),
                float(
                    row.end_longitude
                ),
            )
            * 1000
        )

        combined_accuracy_m = (
            current[
                "anchor_accuracy_m"
            ]
            + float(
                row.end_accuracy
            )
        )

        anchor_ratio = (
            anchor_distance_m
            / combined_accuracy_m
        )


        # Pairwise compatibility with the previous point is already
        # represented by spatial_status. Here we additionally require
        # compatibility with the fixed episode anchor.
        if anchor_ratio > 1.0:

            finish_current()

            current = start_episode(
                row
            )

            previous_end_ms = end_ms
            continue


        duration_ms = float(
            row.duration_ms
        )

        midpoint_lat = (
            float(
                row.start_latitude
            )
            + float(
                row.end_latitude
            )
        ) / 2

        midpoint_lon = (
            float(
                row.start_longitude
            )
            + float(
                row.end_longitude
            )
        ) / 2

        edge_distance_m = (
            float(
                row.distance_km
            )
            * 1000
        )


        current[
            "episode_end_ms"
        ] = end_ms

        current[
            "duration_ms"
        ] += duration_ms

        current[
            "n_edges"
        ] += 1

        current[
            "weighted_latitude_sum"
        ] += (
            midpoint_lat
            * duration_ms
        )

        current[
            "weighted_longitude_sum"
        ] += (
            midpoint_lon
            * duration_ms
        )

        current[
            "weighted_duration_ms"
        ] += duration_ms

        current[
            "path_distance_m"
        ] += edge_distance_m

        current[
            "max_edge_distance_m"
        ] = max(
            current[
                "max_edge_distance_m"
            ],
            edge_distance_m,
        )

        current[
            "max_anchor_distance_m"
        ] = max(
            current[
                "max_anchor_distance_m"
            ],
            anchor_distance_m,
        )

        current[
            "max_anchor_ratio"
        ] = max(
            current[
                "max_anchor_ratio"
            ],
            anchor_ratio,
        )

        current[
            "max_endpoint_accuracy_m"
        ] = max(
            current[
                "max_endpoint_accuracy_m"
            ],
            float(
                row.end_accuracy
            ),
        )

        previous_end_ms = end_ms


    finish_current()


    return pd.DataFrame(
        rows,
        columns=OUTPUT_COLUMNS,
    )
