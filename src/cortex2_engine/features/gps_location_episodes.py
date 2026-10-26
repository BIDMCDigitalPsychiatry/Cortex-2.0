"""Construct contiguous location-compatible GPS episodes.

A location-compatible episode is a consecutive run of GPS edges that:

1. are temporally supported; and
2. have spatial_status == "accuracy_overlap".

This is SENSOR EVIDENCE, not a behavioral classification.

An episode is not automatically:
- stationary time,
- home time,
- a visit,
- a significant location.

Those interpretations occur in later Cortex 2.0 layers.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from cortex2_engine.features.gps_intervals import (
    haversine_km,
)

from cortex2_engine.features.gps_spatial_semantics import (
    SPATIAL_STATUS_ACCURACY_OVERLAP,
)


OUTPUT_COLUMNS = [
    "episode_start_ms",
    "episode_end_ms",
    "duration_ms",
    "n_edges",
    "representative_latitude",
    "representative_longitude",
    "path_distance_m",
    "max_edge_distance_m",
    "median_endpoint_accuracy_m",
    "max_endpoint_accuracy_m",
    "max_radius_from_representative_m",
]


def _validate_columns(
    edges: pd.DataFrame,
) -> None:

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


def _summarize_episode(
    episode_edges: pd.DataFrame,
) -> dict:
    """Summarize one contiguous location-compatible edge run."""

    durations = (
        pd.to_numeric(
            episode_edges[
                "duration_ms"
            ],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(
            dtype=float
        )
    )

    start_lat = pd.to_numeric(
        episode_edges[
            "start_latitude"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    start_lon = pd.to_numeric(
        episode_edges[
            "start_longitude"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    end_lat = pd.to_numeric(
        episode_edges[
            "end_latitude"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    end_lon = pd.to_numeric(
        episode_edges[
            "end_longitude"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )


    # Represent each edge by its geographic midpoint.
    midpoint_lat = (
        start_lat
        + end_lat
    ) / 2

    midpoint_lon = (
        start_lon
        + end_lon
    ) / 2


    valid_midpoint = (
        np.isfinite(
            midpoint_lat
        )
        & np.isfinite(
            midpoint_lon
        )
        & np.isfinite(
            durations
        )
        & (
            durations
            > 0
        )
    )


    if valid_midpoint.any():

        weights = durations[
            valid_midpoint
        ]

        representative_latitude = float(
            np.average(
                midpoint_lat[
                    valid_midpoint
                ],
                weights=weights,
            )
        )

        representative_longitude = float(
            np.average(
                midpoint_lon[
                    valid_midpoint
                ],
                weights=weights,
            )
        )

    else:

        representative_latitude = np.nan
        representative_longitude = np.nan


    distance_km = pd.to_numeric(
        episode_edges[
            "distance_km"
        ],
        errors="coerce",
    )


    path_distance_m = (
        distance_km
        .fillna(0)
        .sum()
        * 1000
    )


    max_edge_distance_m = (
        distance_km
        .max()
        * 1000
        if distance_km.notna().any()
        else np.nan
    )


    accuracy_values = pd.concat(
        [
            pd.to_numeric(
                episode_edges[
                    "start_accuracy"
                ],
                errors="coerce",
            ),
            pd.to_numeric(
                episode_edges[
                    "end_accuracy"
                ],
                errors="coerce",
            ),
        ],
        ignore_index=True,
    )

    accuracy_values = (
        accuracy_values[
            accuracy_values.notna()
            & np.isfinite(
                accuracy_values
            )
            & (
                accuracy_values
                > 0
            )
        ]
    )


    median_accuracy = (
        float(
            accuracy_values.median()
        )
        if len(
            accuracy_values
        )
        else np.nan
    )

    max_accuracy = (
        float(
            accuracy_values.max()
        )
        if len(
            accuracy_values
        )
        else np.nan
    )


    endpoint_coordinates = []

    if len(
        episode_edges
    ):

        first = (
            episode_edges.iloc[0]
        )

        endpoint_coordinates.append(
            (
                float(
                    first[
                        "start_latitude"
                    ]
                ),
                float(
                    first[
                        "start_longitude"
                    ]
                ),
            )
        )

        for row in episode_edges.itertuples(
            index=False
        ):

            endpoint_coordinates.append(
                (
                    float(
                        row.end_latitude
                    ),
                    float(
                        row.end_longitude
                    ),
                )
            )


    radii = []

    if (
        math.isfinite(
            representative_latitude
        )
        and math.isfinite(
            representative_longitude
        )
    ):

        for lat, lon in endpoint_coordinates:

            if not (
                math.isfinite(
                    lat
                )
                and math.isfinite(
                    lon
                )
            ):
                continue

            radii.append(
                haversine_km(
                    representative_latitude,
                    representative_longitude,
                    lat,
                    lon,
                )
                * 1000
            )


    max_radius = (
        max(
            radii
        )
        if radii
        else np.nan
    )


    start_ms = int(
        episode_edges[
            "start_ms"
        ].iloc[0]
    )

    end_ms = int(
        episode_edges[
            "end_ms"
        ].iloc[-1]
    )


    return {
        "episode_start_ms":
            start_ms,

        "episode_end_ms":
            end_ms,

        "duration_ms":
            int(
                episode_edges[
                    "duration_ms"
                ].sum()
            ),

        "n_edges":
            len(
                episode_edges
            ),

        "representative_latitude":
            representative_latitude,

        "representative_longitude":
            representative_longitude,

        "path_distance_m":
            float(
                path_distance_m
            ),

        "max_edge_distance_m":
            (
                float(
                    max_edge_distance_m
                )
                if pd.notna(
                    max_edge_distance_m
                )
                else np.nan
            ),

        "median_endpoint_accuracy_m":
            median_accuracy,

        "max_endpoint_accuracy_m":
            max_accuracy,

        "max_radius_from_representative_m":
            (
                float(
                    max_radius
                )
                if pd.notna(
                    max_radius
                )
                else np.nan
            ),
    }


def build_location_compatible_episodes(
    edges: pd.DataFrame,
) -> pd.DataFrame:
    """Group contiguous accuracy-overlap edges into episodes."""

    _validate_columns(
        edges
    )

    if edges.empty:
        return pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )


    x = edges.copy()

    x["start_ms"] = pd.to_numeric(
        x["start_ms"],
        errors="coerce",
    )

    x["end_ms"] = pd.to_numeric(
        x["end_ms"],
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


    eligible = x[
        x["supported"]
        .fillna(
            False
        )
        .astype(
            bool
        )
        & (
            x[
                "spatial_status"
            ]
            == (
                SPATIAL_STATUS_ACCURACY_OVERLAP
            )
        )
    ].copy()


    if eligible.empty:
        return pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )


    episode_groups = []
    current_indices = []


    previous_end = None


    for idx, row in eligible.iterrows():

        start = int(
            row[
                "start_ms"
            ]
        )

        end = int(
            row[
                "end_ms"
            ]
        )


        # Exact temporal adjacency is required.
        # A displacement edge or temporal gap breaks the episode.
        if (
            previous_end is None
            or start
            == previous_end
        ):

            current_indices.append(
                idx
            )

        else:

            episode_groups.append(
                current_indices
            )

            current_indices = [
                idx
            ]


        previous_end = end


    if current_indices:
        episode_groups.append(
            current_indices
        )


    rows = []

    for indices in episode_groups:

        episode_edges = (
            eligible.loc[
                indices
            ]
            .sort_values(
                "start_ms"
            )
        )

        rows.append(
            _summarize_episode(
                episode_edges
            )
        )


    return pd.DataFrame(
        rows,
        columns=OUTPUT_COLUMNS,
    )
