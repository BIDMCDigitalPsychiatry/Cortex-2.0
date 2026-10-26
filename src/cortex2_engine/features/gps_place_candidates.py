"""Uncertainty-scaled Cortex 2.0 GPS place candidates.

Input episodes are spatially coherent fixed-anchor microsegments.

Microsegments are clustered into participant-level candidate places using
duration-prioritized fixed seeds.

This prevents two failure modes:

1. pointwise pairwise-overlap chaining across large distances;
2. excessive fragmentation caused by anchoring every location episode to
   one raw GPS observation.

For microsegment i:

    support_radius_i =
        anchor_accuracy_i
        + max_anchor_distance_i

Two microsegments are compatible when:

    distance(seed, segment)
        <= compatibility_multiplier
           * (support_radius_seed + support_radius_segment)

The seed remains fixed after cluster creation. Therefore membership cannot
propagate through transitive pairwise chaining.

These are candidate spatial states, not automatically home, work, visits,
or clinically meaningful locations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


EARTH_RADIUS_M = 6_371_008.8


def _haversine_m_vectorized(
    lat: float,
    lon: float,
    other_lat: np.ndarray,
    other_lon: np.ndarray,
) -> np.ndarray:
    """Vectorized great-circle distance in meters."""

    lat1 = np.radians(
        float(lat)
    )

    lon1 = np.radians(
        float(lon)
    )

    lat2 = np.radians(
        other_lat.astype(float)
    )

    lon2 = np.radians(
        other_lon.astype(float)
    )

    dlat = (
        lat2
        - lat1
    )

    dlon = (
        lon2
        - lon1
    )

    a = (
        np.sin(
            dlat / 2
        ) ** 2
        + np.cos(
            lat1
        )
        * np.cos(
            lat2
        )
        * np.sin(
            dlon / 2
        ) ** 2
    )

    a = np.clip(
        a,
        0.0,
        1.0,
    )

    return (
        2
        * EARTH_RADIUS_M
        * np.arctan2(
            np.sqrt(
                a
            ),
            np.sqrt(
                1 - a
            ),
        )
    )


def build_place_candidates(
    episodes: pd.DataFrame,
    *,
    compatibility_multiplier: float = 1.0,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Cluster spatially coherent microsegments into candidate places.

    Parameters
    ----------
    episodes
        Fixed-anchor location microsegments.

    compatibility_multiplier
        Explicit uncertainty-scale multiplier.

        A value of 1 means that the two microsegment support regions must
        geometrically overlap under the Cortex 2.0 support-radius definition.

        This is a model parameter, not a universal DQ threshold.

    Returns
    -------
    assignments
        One row per input microsegment with candidate-place membership.

    places
        One row per resulting candidate place.
    """

    if (
        not np.isfinite(
            compatibility_multiplier
        )
        or compatibility_multiplier
        <= 0
    ):
        raise ValueError(
            "compatibility_multiplier must be > 0"
        )


    required = {
        "episode_start_ms",
        "episode_end_ms",
        "duration_ms",
        "anchor_latitude",
        "anchor_longitude",
        "anchor_accuracy_m",
        "representative_latitude",
        "representative_longitude",
        "max_anchor_distance_m",
    }

    missing = (
        required
        - set(
            episodes.columns
        )
    )

    if missing:
        raise ValueError(
            "Missing required microsegment columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )


    if episodes.empty:
        assignments = episodes.copy()

        assignments[
            "place_candidate_id"
        ] = pd.Series(
            dtype="int64"
        )

        assignments[
            "support_radius_m"
        ] = pd.Series(
            dtype="float64"
        )

        assignments[
            "seed_distance_m"
        ] = pd.Series(
            dtype="float64"
        )

        assignments[
            "seed_normalized_distance"
        ] = pd.Series(
            dtype="float64"
        )

        places = pd.DataFrame(
            columns=[
                "place_candidate_id",
                "seed_episode_start_ms",
                "seed_latitude",
                "seed_longitude",
                "seed_support_radius_m",
                "n_microsegments",
                "total_duration_ms",
                "representative_latitude",
                "representative_longitude",
                "max_seed_distance_m",
                "max_seed_normalized_distance",
            ]
        )

        return (
            assignments,
            places,
        )


    x = episodes.copy()

    x[
        "_original_order"
    ] = np.arange(
        len(x)
    )


    numeric_columns = [
        "episode_start_ms",
        "episode_end_ms",
        "duration_ms",
        "anchor_latitude",
        "anchor_longitude",
        "anchor_accuracy_m",
        "representative_latitude",
        "representative_longitude",
        "max_anchor_distance_m",
    ]


    for column in numeric_columns:

        x[column] = pd.to_numeric(
            x[column],
            errors="coerce",
        )


    invalid = (
        x[
            numeric_columns
        ]
        .isna()
        .any(
            axis=1
        )
        | ~np.isfinite(
            x[
                numeric_columns
            ]
        ).all(
            axis=1
        )
        | (
            x[
                "duration_ms"
            ]
            <= 0
        )
        | (
            x[
                "anchor_accuracy_m"
            ]
            <= 0
        )
        | (
            x[
                "max_anchor_distance_m"
            ]
            < 0
        )
    )


    if invalid.any():
        raise ValueError(
            "Place-candidate input contains "
            f"{int(invalid.sum())} invalid microsegments."
        )


    x[
        "support_radius_m"
    ] = (
        x[
            "anchor_accuracy_m"
        ]
        + x[
            "max_anchor_distance_m"
        ]
    )


    # Long, stable-duration microsegments define candidate-place
    # seeds before short jitter-driven fragments.
    ordered = (
        x.sort_values(
            [
                "duration_ms",
                "episode_start_ms",
                "_original_order",
            ],
            ascending=[
                False,
                True,
                True,
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )


    seed_latitudes = []
    seed_longitudes = []
    seed_radii = []
    seed_episode_starts = []


    assigned_place = np.empty(
        len(ordered),
        dtype=np.int64,
    )

    seed_distance = np.empty(
        len(ordered),
        dtype=float,
    )

    normalized_distance = np.empty(
        len(ordered),
        dtype=float,
    )


    for position, row in enumerate(
        ordered.itertuples(
            index=False
        )
    ):

        latitude = float(
            row.anchor_latitude
        )

        longitude = float(
            row.anchor_longitude
        )

        radius = float(
            row.support_radius_m
        )


        if not seed_latitudes:

            seed_latitudes.append(
                latitude
            )

            seed_longitudes.append(
                longitude
            )

            seed_radii.append(
                radius
            )

            seed_episode_starts.append(
                int(
                    row.episode_start_ms
                )
            )

            assigned_place[
                position
            ] = 0

            seed_distance[
                position
            ] = 0.0

            normalized_distance[
                position
            ] = 0.0

            continue


        seed_lat = np.asarray(
            seed_latitudes,
            dtype=float,
        )

        seed_lon = np.asarray(
            seed_longitudes,
            dtype=float,
        )

        seed_radius = np.asarray(
            seed_radii,
            dtype=float,
        )


        distances = (
            _haversine_m_vectorized(
                latitude,
                longitude,
                seed_lat,
                seed_lon,
            )
        )


        denominator = (
            seed_radius
            + radius
        )


        normalized = (
            distances
            / denominator
        )


        compatible = (
            normalized
            <= compatibility_multiplier
        )


        if compatible.any():

            candidate_indices = np.flatnonzero(
                compatible
            )

            best_local = (
                candidate_indices[
                    np.argmin(
                        normalized[
                            candidate_indices
                        ]
                    )
                ]
            )

            place_id = int(
                best_local
            )

            assigned_place[
                position
            ] = place_id

            seed_distance[
                position
            ] = distances[
                place_id
            ]

            normalized_distance[
                position
            ] = normalized[
                place_id
            ]

        else:

            place_id = len(
                seed_latitudes
            )

            seed_latitudes.append(
                latitude
            )

            seed_longitudes.append(
                longitude
            )

            seed_radii.append(
                radius
            )

            seed_episode_starts.append(
                int(
                    row.episode_start_ms
                )
            )

            assigned_place[
                position
            ] = place_id

            seed_distance[
                position
            ] = 0.0

            normalized_distance[
                position
            ] = 0.0


    ordered[
        "place_candidate_id"
    ] = assigned_place

    ordered[
        "seed_distance_m"
    ] = seed_distance

    ordered[
        "seed_normalized_distance"
    ] = normalized_distance


    # ========================================================
    # PLACE SUMMARIES
    # ========================================================

    place_rows = []


    for place_id, group in ordered.groupby(
        "place_candidate_id",
        sort=True,
    ):

        durations = (
            group[
                "duration_ms"
            ]
            .to_numpy(
                dtype=float
            )
        )

        total_duration = (
            durations.sum()
        )


        representative_latitude = (
            np.average(
                group[
                    "representative_latitude"
                ].to_numpy(
                    dtype=float
                ),
                weights=durations,
            )
        )

        representative_longitude = (
            np.average(
                group[
                    "representative_longitude"
                ].to_numpy(
                    dtype=float
                ),
                weights=durations,
            )
        )


        place_rows.append({
            "place_candidate_id":
                int(
                    place_id
                ),

            "seed_episode_start_ms":
                seed_episode_starts[
                    place_id
                ],

            "seed_latitude":
                seed_latitudes[
                    place_id
                ],

            "seed_longitude":
                seed_longitudes[
                    place_id
                ],

            "seed_support_radius_m":
                seed_radii[
                    place_id
                ],

            "n_microsegments":
                len(
                    group
                ),

            "total_duration_ms":
                float(
                    total_duration
                ),

            "representative_latitude":
                float(
                    representative_latitude
                ),

            "representative_longitude":
                float(
                    representative_longitude
                ),

            "max_seed_distance_m":
                float(
                    group[
                        "seed_distance_m"
                    ].max()
                ),

            "max_seed_normalized_distance":
                float(
                    group[
                        "seed_normalized_distance"
                    ].max()
                ),
        })


    places = pd.DataFrame(
        place_rows
    )


    assignments = (
        ordered.sort_values(
            "_original_order",
            kind="mergesort",
        )
        .drop(
            columns=[
                "_original_order",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    return (
        assignments,
        places,
    )
