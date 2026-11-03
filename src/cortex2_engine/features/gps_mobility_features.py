"""Daily GPS mobility features for Cortex 2.0.

These features are computed from temporally supported GPS edges after
accuracy-aware spatial semantics have been added.

They intentionally remain sensor-evidence summaries. They do not infer
walking, driving, home, work, or other behavioral states.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from cortex2_engine.features.gps_intervals import (
    haversine_km,
)
from cortex2_engine.features.gps_trajectory_plausibility import (
    TRAJECTORY_STATUS_IMPLAUSIBLE_SPEED,
    TRAJECTORY_STATUS_PLAUSIBLE_DISPLACEMENT,
)
from cortex2_engine.features.status import (
    gated_value,
    status_from_counts,
)


def compute_daily_mobility_features(
    edges: pd.DataFrame,
    *,
    minimum_supported_duration_ms: float = 0.0,
) -> dict:
    """Compute conservative daily GPS mobility summaries.

    Parameters
    ----------
    edges
        Output from ``add_gps_spatial_semantics``.

        Required columns:
        - supported
        - duration_ms
        - start_latitude
        - start_longitude
        - end_latitude
        - end_longitude
        - distance_m
        - trajectory_status

    minimum_supported_duration_ms
        Minimum spatially supported time for the distance, radius and
        implausible-fraction values to be interpreted. Default 0 keeps the
        historical behaviour of accepting any positive supported time.

    Returns
    -------
    dict
        - supported_displacement_distance_m (+ ``_status``)
        - supported_radius_of_gyration_m (+ ``_status``)
        - implausible_transition_fraction (+ ``_status``)
        - displacement_edge_count, implausible_transition_count
          (evidence counts, always reported)
        - spatially_supported_duration_ms

    Notes
    -----
    ``supported_displacement_distance_m`` sums observed distance only for
    temporally supported edges whose displacement exceeds the summed
    endpoint accuracy radii.

    This is intentionally more conservative than summing all consecutive
    GPS distances.

    ``supported_radius_of_gyration_m`` measures duration-weighted spatial
    dispersion of supported GPS edge midpoints around their duration-weighted
    geographic center.
    """

    required = {
        "supported",
        "duration_ms",
        "start_latitude",
        "start_longitude",
        "end_latitude",
        "end_longitude",
        "distance_m",
        "trajectory_status",
    }

    missing = (
        required
        - set(edges.columns)
    )

    if missing:
        raise ValueError(
            "Missing required GPS columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    x = edges.copy()

    numeric_columns = [
        "duration_ms",
        "start_latitude",
        "start_longitude",
        "end_latitude",
        "end_longitude",
        "distance_m",
    ]

    for column in numeric_columns:
        x[column] = pd.to_numeric(
            x[column],
            errors="coerce",
        )

    supported = (
        x["supported"]
        .fillna(False)
        .astype(bool)
    )

    plausible_displacement = (
        supported
        & x["trajectory_status"].eq(
            TRAJECTORY_STATUS_PLAUSIBLE_DISPLACEMENT
        )
        & x["distance_m"].notna()
        & np.isfinite(
            x["distance_m"]
        )
        & (
            x["distance_m"] >= 0
        )
    )

    implausible_displacement = (
        supported
        & x["trajectory_status"].eq(
            TRAJECTORY_STATUS_IMPLAUSIBLE_SPEED
        )
    )

    supported_displacement_distance_m = float(
        x.loc[
            plausible_displacement,
            "distance_m",
        ].sum()
    )

    displacement_edge_count = int(
        plausible_displacement.sum()
    )

    implausible_transition_count = int(
        implausible_displacement.sum()
    )

    displacement_candidate_count = (
        displacement_edge_count
        + implausible_transition_count
    )

    # No displacement candidates: the fraction is undefined, not zero.
    implausible_transition_fraction = (
        implausible_transition_count
        / displacement_candidate_count
        if displacement_candidate_count > 0
        else np.nan
    )

    valid_spatial = (
        supported
        & x["duration_ms"].notna()
        & np.isfinite(
            x["duration_ms"]
        )
        & (
            x["duration_ms"] > 0
        )
        & x["start_latitude"].notna()
        & x["start_longitude"].notna()
        & x["end_latitude"].notna()
        & x["end_longitude"].notna()
        & np.isfinite(
            x[
                [
                    "start_latitude",
                    "start_longitude",
                    "end_latitude",
                    "end_longitude",
                ]
            ]
        ).all(axis=1)
    )

    spatially_supported_duration_ms = float(
        x.loc[
            valid_spatial,
            "duration_ms",
        ].sum()
    )

    if not valid_spatial.any():
        radius_of_gyration_m = np.nan

    else:
        z = (
            x.loc[
                valid_spatial,
                [
                    "duration_ms",
                    "start_latitude",
                    "start_longitude",
                    "end_latitude",
                    "end_longitude",
                ],
            ]
            .copy()
        )

        z["mid_latitude"] = (
            z["start_latitude"]
            + z["end_latitude"]
        ) / 2.0

        z["mid_longitude"] = (
            z["start_longitude"]
            + z["end_longitude"]
        ) / 2.0

        weights = (
            z["duration_ms"]
            .to_numpy(dtype=float)
        )

        center_latitude = float(
            np.average(
                z["mid_latitude"],
                weights=weights,
            )
        )

        center_longitude = float(
            np.average(
                z["mid_longitude"],
                weights=weights,
            )
        )

        distances_m = np.array(
            [
                haversine_km(
                    center_latitude,
                    center_longitude,
                    float(lat),
                    float(lon),
                )
                * 1000.0
                for lat, lon in zip(
                    z["mid_latitude"],
                    z["mid_longitude"],
                )
            ],
            dtype=float,
        )

        radius_of_gyration_m = float(
            math.sqrt(
                np.average(
                    distances_m ** 2,
                    weights=weights,
                )
            )
        )

    if minimum_supported_duration_ms < 0:
        raise ValueError(
            "minimum_supported_duration_ms must be >= 0"
        )

    distance_status = status_from_counts(
        spatially_supported_duration_ms,
        minimum_supported_duration_ms,
        supported_displacement_distance_m,
    )

    radius_status = status_from_counts(
        spatially_supported_duration_ms,
        minimum_supported_duration_ms,
        radius_of_gyration_m,
    )

    fraction_status = status_from_counts(
        spatially_supported_duration_ms,
        minimum_supported_duration_ms,
        implausible_transition_fraction,
    )

    return {
        "supported_displacement_distance_m":
            gated_value(
                supported_displacement_distance_m,
                distance_status,
            ),

        "supported_displacement_distance_m_status":
            distance_status.value,

        "supported_radius_of_gyration_m":
            gated_value(
                radius_of_gyration_m,
                radius_status,
            ),

        "supported_radius_of_gyration_m_status":
            radius_status.value,

        "displacement_edge_count":
            displacement_edge_count,

        "implausible_transition_count":
            implausible_transition_count,

        "implausible_transition_fraction":
            gated_value(
                implausible_transition_fraction,
                fraction_status,
            ),

        "implausible_transition_fraction_status":
            fraction_status.value,

        "spatially_supported_duration_ms":
            spatially_supported_duration_ms,
    }
