"""Trajectory plausibility for Cortex 2.0 GPS displacement evidence.

This layer is applied only after:
1. temporal support has been established; and
2. accuracy-aware spatial semantics have been computed.

It distinguishes displacement that exceeds positional uncertainty from
displacement that is also temporally plausible.

This is sensor-validity evidence, not behavioral classification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cortex2_engine.features.gps_spatial_semantics import (
    SPATIAL_STATUS_DISPLACEMENT_EXCEEDS_ACCURACY,
)


TRAJECTORY_STATUS_NOT_DISPLACEMENT = (
    "not_displacement"
)

TRAJECTORY_STATUS_PLAUSIBLE_DISPLACEMENT = (
    "plausible_displacement"
)

TRAJECTORY_STATUS_IMPLAUSIBLE_SPEED = (
    "implausible_speed"
)

TRAJECTORY_STATUS_INVALID_DURATION = (
    "invalid_duration"
)


def add_gps_trajectory_plausibility(
    edges: pd.DataFrame,
    *,
    max_plausible_speed_kmh: float = 1000.0,
) -> pd.DataFrame:
    """Add trajectory-plausibility semantics to GPS edges.

    Parameters
    ----------
    edges
        Output from ``add_gps_spatial_semantics``.

    max_plausible_speed_kmh
        Provisional upper bound for physically plausible movement.

        The default of 1000 km/h is intended to preserve ordinary
        commercial-air-travel-scale movement while rejecting gross
        near-instantaneous coordinate conflicts.

        This is an explicit model parameter, not a universal threshold.
    """

    if (
        not np.isfinite(
            max_plausible_speed_kmh
        )
        or max_plausible_speed_kmh <= 0
    ):
        raise ValueError(
            "max_plausible_speed_kmh must be finite and > 0"
        )

    required = {
        "supported",
        "duration_ms",
        "distance_m",
        "spatial_status",
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

    result = edges.copy()

    result["duration_ms"] = pd.to_numeric(
        result["duration_ms"],
        errors="coerce",
    )

    result["distance_m"] = pd.to_numeric(
        result["distance_m"],
        errors="coerce",
    )

    valid_duration = (
        result["duration_ms"].notna()
        & np.isfinite(
            result["duration_ms"]
        )
        & (
            result["duration_ms"] > 0
        )
    )

    valid_distance = (
        result["distance_m"].notna()
        & np.isfinite(
            result["distance_m"]
        )
        & (
            result["distance_m"] >= 0
        )
    )

    result["derived_speed_kmh"] = np.nan

    valid_speed = (
        valid_duration
        & valid_distance
    )

    result.loc[
        valid_speed,
        "derived_speed_kmh",
    ] = (
        result.loc[
            valid_speed,
            "distance_m",
        ]
        / (
            result.loc[
                valid_speed,
                "duration_ms",
            ]
            / 1000.0
        )
        * 3.6
    )

    displacement = (
        result["supported"]
        .fillna(False)
        .astype(bool)
        & result[
            "spatial_status"
        ].eq(
            SPATIAL_STATUS_DISPLACEMENT_EXCEEDS_ACCURACY
        )
    )

    result["trajectory_status"] = (
        TRAJECTORY_STATUS_NOT_DISPLACEMENT
    )

    result.loc[
        displacement
        & ~valid_duration,
        "trajectory_status",
    ] = (
        TRAJECTORY_STATUS_INVALID_DURATION
    )

    plausible = (
        displacement
        & valid_speed
        & (
            result[
                "derived_speed_kmh"
            ]
            <= max_plausible_speed_kmh
        )
    )

    implausible = (
        displacement
        & valid_speed
        & (
            result[
                "derived_speed_kmh"
            ]
            > max_plausible_speed_kmh
        )
    )

    result.loc[
        plausible,
        "trajectory_status",
    ] = (
        TRAJECTORY_STATUS_PLAUSIBLE_DISPLACEMENT
    )

    result.loc[
        implausible,
        "trajectory_status",
    ] = (
        TRAJECTORY_STATUS_IMPLAUSIBLE_SPEED
    )

    result[
        "trajectory_plausible"
    ] = pd.array(
        np.where(
            plausible,
            True,
            np.where(
                displacement,
                False,
                None,
            ),
        ),
        dtype="boolean",
    )

    return result
