"""Spatial evidence semantics for Cortex 2.0 GPS edges.

This module interprets the relationship between observed displacement and
reported endpoint positional accuracy.

It does NOT classify behavior as stationary, walking, driving, or traveling.

For a temporally supported GPS edge with endpoint accuracy radii a1 and a2:

    combined_accuracy_m = a1 + a2

and

    displacement_accuracy_ratio =
        observed_distance_m / combined_accuracy_m

If observed distance <= combined accuracy, the two endpoint accuracy regions
overlap geometrically.

This is labeled `accuracy_overlap`.

If observed distance > combined accuracy, the observed displacement exceeds
the summed endpoint accuracy radii.

This is labeled `displacement_exceeds_accuracy`.

Reported positional accuracy is not assumed to be a formal statistical
confidence interval. These labels describe sensor evidence only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SPATIAL_STATUS_TEMPORAL_GAP = (
    "temporal_gap"
)

SPATIAL_STATUS_ACCURACY_UNAVAILABLE = (
    "accuracy_unavailable"
)

SPATIAL_STATUS_ACCURACY_OVERLAP = (
    "accuracy_overlap"
)

SPATIAL_STATUS_DISPLACEMENT_EXCEEDS_ACCURACY = (
    "displacement_exceeds_accuracy"
)


def add_gps_spatial_semantics(
    edges: pd.DataFrame,
) -> pd.DataFrame:
    """Add accuracy-aware spatial evidence semantics to GPS edges.

    Parameters
    ----------
    edges
        Output-compatible dataframe from `build_gps_intervals`.

        Required fields:
        - supported
        - distance_km
        - start_accuracy
        - end_accuracy

    Returns
    -------
    pandas.DataFrame
        Copy of the input with additional fields:

        - distance_m
        - combined_accuracy_m
        - displacement_accuracy_ratio
        - displacement_excess_m
        - accuracy_regions_overlap
        - spatial_status
    """

    required = {
        "supported",
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

    result = edges.copy()

    result["distance_km"] = pd.to_numeric(
        result["distance_km"],
        errors="coerce",
    )

    result["start_accuracy"] = pd.to_numeric(
        result["start_accuracy"],
        errors="coerce",
    )

    result["end_accuracy"] = pd.to_numeric(
        result["end_accuracy"],
        errors="coerce",
    )


    result["distance_m"] = (
        result["distance_km"]
        * 1000
    )


    valid_accuracy = (
        result[
            "start_accuracy"
        ].notna()
        & result[
            "end_accuracy"
        ].notna()
        & np.isfinite(
            result[
                "start_accuracy"
            ]
        )
        & np.isfinite(
            result[
                "end_accuracy"
            ]
        )
        & (
            result[
                "start_accuracy"
            ]
            > 0
        )
        & (
            result[
                "end_accuracy"
            ]
            > 0
        )
    )


    result[
        "combined_accuracy_m"
    ] = np.nan

    result.loc[
        valid_accuracy,
        "combined_accuracy_m",
    ] = (
        result.loc[
            valid_accuracy,
            "start_accuracy",
        ]
        + result.loc[
            valid_accuracy,
            "end_accuracy",
        ]
    )


    result[
        "displacement_accuracy_ratio"
    ] = np.nan

    result.loc[
        valid_accuracy,
        "displacement_accuracy_ratio",
    ] = (
        result.loc[
            valid_accuracy,
            "distance_m",
        ]
        / result.loc[
            valid_accuracy,
            "combined_accuracy_m",
        ]
    )


    result[
        "displacement_excess_m"
    ] = np.nan

    result.loc[
        valid_accuracy,
        "displacement_excess_m",
    ] = np.maximum(
        0.0,
        (
            result.loc[
                valid_accuracy,
                "distance_m",
            ]
            - result.loc[
                valid_accuracy,
                "combined_accuracy_m",
            ]
        ),
    )


    overlap_values = np.full(
        len(result),
        None,
        dtype=object,
    )


    supported = (
        result[
            "supported"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
    )


    usable = (
        supported
        & valid_accuracy
        & result[
            "distance_m"
        ].notna()
        & np.isfinite(
            result[
                "distance_m"
            ]
        )
    )


    overlap = (
        usable
        & (
            result[
                "distance_m"
            ]
            <= result[
                "combined_accuracy_m"
            ]
        )
    )


    separated = (
        usable
        & (
            result[
                "distance_m"
            ]
            > result[
                "combined_accuracy_m"
            ]
        )
    )


    overlap_values[
        overlap.to_numpy()
    ] = True

    overlap_values[
        separated.to_numpy()
    ] = False

    result[
        "accuracy_regions_overlap"
    ] = pd.array(
        overlap_values,
        dtype="boolean",
    )


    result[
        "spatial_status"
    ] = (
        SPATIAL_STATUS_ACCURACY_UNAVAILABLE
    )


    result.loc[
        ~supported,
        "spatial_status",
    ] = (
        SPATIAL_STATUS_TEMPORAL_GAP
    )


    result.loc[
        overlap,
        "spatial_status",
    ] = (
        SPATIAL_STATUS_ACCURACY_OVERLAP
    )


    result.loc[
        separated,
        "spatial_status",
    ] = (
        SPATIAL_STATUS_DISPLACEMENT_EXCEEDS_ACCURACY
    )


    return result
