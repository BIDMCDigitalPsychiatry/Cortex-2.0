"""Time-weighted RAPHE location features.

Candidate places are converted into daily behavioral summaries using
OBSERVED TIME, not GPS point counts.

This module does not assume that the dominant place is home.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def compute_daily_location_features(
    assignments: pd.DataFrame,
    *,
    supported_duration_ms: float | None = None,
) -> dict:
    """Compute daily time-weighted location features.

    Parameters
    ----------
    assignments
        Candidate-place assignments for one participant-day.
        Requires:
        - place_candidate_id
        - duration_ms

    supported_duration_ms
        Total temporally supported GPS duration for the day, if known.

    Returns
    -------
    dict
        Time-weighted location entropy, dominant-place statistics,
        and observability context.
    """

    required = {
        "place_candidate_id",
        "duration_ms",
    }

    missing = (
        required
        - set(assignments.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    x = assignments.copy()

    x["duration_ms"] = pd.to_numeric(
        x["duration_ms"],
        errors="coerce",
    )

    x["place_candidate_id"] = pd.to_numeric(
        x["place_candidate_id"],
        errors="coerce",
    )

    x = x[
        x["duration_ms"].notna()
        & np.isfinite(
            x["duration_ms"]
        )
        & (
            x["duration_ms"]
            > 0
        )
        & x["place_candidate_id"].notna()
    ].copy()

    if supported_duration_ms is not None:
        supported_duration_ms = float(
            supported_duration_ms
        )

        if (
            not math.isfinite(
                supported_duration_ms
            )
            or supported_duration_ms < 0
        ):
            raise ValueError(
                "supported_duration_ms must be finite and >= 0"
            )

    if x.empty:

        return {
            "location_entropy_nats":
                np.nan,

            "location_entropy_status":
                "insufficient_observability",

            "effective_number_places":
                np.nan,

            "dominant_place_id":
                np.nan,

            "dominant_place_time_ms":
                np.nan,

            "dominant_place_fraction":
                np.nan,

            "dominant_place_status":
                "insufficient_observability",

            "n_observed_places":
                0,

            "assigned_place_time_ms":
                0.0,

            "supported_duration_ms":
                supported_duration_ms,

            "place_time_fraction_of_supported":
                np.nan,

            "unassigned_supported_duration_ms":
                (
                    supported_duration_ms
                    if supported_duration_ms
                    is not None
                    else np.nan
                ),
        }

    occupancy = (
        x.groupby(
            "place_candidate_id",
            sort=True,
        )[
            "duration_ms"
        ]
        .sum()
        .reset_index()
    )

    total_time = float(
        occupancy[
            "duration_ms"
        ].sum()
    )

    occupancy[
        "fraction"
    ] = (
        occupancy[
            "duration_ms"
        ]
        / total_time
    )

    probabilities = (
        occupancy[
            "fraction"
        ]
        .to_numpy(
            dtype=float
        )
    )

    entropy = -float(
        np.sum(
            probabilities
            * np.log(
                probabilities
            )
        )
    )

    # Avoid negative zero from floating-point arithmetic.
    if abs(entropy) < 1e-15:
        entropy = 0.0

    effective_number_places = float(
        np.exp(
            entropy
        )
    )

    ranked = (
        occupancy.sort_values(
            [
                "duration_ms",
                "place_candidate_id",
            ],
            ascending=[
                False,
                True,
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    dominant = ranked.iloc[0]

    dominant_place_id = int(
        dominant[
            "place_candidate_id"
        ]
    )

    dominant_place_time_ms = float(
        dominant[
            "duration_ms"
        ]
    )

    dominant_place_fraction = (
        dominant_place_time_ms
        / total_time
    )

    if supported_duration_ms is not None:

        if supported_duration_ms > 0:

            place_time_fraction = (
                total_time
                / supported_duration_ms
            )

            unassigned_supported = max(
                0.0,
                supported_duration_ms
                - total_time,
            )

        else:

            place_time_fraction = np.nan
            unassigned_supported = 0.0

    else:

        place_time_fraction = np.nan
        unassigned_supported = np.nan

    return {
        "location_entropy_nats":
            entropy,

        "location_entropy_status":
            (
                "observed_zero"
                if entropy == 0.0
                else "observed_value"
            ),

        "effective_number_places":
            effective_number_places,

        "dominant_place_id":
            dominant_place_id,

        "dominant_place_time_ms":
            dominant_place_time_ms,

        "dominant_place_fraction":
            dominant_place_fraction,

        "dominant_place_status":
            "observed_value",

        "n_observed_places":
            int(
                len(
                    occupancy
                )
            ),

        "assigned_place_time_ms":
            total_time,

        "supported_duration_ms":
            supported_duration_ms,

        "place_time_fraction_of_supported":
            place_time_fraction,

        "unassigned_supported_duration_ms":
            unassigned_supported,
    }
