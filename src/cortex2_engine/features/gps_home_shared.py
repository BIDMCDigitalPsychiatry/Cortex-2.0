"""Home time from Cortex persistent-place episodes.

Home is defined from observed nighttime occupancy across the participant's
entire available timeline.

The persistent place accounting for the greatest cumulative observed
occupancy between 00:00 and 06:00 local time is the candidate home.

Home confidence is the fraction of all observed nighttime occupancy assigned
to that place. A home is accepted when this fraction is at least 0.50.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HomeConfig:
    timezone: str = "America/New_York"
    min_home_confidence: float = 0.50
    min_daily_supported_ms: float = 6 * 60 * 60 * 1000


def calculate_home_time(
    assignments: pd.DataFrame,
    *,
    supported_duration_ms: float,
    start_ms: int,
    end_ms: int,
    config: HomeConfig = HomeConfig(),
) -> dict:
    """Return home time and the inferred persistent home place.

    Home inference uses all available observed nighttime occupancy from
    00:00--06:00 local time across the supplied participant timeline.

    No fixed number of reference days or minimum number of nights is
    required. Home confidence is the proportion of observed nighttime
    occupancy assigned to the dominant nighttime place.
    """

    missing = {
        "episode_start_ms",
        "episode_end_ms",
        "place_candidate_id",
    } - set(assignments.columns)

    if missing:
        raise ValueError(
            f"Home assignments missing {sorted(missing)}"
        )

    a = assignments.copy()

    for col in (
        "episode_start_ms",
        "episode_end_ms",
        "place_candidate_id",
    ):
        a[col] = pd.to_numeric(
            a[col],
            errors="coerce",
        )

    a = a.dropna(
        subset=[
            "episode_start_ms",
            "episode_end_ms",
            "place_candidate_id",
        ]
    )

    a = a.loc[
        a["episode_end_ms"]
        > a["episode_start_ms"]
    ].copy()

    result = {
        "home_time_ms": np.nan,
        "home_place_candidate_id": np.nan,
        "home_confidence": np.nan,
        "home_valid_nights": 0,
        "home_nighttime_observed_ms": 0.0,
        "home_nighttime_at_home_ms": 0.0,
        "home_status": "insufficient_observability",
    }

    if a.empty:
        return result

    starts = a[
        "episode_start_ms"
    ].to_numpy(dtype=float)

    ends = a[
        "episode_end_ms"
    ].to_numpy(dtype=float)

    places = a[
        "place_candidate_id"
    ].to_numpy(dtype=int)

    first_date = (
        pd.to_datetime(
            starts.min(),
            unit="ms",
            utc=True,
        )
        .tz_convert(config.timezone)
        .date()
    )

    last_date = (
        pd.to_datetime(
            ends.max(),
            unit="ms",
            utc=True,
        )
        .tz_convert(config.timezone)
        .date()
    )

    dates = pd.date_range(
        first_date,
        last_date,
        freq="D",
    )

    total_by_place = {}
    n_observed_nights = 0
    total_nighttime_ms = 0.0

    for day in dates:
        local_date = day.date()

        lo = pd.Timestamp(
            f"{local_date} 00:00:00"
        ).tz_localize(
            config.timezone
        )

        hi = pd.Timestamp(
            f"{local_date} 06:00:00"
        ).tz_localize(
            config.timezone
        )

        lo_ms = (
            lo.tz_convert("UTC").timestamp()
            * 1000
        )

        hi_ms = (
            hi.tz_convert("UTC").timestamp()
            * 1000
        )

        overlap = np.maximum(
            0.0,
            np.minimum(
                ends,
                hi_ms,
            )
            - np.maximum(
                starts,
                lo_ms,
            ),
        )

        observed_ms = float(
            overlap.sum()
        )

        if observed_ms <= 0:
            continue

        n_observed_nights += 1
        total_nighttime_ms += observed_ms

        night = pd.DataFrame(
            {
                "place": places,
                "duration": overlap,
            }
        )

        night = night.loc[
            night["duration"] > 0
        ]

        if night.empty:
            continue

        sums = (
            night
            .groupby("place")[
                "duration"
            ]
            .sum()
        )

        for place_id, duration in sums.items():
            place_id = int(place_id)

            total_by_place[
                place_id
            ] = (
                total_by_place.get(
                    place_id,
                    0.0,
                )
                + float(duration)
            )

    result[
        "home_valid_nights"
    ] = n_observed_nights

    result[
        "home_nighttime_observed_ms"
    ] = total_nighttime_ms

    if (
        total_nighttime_ms <= 0
        or not total_by_place
    ):
        return result

    ranked = sorted(
        total_by_place.items(),
        key=lambda x: (
            -x[1],
            x[0],
        ),
    )

    home_id, home_night_ms = ranked[0]

    confidence = (
        home_night_ms
        / total_nighttime_ms
    )

    result[
        "home_confidence"
    ] = float(confidence)

    result[
        "home_nighttime_at_home_ms"
    ] = float(home_night_ms)

    if (
        confidence
        < config.min_home_confidence
    ):
        result[
            "home_status"
        ] = "low_home_confidence"

        return result

    result[
        "home_place_candidate_id"
    ] = int(home_id)

    # Home time in the requested output window.
    overlap = np.maximum(
        0.0,
        np.minimum(
            ends,
            float(end_ms),
        )
        - np.maximum(
            starts,
            float(start_ms),
        ),
    )

    if (
        supported_duration_ms
        < config.min_daily_supported_ms
    ):
        result[
            "home_status"
        ] = "insufficient_window_support"

        return result

    result[
        "home_time_ms"
    ] = float(
        overlap[
            places == home_id
        ].sum()
    )

    result[
        "home_status"
    ] = "observed_or_reconstructed"

    return result
