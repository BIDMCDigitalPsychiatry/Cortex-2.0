import math

import numpy as np
import pandas as pd
import pytest

from raphe.features.gps_location_features import (
    compute_daily_location_features,
)


HOUR = 3_600_000


def test_two_places_are_time_weighted():

    assignments = pd.DataFrame({
        "place_candidate_id": [
            0,
            1,
        ],
        "duration_ms": [
            8 * HOUR,
            2 * HOUR,
        ],
    })

    result = compute_daily_location_features(
        assignments,
        supported_duration_ms=10 * HOUR,
    )

    expected_entropy = -(
        0.8 * math.log(0.8)
        + 0.2 * math.log(0.2)
    )

    assert result[
        "dominant_place_fraction"
    ] == pytest.approx(
        0.8
    )

    assert result[
        "location_entropy_nats"
    ] == pytest.approx(
        expected_entropy
    )

    assert result[
        "place_time_fraction_of_supported"
    ] == pytest.approx(
        1.0
    )


def test_many_rows_do_not_outvote_more_time():

    # Place 0:
    # 8 one-hour segments = 8 hours.
    #
    # Place 1:
    # 100 tiny rows = only 2 hours total.
    #
    # Number of rows must not determine the dominant place.

    assignments = pd.DataFrame({
        "place_candidate_id":
            [0] * 8
            + [1] * 100,

        "duration_ms":
            [HOUR] * 8
            + [
                (2 * HOUR) / 100
            ] * 100,
    })

    result = compute_daily_location_features(
        assignments
    )

    assert result[
        "dominant_place_id"
    ] == 0

    assert result[
        "dominant_place_fraction"
    ] == pytest.approx(
        0.8
    )


def test_single_place_has_zero_entropy():

    assignments = pd.DataFrame({
        "place_candidate_id": [
            3,
            3,
            3,
        ],
        "duration_ms": [
            HOUR,
            HOUR,
            HOUR,
        ],
    })

    result = compute_daily_location_features(
        assignments
    )

    assert result[
        "location_entropy_nats"
    ] == pytest.approx(
        0.0
    )

    assert result[
        "location_entropy_status"
    ] == "observed_zero"

    assert result[
        "effective_number_places"
    ] == pytest.approx(
        1.0
    )

    assert result[
        "dominant_place_fraction"
    ] == pytest.approx(
        1.0
    )


def test_supported_time_can_exceed_place_time():

    assignments = pd.DataFrame({
        "place_candidate_id": [
            0,
        ],
        "duration_ms": [
            6 * HOUR,
        ],
    })

    result = compute_daily_location_features(
        assignments,
        supported_duration_ms=10 * HOUR,
    )

    assert result[
        "place_time_fraction_of_supported"
    ] == pytest.approx(
        0.6
    )

    assert result[
        "unassigned_supported_duration_ms"
    ] == pytest.approx(
        4 * HOUR
    )


def test_empty_input_is_insufficient():

    assignments = pd.DataFrame(
        columns=[
            "place_candidate_id",
            "duration_ms",
        ]
    )

    result = compute_daily_location_features(
        assignments,
        supported_duration_ms=HOUR,
    )

    assert np.isnan(
        result[
            "location_entropy_nats"
        ]
    )

    assert result[
        "location_entropy_status"
    ] == "insufficient_observability"

    assert result[
        "n_observed_places"
    ] == 0


def test_zero_duration_rows_are_ignored():

    assignments = pd.DataFrame({
        "place_candidate_id": [
            0,
            1,
        ],
        "duration_ms": [
            HOUR,
            0,
        ],
    })

    result = compute_daily_location_features(
        assignments
    )

    assert result[
        "n_observed_places"
    ] == 1

    assert result[
        "dominant_place_fraction"
    ] == pytest.approx(
        1.0
    )


def test_invalid_supported_duration_raises():

    assignments = pd.DataFrame({
        "place_candidate_id": [
            0
        ],
        "duration_ms": [
            HOUR
        ],
    })

    with pytest.raises(
        ValueError
    ):
        compute_daily_location_features(
            assignments,
            supported_duration_ms=-1,
        )


def test_missing_column_raises():

    assignments = pd.DataFrame({
        "duration_ms": [
            HOUR
        ]
    })

    with pytest.raises(
        ValueError
    ):
        compute_daily_location_features(
            assignments
        )
