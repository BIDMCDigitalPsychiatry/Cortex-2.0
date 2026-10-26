import pytest

from cortex2_engine.features.gps_observability import (
    compute_gps_temporal_observability,
)


MINUTE = 60_000


def test_no_raw_data():
    result = compute_gps_temporal_observability(
        [],
        window_start_ms=0,
        window_end_ms=10 * MINUTE,
        max_gap_ms=2 * MINUTE,
    )

    assert result["status"] == "no_raw_data"
    assert result["n_observations"] == 0
    assert result["observed_duration_ms"] == 0
    assert result["unobserved_duration_ms"] == 10 * MINUTE
    assert result["observability"] == 0.0
    assert result["longest_unobserved_gap_ms"] == 10 * MINUTE


def test_single_point_does_not_imply_observed_time():
    result = compute_gps_temporal_observability(
        [5 * MINUTE],
        window_start_ms=0,
        window_end_ms=10 * MINUTE,
        max_gap_ms=2 * MINUTE,
    )

    assert result["status"] == "insufficient_observability"
    assert result["n_observations"] == 1
    assert result["n_supported_edges"] == 0
    assert result["observed_duration_ms"] == 0
    assert result["observability"] == 0.0
    assert result["longest_unobserved_gap_ms"] == 10 * MINUTE


def test_regular_observations_create_one_supported_segment():
    result = compute_gps_temporal_observability(
        [
            0,
            1 * MINUTE,
            2 * MINUTE,
            3 * MINUTE,
            4 * MINUTE,
        ],
        window_start_ms=0,
        window_end_ms=5 * MINUTE,
        max_gap_ms=1 * MINUTE,
    )

    assert result["status"] == "observed_value"
    assert result["n_observations"] == 5
    assert result["n_supported_edges"] == 4
    assert result["n_observable_segments"] == 1
    assert result["observed_duration_ms"] == 4 * MINUTE
    assert result["unobserved_duration_ms"] == 1 * MINUTE
    assert result["observability"] == pytest.approx(0.8)
    assert result["leading_gap_ms"] == 0
    assert result["trailing_gap_ms"] == 1 * MINUTE
    assert result["longest_unobserved_gap_ms"] == 1 * MINUTE


def test_long_gap_splits_observable_segments():
    result = compute_gps_temporal_observability(
        [
            0,
            1 * MINUTE,
            2 * MINUTE,
            10 * MINUTE,
            11 * MINUTE,
        ],
        window_start_ms=0,
        window_end_ms=12 * MINUTE,
        max_gap_ms=2 * MINUTE,
    )

    assert result["n_supported_edges"] == 3
    assert result["n_observable_segments"] == 2

    assert result["observed_duration_ms"] == 3 * MINUTE
    assert result["unobserved_duration_ms"] == 9 * MINUTE
    assert result["observability"] == pytest.approx(0.25)

    assert result["longest_unobserved_gap_ms"] == 8 * MINUTE


def test_gap_equal_to_threshold_is_supported():
    result = compute_gps_temporal_observability(
        [
            0,
            2 * MINUTE,
        ],
        window_start_ms=0,
        window_end_ms=3 * MINUTE,
        max_gap_ms=2 * MINUTE,
    )

    assert result["n_supported_edges"] == 1
    assert result["observed_duration_ms"] == 2 * MINUTE


def test_gap_above_threshold_is_not_supported():
    result = compute_gps_temporal_observability(
        [
            0,
            2 * MINUTE + 1,
        ],
        window_start_ms=0,
        window_end_ms=3 * MINUTE,
        max_gap_ms=2 * MINUTE,
    )

    assert result["n_supported_edges"] == 0
    assert result["observed_duration_ms"] == 0
    assert result["status"] == "insufficient_observability"


def test_duplicates_and_outside_window_are_audited():
    result = compute_gps_temporal_observability(
        [
            -1,
            0,
            0,
            1 * MINUTE,
            2 * MINUTE,
            3 * MINUTE,
        ],
        window_start_ms=0,
        window_end_ms=3 * MINUTE,
        max_gap_ms=1 * MINUTE,
    )

    # -1 and the end-boundary timestamp are outside the [start, end) window.
    assert result["n_input_timestamps"] == 6
    assert result["n_outside_window"] == 2
    assert result["n_duplicate_timestamps"] == 1
    assert result["n_observations"] == 3

    assert result["observed_duration_ms"] == 2 * MINUTE
    assert result["trailing_gap_ms"] == 1 * MINUTE


def test_invalid_timestamps_are_not_silently_treated_as_missing_behavior():
    result = compute_gps_temporal_observability(
        [
            None,
            "bad",
            float("nan"),
            0,
            1 * MINUTE,
        ],
        window_start_ms=0,
        window_end_ms=2 * MINUTE,
        max_gap_ms=1 * MINUTE,
    )

    assert result["n_invalid_timestamps"] == 3
    assert result["n_observations"] == 2
    assert result["observed_duration_ms"] == 1 * MINUTE


def test_invalid_window_raises():
    with pytest.raises(ValueError):
        compute_gps_temporal_observability(
            [0],
            window_start_ms=1000,
            window_end_ms=1000,
            max_gap_ms=100,
        )


def test_invalid_gap_tolerance_raises():
    with pytest.raises(ValueError):
        compute_gps_temporal_observability(
            [0],
            window_start_ms=0,
            window_end_ms=1000,
            max_gap_ms=0,
        )
