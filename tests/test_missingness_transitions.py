import pandas as pd

from cortex2_engine.missingness.transitions import (
    add_hourly_missingness_transitions,
)


def base_df():
    return pd.DataFrame({
        "episode_id": ["e1"] * 4,
        "hour_index": [0, 1, 2, 3],

        "gps_observed": [
            True,
            True,
            False,
            False,
        ],

        "acc_observed": [
            True,
            True,
            True,
            True,
        ],

        "screen_event_observed": [
            False,
            True,
            True,
            False,
        ],

        "gps_dq": [
            1.0,
            1.0,
            0.0,
            0.0,
        ],

        "acc_dq": [
            1.0,
            1.0,
            1.0,
            1.0,
        ],

        "gps_source_unavailable": [False] * 4,
        "acc_source_unavailable": [False] * 4,
        "gps_partial_source": [False] * 4,
        "acc_partial_source": [False] * 4,
        "gps_boundary_unknown": [False] * 4,
        "acc_boundary_unknown": [False] * 4,
    })


def test_gps_gap_onset_detected_once():
    x = add_hourly_missingness_transitions(
        base_df()
    )

    assert x["gps_gap_onset"].sum() == 1
    assert bool(
        x.loc[
            x["hour_index"].eq(2),
            "gps_gap_onset",
        ].iloc[0]
    )


def test_gps_specific_onset():
    x = add_hourly_missingness_transitions(
        base_df()
    )

    assert (
        x["gps_specific_gap_onset"]
        .sum()
        == 1
    )


def test_following_gap_is_persistence():
    x = add_hourly_missingness_transitions(
        base_df()
    )

    assert (
        x["gps_gap_persistence"]
        .sum()
        == 1
    )


def test_discontinuous_hours_do_not_create_transition():
    df = base_df().iloc[
        [0, 2]
    ].copy()

    x = add_hourly_missingness_transitions(
        df
    )

    assert x["gps_gap_onset"].sum() == 0
