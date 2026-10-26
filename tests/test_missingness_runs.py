import pandas as pd

from cortex2_engine.missingness.runs import (
    assign_contiguous_runs,
)


def test_same_state_consecutive_hours_same_run():

    df = pd.DataFrame({
        "episode_id": ["e1"] * 3,
        "hour_index": [0, 1, 2],
        "gap_pattern": [
            "gps_specific_gap",
            "gps_specific_gap",
            "gps_specific_gap",
        ],
    })

    out = assign_contiguous_runs(df)

    assert out["run_id"].nunique() == 1


def test_state_change_breaks_run():

    df = pd.DataFrame({
        "episode_id": ["e1"] * 3,
        "hour_index": [0, 1, 2],
        "gap_pattern": [
            "gps_specific_gap",
            "both_observed",
            "gps_specific_gap",
        ],
    })

    out = assign_contiguous_runs(df)

    assert out["run_id"].nunique() == 3


def test_hour_discontinuity_breaks_run():

    df = pd.DataFrame({
        "episode_id": ["e1"] * 2,
        "hour_index": [0, 2],
        "gap_pattern": [
            "gps_specific_gap",
            "gps_specific_gap",
        ],
    })

    out = assign_contiguous_runs(df)

    assert out["run_id"].nunique() == 2


def test_episode_change_breaks_run():

    df = pd.DataFrame({
        "episode_id": ["e1", "e2"],
        "hour_index": [0, 0],
        "gap_pattern": [
            "gps_specific_gap",
            "gps_specific_gap",
        ],
    })

    out = assign_contiguous_runs(df)

    assert out["run_id"].nunique() == 2
