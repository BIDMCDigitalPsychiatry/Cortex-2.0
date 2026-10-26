import pandas as pd

from cortex2_engine.canonical.validation import (
    validate_canonical,
)


def test_valid_gps():

    df = pd.DataFrame({
        "study_id":
            ["study"],

        "participant_id":
            ["P001"],

        "source_participant_id":
            ["U001"],

        "timestamp_ms":
            [1_700_000_000_000],

        "timestamp_utc":
            pd.to_datetime(
                [1_700_000_000_000],
                unit="ms",
                utc=True,
            ),

        "latitude":
            [42.3],

        "longitude":
            [-71.1],

        "source_platform":
            ["test"],

        "source_stream":
            ["gps"],
    })

    result = validate_canonical(
        df,
        "gps",
    )

    assert result.valid


def test_invalid_latitude():

    df = pd.DataFrame({
        "study_id":
            ["study"],

        "participant_id":
            ["P001"],

        "source_participant_id":
            ["U001"],

        "timestamp_ms":
            [1_700_000_000_000],

        "timestamp_utc":
            pd.to_datetime(
                [1_700_000_000_000],
                unit="ms",
                utc=True,
            ),

        "latitude":
            [142.3],

        "longitude":
            [-71.1],

        "source_platform":
            ["test"],

        "source_stream":
            ["gps"],
    })

    result = validate_canonical(
        df,
        "gps",
    )

    assert not result.valid
    assert any(
        "latitude" in error
        for error in result.errors
    )
