import pandas as pd
import pytest

from cortex2_engine.canonical.episodes import build_episodes_from_windows
from cortex2_engine.canonical.timestamps import datetime_to_epoch_ms
from cortex2_engine.canonical.validation import validate_canonical

MS = 1_700_000_000_123


@pytest.mark.parametrize(
    "unit",
    [
        "s",
        "ms",
        "us",
        "ns",
    ],
)
def test_conversion_is_resolution_independent(unit):

    values = pd.Series(
        pd.to_datetime(
            [MS - MS % 1000],
            unit="ms",
            utc=True,
        )
    ).dt.as_unit(unit)

    result = datetime_to_epoch_ms(
        values
    )

    assert int(result.iloc[0]) == MS - MS % 1000


def test_millisecond_precision_is_kept():

    values = pd.Series(
        pd.to_datetime(
            [MS],
            unit="ms",
            utc=True,
        )
    )

    assert int(datetime_to_epoch_ms(values).iloc[0]) == MS


def test_nat_becomes_na():

    values = pd.Series(
        [pd.NaT],
        dtype="datetime64[ns, UTC]",
    )

    assert datetime_to_epoch_ms(values).isna().all()


def test_validation_accepts_microsecond_resolution():

    utc = pd.Series(
        pd.to_datetime(
            [MS],
            unit="ms",
            utc=True,
        )
    ).dt.as_unit("us")

    df = pd.DataFrame({
        "study_id": ["S"],
        "timestamp_ms": pd.Series([MS], dtype="int64"),
        "timestamp_utc": utc,
        "latitude": [42.0],
        "longitude": [-71.0],
        "source_platform": ["test"],
        "source_stream": ["gps"],
        "source_participant_id": ["U1"],
    })

    report = validate_canonical(
        df,
        "gps",
    )

    assert report.valid, report.errors


def test_episode_bounds_are_epoch_ms():

    windows = pd.DataFrame({
        "participant_id": ["P1"],
        "uid": ["U1"],
        "start": ["2026-03-07"],
        "end": ["2026-03-08"],
    })

    episodes = build_episodes_from_windows(
        windows,
        study_id="S",
        participant_id_column="participant_id",
        source_participant_id_column="uid",
        start_column="start",
        end_column="end",
    )

    assert episodes.loc[0, "start_ms"] == 1_772_841_600_000
    assert episodes.loc[0, "end_ms"] == 1_772_841_600_000 + 2 * 86_400_000
