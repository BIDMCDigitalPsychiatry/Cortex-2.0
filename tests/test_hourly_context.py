import pandas as pd
import pytest

from raphe.missingness.hourly_context import (
    episode_hourly_context,
)


class FakeReader:

    def __init__(self):

        self.ep = pd.Series({
            "study_id": "s",
            "participant_id": "p",
            "episode_id": "e",
            "source_participant_id": "u",
            "start_ms": 0,
            "end_ms": 3_600_000,
        })

    def episode(self, episode_id):
        return self.ep

    def iter_stream(
        self,
        stream,
        *,
        episode_id,
        chunksize,
    ):

        if stream == "screen":

            yield pd.DataFrame({
                "timestamp_ms": [
                    1000,
                    2000,
                ],
                "battery_level_fraction": [
                    0.50,
                    0.48,
                ],
            })

        elif stream == "device_usage":

            yield pd.DataFrame({
                "timestamp_ms": [
                    900_000,
                    1_800_000,
                ],
                "interval_duration_ms": [
                    900_000,
                    900_000,
                ],
                "total_unlock_duration_ms": [
                    100_000,
                    200_000,
                ],
                "total_unlocks": [
                    1,
                    2,
                ],
                "total_screen_wakes": [
                    2,
                    3,
                ],
            })

        elif stream == "nearby_device":

            yield pd.DataFrame({
                "timestamp_ms": [
                    1000,
                    2000,
                    3000,
                ],
                "device_type": [
                    "wifi",
                    "wifi",
                    "bluetooth",
                ],
                "nearby_identifier_hash": [
                    "a",
                    "a",
                    "b",
                ],
            })


def test_hourly_context():

    out = episode_hourly_context(
        FakeReader(),
        episode_id="e",
        stream_evidence={
            "screen": True,
            "device_usage": True,
            "nearby_device": True,
        },
    )

    row = out.iloc[0]

    assert row[
        "screen_event_count"
    ] == 2

    assert row[
        "battery_last"
    ] == 0.48

    assert row[
        "battery_within_hour_delta"
    ] == pytest.approx(-0.02)

    assert row[
        "device_usage_record_count"
    ] == 2

    assert row[
        "unlock_count"
    ] == 3

    assert row[
        "screen_wake_count"
    ] == 5

    assert row[
        "nearby_record_count"
    ] == 3

    assert row[
        "unique_wifi_id_count"
    ] == 1

    assert row[
        "unique_bluetooth_id_count"
    ] == 1


class DuplicateDeviceUsageReader(
    FakeReader
):

    def iter_stream(
        self,
        stream,
        *,
        episode_id,
        chunksize,
    ):

        if stream == "device_usage":

            yield pd.DataFrame({
                "timestamp_ms": [
                    900_000,
                    900_000,
                    1_800_000,
                ],
                "interval_duration_ms": [
                    900_000,
                    900_000,
                    900_000,
                ],
                "total_unlock_duration_ms": [
                    100_000,
                    100_000,
                    200_000,
                ],
                "total_unlocks": [
                    1,
                    1,
                    2,
                ],
                "total_screen_wakes": [
                    2,
                    2,
                    3,
                ],
            })

        else:
            yield from super().iter_stream(
                stream,
                episode_id=episode_id,
                chunksize=chunksize,
            )


def test_device_usage_duplicates_are_not_summed():

    out = episode_hourly_context(
        DuplicateDeviceUsageReader(),
        episode_id="e",
        stream_evidence={
            "screen": True,
            "device_usage": True,
            "nearby_device": True,
        },
    )

    row = out.iloc[0]

    assert (
        row[
            "device_usage_raw_record_count"
        ]
        == 3
    )

    assert (
        row[
            "device_usage_record_count"
        ]
        == 2
    )

    assert (
        row[
            "device_usage_duplicate_record_count"
        ]
        == 1
    )

    assert (
        row[
            "device_usage_max_duplicate_multiplicity"
        ]
        == 2
    )

    assert (
        row[
            "device_usage_conflict_timestamp_count"
        ]
        == 0
    )

    assert (
        row["unlock_count"]
        == 3
    )

    assert (
        row["screen_wake_count"]
        == 5
    )


class ConflictingDeviceUsageReader(
    FakeReader
):

    def iter_stream(
        self,
        stream,
        *,
        episode_id,
        chunksize,
    ):

        if stream == "device_usage":

            yield pd.DataFrame({
                "timestamp_ms": [
                    900_000,
                    900_000,
                ],
                "interval_duration_ms": [
                    900_000,
                    900_000,
                ],
                "total_unlock_duration_ms": [
                    0,
                    252_000,
                ],
                "total_unlocks": [
                    0,
                    4,
                ],
                "total_screen_wakes": [
                    0,
                    4,
                ],
            })

        else:
            yield from super().iter_stream(
                stream,
                episode_id=episode_id,
                chunksize=chunksize,
            )


def test_conflicting_device_usage_is_flagged():

    out = episode_hourly_context(
        ConflictingDeviceUsageReader(),
        episode_id="e",
        stream_evidence={
            "screen": True,
            "device_usage": True,
            "nearby_device": True,
        },
    )

    row = out.iloc[0]

    assert (
        row[
            "device_usage_raw_record_count"
        ]
        == 2
    )

    assert (
        row[
            "device_usage_record_count"
        ]
        == 1
    )

    assert (
        row[
            "device_usage_conflict_timestamp_count"
        ]
        == 1
    )

    assert pd.isna(
        row["unlock_count"]
    )

    assert pd.isna(
        row["screen_wake_count"]
    )
