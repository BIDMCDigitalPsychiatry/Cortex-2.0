import pandas as pd
import pytest

from cortex2_engine.adapters.mindlamp import MindLAMPAdapter
from cortex2_engine.canonical.episodes import (
    build_episodes_from_windows,
)
from cortex2_engine.canonical.reader import EpisodeReader


def make_source(tmp_path):

    gps = tmp_path / "gps"
    gps.mkdir()

    # Two distinct periods belonging to the same source UID.
    pd.DataFrame({
        "timestamp": [
            1704067200000,  # 2024-01-01
            1704153600000,  # 2024-01-02
            1735689600000,  # 2025-01-01
            1735776000000,  # 2025-01-02
        ],
        "latitude": [
            42.1, 42.2, 43.1, 43.2,
        ],
        "longitude": [
            -71.1, -71.2, -72.1, -72.2,
        ],
        "accuracy": [
            5, 5, 5, 5,
        ],
        "altitude": [
            10, 10, 10, 10,
        ],
    }).to_csv(
        gps / "gps_U123_1.csv"
    )


def test_episode_reader_resolves_ambiguous_uid(
    tmp_path,
):

    make_source(tmp_path)

    mapping = pd.DataFrame({
        "mindlamp_uid": [
            "U123",
            "U123",
        ],
        "participant_id": [
            "P001",
            "P002",
        ],
    })

    mapping_path = tmp_path / "mapping.csv"

    mapping.to_csv(
        mapping_path,
        index=False,
    )

    windows = pd.DataFrame({
        "participant_id": [
            "P001",
            "P002",
        ],
        "mindlamp_uid": [
            "U123",
            "U123",
        ],
        "start": [
            "2024-01-01",
            "2025-01-01",
        ],
        "end": [
            "2024-01-02",
            "2025-01-02",
        ],
    })

    episodes = build_episodes_from_windows(
        windows,
        study_id="test",
        participant_id_column="participant_id",
        source_participant_id_column="mindlamp_uid",
        start_column="start",
        end_column="end",
        inclusive_end_date=True,
        source_platform="mindlamp",
    )

    adapter = MindLAMPAdapter(
        study_id="test",
        passive_root=tmp_path,
        participant_map=mapping_path,
    )

    reader = EpisodeReader(
        adapter,
        episodes,
    )

    first_id = episodes.loc[
        episodes["participant_id"] == "P001",
        "episode_id",
    ].iloc[0]

    second_id = episodes.loc[
        episodes["participant_id"] == "P002",
        "episode_id",
    ].iloc[0]

    first = pd.concat(
        list(
            reader.iter_stream(
                "gps",
                episode_id=first_id,
            )
        ),
        ignore_index=True,
    )

    second = pd.concat(
        list(
            reader.iter_stream(
                "gps",
                episode_id=second_id,
            )
        ),
        ignore_index=True,
    )

    assert first["participant_id"].unique().tolist() == [
        "P001"
    ]

    assert second["participant_id"].unique().tolist() == [
        "P002"
    ]

    assert first["source_participant_id"].unique().tolist() == [
        "U123"
    ]

    assert first["latitude"].tolist() == [
        42.1,
        42.2,
    ]

    assert second["latitude"].tolist() == [
        43.1,
        43.2,
    ]


def test_inclusive_dates_become_half_open():

    windows = pd.DataFrame({
        "participant_id": ["P001"],
        "uid": ["U1"],
        "start": ["2024-02-06"],
        "end": ["2024-03-27"],
    })

    episodes = build_episodes_from_windows(
        windows,
        study_id="test",
        participant_id_column="participant_id",
        source_participant_id_column="uid",
        start_column="start",
        end_column="end",
        inclusive_end_date=True,
    )

    assert (
        episodes.iloc[0]["end_utc"]
        == pd.Timestamp(
            "2024-03-28",
            tz="UTC",
        )
    )


def test_overlapping_different_participants_rejected():

    windows = pd.DataFrame({
        "participant_id": [
            "P001",
            "P002",
        ],
        "uid": [
            "U1",
            "U1",
        ],
        "start": [
            "2024-01-01",
            "2024-01-05",
        ],
        "end": [
            "2024-01-10",
            "2024-01-15",
        ],
    })

    with pytest.raises(
        ValueError,
        match="Overlapping episodes",
    ):
        build_episodes_from_windows(
            windows,
            study_id="test",
            participant_id_column="participant_id",
            source_participant_id_column="uid",
            start_column="start",
            end_column="end",
        )


def test_episode_builder_preserves_study_id():

    windows = pd.DataFrame({
        "participant_id": [
            "P001",
            "P002",
        ],
        "uid": [
            "U1",
            "U2",
        ],
        "start": [
            "2024-01-01",
            "2024-02-01",
        ],
        "end": [
            "2024-01-05",
            "2024-02-05",
        ],
    })

    episodes = build_episodes_from_windows(
        windows,
        study_id="my_study",
        participant_id_column="participant_id",
        source_participant_id_column="uid",
        start_column="start",
        end_column="end",
    )

    assert episodes["study_id"].tolist() == [
        "my_study",
        "my_study",
    ]

    assert not episodes[
        "study_id"
    ].isna().any()
