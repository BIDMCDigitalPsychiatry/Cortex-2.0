import numpy as np
import pandas as pd

from cortex2_engine.canonical.reader import (
    EpisodeReader,
)
from cortex2_engine.quality.episode import (
    DAY_MS,
    episode_daily_quality,
)


class FakeAdapter:

    def __init__(self, df):
        self.df = df

    def iter_canonical(
        self,
        stream,
        *,
        chunksize=1_000_000,
        source_participant_id=None,
        start_ms=None,
        end_ms=None,
    ):
        z = self.df.copy()

        if source_participant_id is not None:
            z = z[
                z[
                    "source_participant_id"
                ].eq(
                    source_participant_id
                )
            ]

        if start_ms is not None:
            z = z[
                z["timestamp_ms"]
                >= start_ms
            ]

        if end_ms is not None:
            z = z[
                z["timestamp_ms"]
                < end_ms
            ]

        for start in range(
            0,
            len(z),
            chunksize,
        ):
            yield z.iloc[
                start:
                start + chunksize
            ].copy()


def make_reader():
    episodes = pd.DataFrame([
        {
            "study_id": "S1",
            "episode_id": "E1",
            "participant_id": "P1",
            "source_participant_id": "U1",
            "start_ms": 0,
            "end_ms": 2 * DAY_MS,
        }
    ])

    raw = pd.DataFrame({
        "source_participant_id": [
            "U1",
            "U1",
        ],
        "timestamp_ms": [
            1_000,
            601_000,
        ],
    })

    return EpisodeReader(
        FakeAdapter(raw),
        episodes,
    )


def test_episode_quality_source_aware():
    reader = make_reader()

    source_intervals = pd.DataFrame([
        {
            "episode_id": "E1",
            "stream": "gps",
            "source_start_ms": None,
            "source_end_ms": DAY_MS,
            "boundary_source": "test",
        }
    ])

    out = episode_daily_quality(
        reader,
        source_intervals,
        episode_id="E1",
        stream="gps",
        chunksize=1,
    )

    assert len(out) == 2

    assert (
        out.loc[
            0,
            "n_observations",
        ]
        == 2
    )

    assert (
        out.loc[
            0,
            "n_occupied_bins",
        ]
        == 2
    )

    assert np.isclose(
        out.loc[
            0,
            "data_quality",
        ],
        2 / 144,
    )

    assert (
        out.loc[
            1,
            "source_status",
        ]
        == "outside_declared_end"
    )

    assert (
        out.loc[
            1,
            "raw_data_quality",
        ]
        == 0.0
    )

    assert np.isnan(
        out.loc[
            1,
            "data_quality",
        ]
    )


def test_unknown_boundary_is_preserved():
    reader = make_reader()

    source_intervals = pd.DataFrame(
        columns=[
            "episode_id",
            "stream",
            "source_start_ms",
            "source_end_ms",
            "boundary_source",
        ]
    )

    out = episode_daily_quality(
        reader,
        source_intervals,
        episode_id="E1",
        stream="gps",
    )

    assert out[
        "boundary_unknown"
    ].all()

    assert (
        out.loc[
            1,
            "data_quality",
        ]
        == 0.0
    )
