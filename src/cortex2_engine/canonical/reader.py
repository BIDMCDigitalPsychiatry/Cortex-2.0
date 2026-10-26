from collections.abc import Iterator

import pandas as pd

from cortex2_engine.adapters.base import BaseAdapter


class EpisodeReader:
    """
    Read canonical sensor data for an explicitly defined episode.

    Raw observations are selected using:
        source_participant_id + [start_ms, end_ms)

    Study identity is attached only after that temporal selection.
    """

    def __init__(
        self,
        adapter: BaseAdapter,
        episodes: pd.DataFrame,
    ):
        self.adapter = adapter
        self.episodes = episodes.copy()

        required = {
            "episode_id",
            "participant_id",
            "source_participant_id",
            "start_ms",
            "end_ms",
        }

        missing = (
            required
            - set(self.episodes.columns)
        )

        if missing:
            raise ValueError(
                f"Episode table missing: {sorted(missing)}"
            )

        if self.episodes[
            "episode_id"
        ].duplicated().any():
            raise ValueError(
                "episode_id must be unique."
            )

    def episode(
        self,
        episode_id: str,
    ) -> pd.Series:

        matches = self.episodes[
            self.episodes[
                "episode_id"
            ].astype(str).eq(
                str(episode_id)
            )
        ]

        if len(matches) != 1:
            raise KeyError(
                f"Unknown episode_id: {episode_id}"
            )

        return matches.iloc[0]

    def iter_stream(
        self,
        stream: str,
        *,
        episode_id: str,
        chunksize: int = 1_000_000,
    ) -> Iterator[pd.DataFrame]:

        ep = self.episode(
            episode_id
        )

        for chunk in self.adapter.iter_canonical(
            stream,
            chunksize=chunksize,
            source_participant_id=str(
                ep["source_participant_id"]
            ),
            start_ms=int(
                ep["start_ms"]
            ),
            end_ms=int(
                ep["end_ms"]
            ),
        ):

            out = chunk.copy()

            # Episode assignment happens here, after
            # source identity + temporal selection.
            out["participant_id"] = str(
                ep["participant_id"]
            )

            out["episode_id"] = str(
                ep["episode_id"]
            )

            yield out
