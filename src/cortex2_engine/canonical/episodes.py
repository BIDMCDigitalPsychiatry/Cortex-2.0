import hashlib

import pandas as pd


DAY_MS = 86_400_000


def make_episode_id(
    study_id: str,
    participant_id: str,
    source_participant_id: str,
    start_ms: int,
    end_ms: int,
) -> str:
    """
    Stable short episode identifier derived from episode identity.
    """

    raw = (
        f"{study_id}|{participant_id}|"
        f"{source_participant_id}|"
        f"{start_ms}|{end_ms}"
    )

    digest = hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:12]

    return f"{participant_id}__{digest}"


def build_episodes_from_windows(
    df: pd.DataFrame,
    *,
    study_id: str,
    participant_id_column: str,
    source_participant_id_column: str,
    start_column: str,
    end_column: str,
    inclusive_end_date: bool = True,
    source_platform: str | None = None,
    window_name: str = "study",
) -> pd.DataFrame:
    """
    Convert study windows into Cortex 2.0 canonical episodes.

    Date-based inclusive endpoints are converted to half-open
    intervals by adding one day to the end.
    """

    required = {
        participant_id_column,
        source_participant_id_column,
        start_column,
        end_column,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing episode columns: {sorted(missing)}"
        )

    out = pd.DataFrame(
        index=df.index
    )

    out["study_id"] = study_id

    out["participant_id"] = (
        df[participant_id_column]
        .astype(str)
    )

    out["source_participant_id"] = (
        df[source_participant_id_column]
        .astype(str)
    )

    start = pd.to_datetime(
        df[start_column],
        utc=True,
        errors="raise",
    )

    end = pd.to_datetime(
        df[end_column],
        utc=True,
        errors="raise",
    )

    if inclusive_end_date:
        end = end + pd.Timedelta(days=1)

    out["start_utc"] = start
    out["end_utc"] = end

    out["start_ms"] = (
        start.astype("int64")
        // 1_000_000
    ).astype("int64")

    out["end_ms"] = (
        end.astype("int64")
        // 1_000_000
    ).astype("int64")

    if (
        out["end_ms"]
        <= out["start_ms"]
    ).any():
        raise ValueError(
            "Episode end must be after episode start."
        )

    out["window_name"] = window_name

    if source_platform is not None:
        out["source_platform"] = (
            source_platform
        )

    out["episode_id"] = [
        make_episode_id(
            study_id,
            pid,
            sid,
            int(start_ms),
            int(end_ms),
        )
        for pid, sid, start_ms, end_ms in zip(
            out["participant_id"],
            out["source_participant_id"],
            out["start_ms"],
            out["end_ms"],
        )
    ]

    validate_episode_overlaps(out)

    columns = [
        "study_id",
        "episode_id",
        "participant_id",
        "source_participant_id",
        "start_ms",
        "start_utc",
        "end_ms",
        "end_utc",
        "window_name",
    ]

    if source_platform is not None:
        columns.append(
            "source_platform"
        )

    return out[columns].reset_index(
        drop=True
    )


def validate_episode_overlaps(
    episodes: pd.DataFrame,
):
    """
    Reject overlapping episodes assigned to different participants
    for the same source identity.

    Adjacent episodes are allowed because intervals are half-open.
    """

    for source_id, group in episodes.groupby(
        "source_participant_id",
        sort=False,
    ):

        g = group.sort_values(
            "start_ms"
        )

        previous = None

        for row in g.itertuples(
            index=False
        ):

            if (
                previous is not None
                and row.start_ms
                < previous.end_ms
                and row.participant_id
                != previous.participant_id
            ):
                raise ValueError(
                    "Overlapping episodes for source identity "
                    f"{source_id}: "
                    f"{previous.participant_id} and "
                    f"{row.participant_id}"
                )

            if (
                previous is None
                or row.end_ms
                > previous.end_ms
            ):
                previous = row
