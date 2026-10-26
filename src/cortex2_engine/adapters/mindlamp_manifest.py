from pathlib import Path

import pandas as pd


STREAMS = {
    "gps": {
        "folder": "gps",
        "prefix": "gps",
    },
    "accelerometer": {
        "folder": "acc",
        "prefix": "acc",
    },
    "screen": {
        "folder": "screen",
        "prefix": "screen",
    },
    "device_usage": {
        "folder": "device_usage",
        "prefix": "device_usage",
    },
    "nearby_device": {
        "folder": "nearby_device",
        "prefix": "nearby_device",
    },
}


def source_id_from_filename(
    path: Path,
    prefix: str,
) -> str:

    stem = path.stem
    expected = f"{prefix}_"

    if not stem.startswith(expected):
        raise ValueError(
            f"Unexpected filename: {path.name}"
        )

    return stem[
        len(expected):
    ].split("_", 1)[0]


def list_mindlamp_files(
    passive_root,
    stream,
):
    """
    Enumerate raw files for one MindLAMP stream.
    """

    if stream not in STREAMS:
        raise ValueError(
            f"Unsupported stream: {stream}"
        )

    passive_root = Path(
        passive_root
    ).resolve()

    cfg = STREAMS[stream]

    root = (
        passive_root
        / cfg["folder"]
    )

    return sorted(
        root.glob(
            f"{cfg['prefix']}_*.csv"
        )
    )


def scan_timestamp_bounds(
    path,
    *,
    chunksize=1_000_000,
):
    """
    Read only the timestamp column and return raw file
    temporal bounds.

    No behavioral data are interpreted here.
    """

    n_rows = 0
    n_valid_timestamps = 0

    min_timestamp_ms = None
    max_timestamp_ms = None

    for chunk in pd.read_csv(
        path,
        usecols=["timestamp"],
        chunksize=chunksize,
    ):

        n_rows += len(chunk)

        ts = pd.to_numeric(
            chunk["timestamp"],
            errors="coerce",
        ).dropna()

        n_valid_timestamps += len(ts)

        if len(ts) == 0:
            continue

        cmin = int(ts.min())
        cmax = int(ts.max())

        min_timestamp_ms = (
            cmin
            if min_timestamp_ms is None
            else min(
                min_timestamp_ms,
                cmin,
            )
        )

        max_timestamp_ms = (
            cmax
            if max_timestamp_ms is None
            else max(
                max_timestamp_ms,
                cmax,
            )
        )

    return {
        "n_rows":
            int(n_rows),

        "n_valid_timestamps":
            int(n_valid_timestamps),

        "min_timestamp_ms":
            min_timestamp_ms,

        "max_timestamp_ms":
            max_timestamp_ms,
    }


def file_manifest_record(
    path,
    *,
    stream,
    chunksize=1_000_000,
):
    """
    Build one standardized source-file manifest record.
    """

    path = Path(path).resolve()

    cfg = STREAMS[stream]

    stat = path.stat()

    record = {
        "stream":
            stream,

        "source_platform":
            "mindlamp",

        "source_participant_id":
            source_id_from_filename(
                path,
                cfg["prefix"],
            ),

        "source_file":
            str(path),

        "filename":
            path.name,

        "size_bytes":
            int(stat.st_size),

        "modified_ns":
            int(stat.st_mtime_ns),

        "scan_status":
            "ok",

        "scan_error":
            None,
    }

    try:
        bounds = scan_timestamp_bounds(
            path,
            chunksize=chunksize,
        )

        record.update(
            bounds
        )

    except Exception as e:

        record.update({
            "n_rows":
                None,

            "n_valid_timestamps":
                None,

            "min_timestamp_ms":
                None,

            "max_timestamp_ms":
                None,

            "scan_status":
                "error",

            "scan_error":
                repr(e),
        })

    return record
