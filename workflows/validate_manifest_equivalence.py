from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from cortex2_engine.adapters.mindlamp import MindLAMPAdapter


PARTICIPANT_ID = "CHO370"

EPISODES = Path(
    "../cortex2_outputs/digital_clinic/inventory/episodes.parquet"
)

PASSIVE_ROOT = Path("../passive")

MANIFEST_DIR = Path(
    "../cortex2_outputs/digital_clinic/inventory/source_files"
)


def fingerprint(iterator):
    """
    Streaming, order-independent fingerprint of canonical rows.

    Avoids loading an entire stream into memory.
    """
    n_rows = 0
    hash_sum = np.uint64(0)
    hash_xor = np.uint64(0)
    hash_sq_sum = np.uint64(0)

    min_ts = None
    max_ts = None

    file_counts = Counter()
    columns = None

    for chunk in iterator:
        if chunk is None or len(chunk) == 0:
            continue

        cols = sorted(chunk.columns.tolist())

        if columns is None:
            columns = cols
        elif columns != cols:
            raise RuntimeError(
                f"Canonical columns changed between chunks:\n"
                f"{columns}\nvs\n{cols}"
            )

        h = pd.util.hash_pandas_object(
            chunk[cols],
            index=False,
        ).to_numpy(dtype=np.uint64)

        n_rows += len(chunk)

        hash_sum = np.uint64(
            hash_sum + h.sum(dtype=np.uint64)
        )

        if len(h):
            hash_xor = np.uint64(
                hash_xor
                ^ np.bitwise_xor.reduce(h)
            )

            hash_sq_sum = np.uint64(
                hash_sq_sum
                + (h * h).sum(dtype=np.uint64)
            )

        if "timestamp_ms" in chunk.columns:
            lo = int(chunk["timestamp_ms"].min())
            hi = int(chunk["timestamp_ms"].max())

            min_ts = lo if min_ts is None else min(min_ts, lo)
            max_ts = hi if max_ts is None else max(max_ts, hi)

        if "source_file" in chunk.columns:
            file_counts.update(
                chunk["source_file"]
                .astype(str)
                .value_counts()
                .to_dict()
            )

    return {
        "rows": n_rows,
        "hash_sum": int(hash_sum),
        "hash_xor": int(hash_xor),
        "hash_sq_sum": int(hash_sq_sum),
        "min_timestamp_ms": min_ts,
        "max_timestamp_ms": max_ts,
        "file_counts": dict(file_counts),
        "columns": columns,
    }


episodes = pd.read_parquet(EPISODES)

matches = episodes[
    episodes["participant_id"].eq(PARTICIPANT_ID)
]

if len(matches) != 1:
    raise RuntimeError(
        f"Expected exactly one episode for {PARTICIPANT_ID}; "
        f"found {len(matches)}"
    )

ep = matches.iloc[0]

source_id = str(ep["source_participant_id"])
start_ms = int(ep["start_ms"])
end_ms = int(ep["end_ms"])


legacy = MindLAMPAdapter(
    study_id="digital_clinic",
    passive_root=PASSIVE_ROOT,
)

indexed = MindLAMPAdapter(
    study_id="digital_clinic",
    passive_root=PASSIVE_ROOT,
    source_manifest_dir=MANIFEST_DIR,
)


print("Participant:", PARTICIPANT_ID)
print("Episode:", ep["episode_id"])
print("Source ID:", source_id)
print("Window:", ep["start_utc"], "->", ep["end_utc"])


all_ok = True

for stream in [
    "gps",
    "accelerometer",
    "screen",
]:
    print("\n" + "=" * 70)
    print(stream.upper())
    print("=" * 70)

    legacy_paths = list(
        legacy._paths(
            stream,
            source_id,
            start_ms,
            end_ms,
        )
    )

    indexed_paths = list(
        indexed._paths(
            stream,
            source_id,
            start_ms,
            end_ms,
        )
    )

    legacy_bytes = sum(
        p.stat().st_size
        for p in legacy_paths
        if p.exists()
    )

    indexed_bytes = sum(
        p.stat().st_size
        for p in indexed_paths
        if p.exists()
    )

    print(
        f"Legacy candidate files:   {len(legacy_paths):,}"
    )
    print(
        f"Manifest candidate files: {len(indexed_paths):,}"
    )

    print(
        f"Legacy candidate GB:      "
        f"{legacy_bytes / 1024**3:.3f}"
    )

    print(
        f"Manifest candidate GB:    "
        f"{indexed_bytes / 1024**3:.3f}"
    )

    if legacy_bytes:
        reduction = (
            1 - indexed_bytes / legacy_bytes
        ) * 100

        print(
            f"Pre-read byte reduction:  "
            f"{reduction:.2f}%"
        )

    print("\nReading legacy path...")

    a = fingerprint(
        legacy.iter_canonical(
            stream,
            chunksize=250_000,
            source_participant_id=source_id,
            start_ms=start_ms,
            end_ms=end_ms,
        )
    )

    print("Reading manifest-backed path...")

    b = fingerprint(
        indexed.iter_canonical(
            stream,
            chunksize=250_000,
            source_participant_id=source_id,
            start_ms=start_ms,
            end_ms=end_ms,
        )
    )

    checks = {
        "row count":
            a["rows"] == b["rows"],

        "columns":
            a["columns"] == b["columns"],

        "timestamp minimum":
            a["min_timestamp_ms"]
            == b["min_timestamp_ms"],

        "timestamp maximum":
            a["max_timestamp_ms"]
            == b["max_timestamp_ms"],

        "row hash sum":
            a["hash_sum"] == b["hash_sum"],

        "row hash xor":
            a["hash_xor"] == b["hash_xor"],

        "row hash square sum":
            a["hash_sq_sum"]
            == b["hash_sq_sum"],

        "rows by source file":
            a["file_counts"] == b["file_counts"],
    }

    print(
        f"\nLegacy rows:   {a['rows']:,}"
    )

    print(
        f"Manifest rows: {b['rows']:,}"
    )

    for name, ok in checks.items():
        print(
            f"{name:24s}: "
            f"{'MATCH' if ok else 'MISMATCH'}"
        )

    stream_ok = all(checks.values())

    print(
        "\nRESULT:",
        "PASS" if stream_ok else "FAIL",
    )

    all_ok &= stream_ok


print("\n" + "=" * 70)

if all_ok:
    print(
        "OVERALL: PASS — manifest-backed reading "
        "preserves canonical observations."
    )
else:
    print(
        "OVERALL: FAIL — investigate before "
        "using manifest-backed reading."
    )
    raise SystemExit(1)
