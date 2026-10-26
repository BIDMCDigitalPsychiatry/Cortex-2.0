from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from cortex2_engine.adapters.base import BaseAdapter
from cortex2_engine.canonical.validation import validate_canonical


class VirtualCanonicalStore:
    """
    Canonical access without copying source data.

    The adapter performs source -> canonical transformation
    lazily as chunks are requested.
    """

    mode = "virtual"

    def __init__(
        self,
        adapter: BaseAdapter,
    ):
        self.adapter = adapter

    def available_streams(self) -> list[str]:
        return self.adapter.available_streams()

    def iter_stream(
        self,
        stream: str,
        *,
        chunksize: int = 1_000_000,
        **kwargs,
    ) -> Iterator[pd.DataFrame]:

        yield from self.adapter.iter_canonical(
            stream,
            chunksize=chunksize,
            **kwargs,
        )

    def provenance(self) -> dict:
        return {
            "storage_mode": self.mode,
            **self.adapter.provenance(),
        }


def sha256_file(
    path: Path,
    chunk_bytes: int = 1024 * 1024,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_bytes)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


class MaterializedCanonicalStore:
    """
    Partitioned Parquet representation of Cortex 2.0 canonical data.

    Layout:
        root/
            gps/
                participant_id=P001/
                    part-000001.parquet
            accelerometer/
            screen/
            manifest.parquet
            provenance.json
    """

    mode = "materialized"

    def __init__(
        self,
        root: str | Path,
    ):
        self.root = Path(root).resolve()
        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _stream_root(
        self,
        stream: str,
    ) -> Path:

        path = self.root / stream

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        return path

    def materialize(
        self,
        adapter: BaseAdapter,
        *,
        streams: list[str] | None = None,
        chunksize: int = 1_000_000,
        checksum: bool = True,
    ) -> pd.DataFrame:

        if streams is None:
            streams = (
                adapter.available_streams()
            )

        manifest_rows = []

        counters = {}

        for stream in streams:

            print(
                f"Materializing stream: {stream}",
                flush=True,
            )

            for chunk in adapter.iter_canonical(
                stream,
                chunksize=chunksize,
            ):

                report = validate_canonical(
                    chunk,
                    stream,
                )

                if not report.valid:
                    raise ValueError(
                        f"{stream}: "
                        f"{report.errors}"
                    )

                # A generic adapter may yield multiple
                # participants in the same chunk.
                for participant_id, part in (
                    chunk.groupby(
                        "participant_id",
                        sort=False,
                    )
                ):

                    participant_id = str(
                        participant_id
                    )

                    key = (
                        stream,
                        participant_id,
                    )

                    index = counters.get(
                        key,
                        0,
                    )

                    counters[key] = (
                        index + 1
                    )

                    directory = (
                        self._stream_root(stream)
                        / (
                            "participant_id="
                            f"{participant_id}"
                        )
                    )

                    directory.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    path = (
                        directory
                        / f"part-{index:06d}.parquet"
                    )

                    part.to_parquet(
                        path,
                        index=False,
                        compression="zstd",
                    )

                    row = {
                        "stream":
                            stream,

                        "participant_id":
                            participant_id,

                        "path":
                            str(path),

                        "rows":
                            int(len(part)),

                        "size_bytes":
                            int(
                                path.stat().st_size
                            ),

                        "timestamp_min_ms":
                            int(
                                part[
                                    "timestamp_ms"
                                ].min()
                            ),

                        "timestamp_max_ms":
                            int(
                                part[
                                    "timestamp_ms"
                                ].max()
                            ),
                    }

                    if checksum:
                        row["sha256"] = (
                            sha256_file(path)
                        )

                    manifest_rows.append(
                        row
                    )

        manifest = pd.DataFrame(
            manifest_rows
        )

        manifest_path = (
            self.root
            / "manifest.parquet"
        )

        manifest.to_parquet(
            manifest_path,
            index=False,
        )

        provenance = {
            "created_at_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "storage_mode":
                self.mode,

            "adapter":
                adapter.provenance(),

            "streams":
                streams,

            "files":
                int(len(manifest)),

            "rows":
                (
                    int(
                        manifest["rows"].sum()
                    )
                    if len(manifest)
                    else 0
                ),
        }

        with (
            self.root
            / "provenance.json"
        ).open("w") as f:

            json.dump(
                provenance,
                f,
                indent=2,
            )

        return manifest

    def available_streams(
        self,
    ) -> list[str]:

        manifest = self.manifest()

        if len(manifest) == 0:
            return []

        return sorted(
            manifest[
                "stream"
            ].unique()
        )

    def manifest(
        self,
    ) -> pd.DataFrame:

        path = (
            self.root
            / "manifest.parquet"
        )

        if not path.exists():
            return pd.DataFrame()

        return pd.read_parquet(
            path
        )

    def iter_stream(
        self,
        stream: str,
        *,
        participant_id: str | None = None,
    ) -> Iterator[pd.DataFrame]:

        manifest = self.manifest()

        files = manifest[
            manifest["stream"].eq(
                stream
            )
        ]

        if participant_id is not None:
            files = files[
                files[
                    "participant_id"
                ].astype(str).eq(
                    str(participant_id)
                )
            ]

        for path in files["path"]:
            yield pd.read_parquet(
                path
            )
