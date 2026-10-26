from pathlib import Path
import re

import pandas as pd
import yaml

from cortex2_engine.adapters.base import BaseAdapter
from cortex2_engine.canonical.validation import (
    validate_canonical,
)


class GenericCSVAdapter(BaseAdapter):

    adapter_name = "generic_csv"
    adapter_version = "1.0"

    def __init__(
        self,
        config_path: str | Path,
    ):
        self.config_path = Path(
            config_path
        ).resolve()

        with self.config_path.open("r") as f:
            self.config = yaml.safe_load(f)

        self.study_id = str(
            self.config["study_id"]
        )

        self.source_platform = str(
            self.config.get(
                "source_platform",
                "generic",
            )
        )

        self.base_dir = self.config_path.parent

    def available_streams(self) -> list[str]:
        return sorted(
            self.config.get(
                "streams",
                {}
            ).keys()
        )

    def _stream_config(
        self,
        stream,
    ):
        streams = self.config.get(
            "streams",
            {}
        )

        if stream not in streams:
            raise ValueError(
                f"Stream '{stream}' is not configured. "
                f"Available: {sorted(streams)}"
            )

        return streams[stream]

    def _resolve_path(
        self,
        value,
    ) -> Path:

        p = Path(value)

        if not p.is_absolute():
            p = (
                self.base_dir
                / p
            ).resolve()

        return p

    def _participant_from_filename(
        self,
        filename: str,
        pattern: str,
    ) -> str:

        match = re.search(
            pattern,
            filename,
        )

        if not match:
            raise ValueError(
                "Could not extract participant_id from "
                f"filename: {filename}"
            )

        if "participant_id" not in match.groupdict():
            raise ValueError(
                "participant_regex must contain named group "
                "'participant_id'"
            )

        return str(
            match.group(
                "participant_id"
            )
        )

    def _timestamp_ms(
        self,
        series: pd.Series,
        config: dict,
    ) -> pd.Series:

        unit = config.get(
            "unit",
            "ms",
        )

        if unit in {
            "ms",
            "s",
            "us",
            "ns",
        }:

            values = pd.to_numeric(
                series,
                errors="coerce",
            )

            factors = {
                "s": 1000,
                "ms": 1,
                "us": 1 / 1000,
                "ns": 1 / 1_000_000,
            }

            return (
                values
                * factors[unit]
            ).round().astype(
                "Int64"
            )

        # Otherwise interpret as datetime text.
        timezone = config.get(
            "timezone",
            "UTC",
        )

        dt = pd.to_datetime(
            series,
            errors="coerce",
        )

        if dt.dt.tz is None:
            dt = (
                dt.dt.tz_localize(
                    timezone
                )
            )

        dt = dt.dt.tz_convert(
            "UTC"
        )

        return (
            dt.astype("int64")
            // 1_000_000
        ).astype("Int64")

    def _canonicalize_chunk(
        self,
        chunk,
        stream,
        cfg,
        path,
    ):
        columns = cfg.get(
            "columns",
            {}
        )

        out = pd.DataFrame(
            index=chunk.index
        )

        # ----------------------------
        # Participant
        # ----------------------------

        participant_source = cfg.get(
            "participant"
        )

        if participant_source:
            source_col = (
                participant_source.get(
                    "column"
                )
            )

            regex = (
                participant_source.get(
                    "filename_regex"
                )
            )
        else:
            source_col = None
            regex = None

        if source_col:
            out["participant_id"] = (
                chunk[source_col]
                .astype(str)
            )

        elif regex:
            pid = (
                self
                ._participant_from_filename(
                    path.name,
                    regex,
                )
            )

            out["participant_id"] = pid

        else:
            raise ValueError(
                f"{stream}: configure participant.column "
                "or participant.filename_regex"
            )

        # Unless a dataset supplies a separate study-level
        # identity mapping, the source identity and canonical
        # participant identity initially coincide.
        out["source_participant_id"] = (
            out["participant_id"].astype(str)
        )

        # ----------------------------
        # Shared fields
        # ----------------------------

        out["study_id"] = (
            self.study_id
        )

        out["source_platform"] = (
            self.source_platform
        )

        out["source_stream"] = (
            stream
        )

        out["source_file"] = str(
            path
        )

        # ----------------------------
        # Timestamp
        # ----------------------------

        timestamp_col = columns.get(
            "timestamp"
        )

        if not timestamp_col:
            raise ValueError(
                f"{stream}: timestamp column not configured"
            )

        out["timestamp_ms"] = (
            self._timestamp_ms(
                chunk[timestamp_col],
                cfg.get(
                    "timestamp",
                    {}
                ),
            )
        )

        out["timestamp_utc"] = (
            pd.to_datetime(
                out["timestamp_ms"],
                unit="ms",
                utc=True,
                errors="coerce",
            )
        )

        # ----------------------------
        # Stream-specific columns
        # ----------------------------

        skip = {
            "timestamp",
            "participant_id",
        }

        for canonical, source in columns.items():

            if canonical in skip:
                continue

            if source not in chunk.columns:
                raise ValueError(
                    f"{stream}: source column '{source}' "
                    f"for '{canonical}' not found in {path}"
                )

            out[canonical] = (
                chunk[source]
            )

        # ----------------------------
        # Fixed values
        # ----------------------------

        for key, value in (
            cfg.get(
                "constants",
                {}
            ).items()
        ):
            out[key] = value

        # ----------------------------
        # Screen event mapping
        # ----------------------------

        if (
            stream == "screen"
            and "event" in out.columns
        ):

            original = (
                out["event"]
                .astype(str)
            )

            out[
                "representation_original"
            ] = original

            mapping = cfg.get(
                "event_mapping",
                {}
            )

            out["event"] = (
                original
                .map(mapping)
                .fillna("unknown")
            )

        return out.reset_index(
            drop=True
        )

    def iter_canonical(
        self,
        stream: str,
        *,
        chunksize: int = 1_000_000,
        source_participant_id: str | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ):

        cfg = self._stream_config(
            stream
        )

        directory = self._resolve_path(
            cfg["path"]
        )

        pattern = cfg.get(
            "file_pattern",
            "*.csv",
        )

        paths = sorted(
            directory.glob(pattern)
        )

        for path in paths:

            for chunk in pd.read_csv(
                path,
                chunksize=chunksize,
            ):

                canonical = (
                    self._canonicalize_chunk(
                        chunk,
                        stream,
                        cfg,
                        path,
                    )
                )

                if source_participant_id is not None:
                    canonical = canonical[
                        canonical[
                            "source_participant_id"
                        ].astype(str).eq(
                            str(source_participant_id)
                        )
                    ]

                if start_ms is not None:
                    canonical = canonical[
                        canonical[
                            "timestamp_ms"
                        ] >= start_ms
                    ]

                if end_ms is not None:
                    canonical = canonical[
                        canonical[
                            "timestamp_ms"
                        ] < end_ms
                    ]

                if len(canonical) == 0:
                    continue

                canonical = canonical.reset_index(
                    drop=True
                )

                report = validate_canonical(
                    canonical,
                    stream,
                )

                if not report.valid:
                    raise ValueError(
                        f"Canonical validation failed for "
                        f"{path}: {report.errors}"
                    )

                yield canonical
