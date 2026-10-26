from pathlib import Path
import hashlib
from collections.abc import Iterator

import pandas as pd

from cortex2_engine.adapters.base import BaseAdapter
from cortex2_engine.io.source_index import SourceFileIndex
from cortex2_engine.canonical.validation import validate_canonical


class MindLAMPAdapter(BaseAdapter):
    """
    Convert exported mindLAMP passive sensor CSVs into the
    Cortex 2.0 Canonical Data Model.

    Expected filenames
    ------------------
    GPS:
        gps_<mindlamp_uid>_*.csv

    Accelerometer:
        acc_<mindlamp_uid>_*.csv

    Screen:
        screen_<mindlamp_uid>_*.csv

    The adapter does not perform feature engineering,
    filtering by data quality, interpolation, or imputation.
    """

    adapter_name = "mindlamp"
    adapter_version = "1.0"

    STREAM_PREFIX = {
        "gps": "gps",
        "accelerometer": "acc",
        "screen": "screen",
        "device_usage": "device_usage",
        "nearby_device": "nearby_device",
    }

    SCHEMA_NAME = {
        "gps": "gps",
        "accelerometer": "accelerometer",
        "screen": "screen",
        "device_usage": "device_usage",
        "nearby_device": "nearby_device",
    }

    def __init__(
        self,
        *,
        study_id: str,
        passive_root: str | Path,
        participant_map: str | Path | None = None,
        source_id_column: str = "mindlamp_uid",
        participant_id_column: str = "participant_id",
        source_manifest_dir: str | Path | None = None,
    ):
        self.study_id = str(study_id)

        self.passive_root = Path(
            passive_root
        ).resolve()

        self.source_platform = "mindlamp"

        self.source_index = (
            SourceFileIndex(
                source_manifest_dir
            )
            if source_manifest_dir is not None
            else None
        )

        self.source_id_column = source_id_column
        self.participant_id_column = participant_id_column

        self.id_map = {}
        self.ambiguous_source_ids = set()
        self.mapping_supplied = (
            participant_map is not None
        )

        if participant_map is not None:
            m = pd.read_csv(
                participant_map
            )

            required = {
                source_id_column,
                participant_id_column,
            }

            missing = required - set(m.columns)

            if missing:
                raise ValueError(
                    "Participant map missing columns: "
                    f"{sorted(missing)}"
                )

            m[source_id_column] = (
                m[source_id_column]
                .astype(str)
            )

            m[participant_id_column] = (
                m[participant_id_column]
                .astype(str)
            )

            grouped = (
                m.groupby(
                    source_id_column
                )[participant_id_column]
                .agg(
                    lambda x: sorted(
                        set(x)
                    )
                )
            )

            for source_id, participant_ids in (
                grouped.items()
            ):
                if len(participant_ids) == 1:
                    self.id_map[
                        source_id
                    ] = participant_ids[0]
                else:
                    self.ambiguous_source_ids.add(
                        source_id
                    )

    def available_streams(self) -> list[str]:
        available = []

        for stream, prefix in self.STREAM_PREFIX.items():

            directory = (
                self.passive_root
                / prefix
            )

            if (
                directory.exists()
                and any(
                    directory.glob(
                        f"{prefix}_*.csv"
                    )
                )
            ):
                available.append(stream)

        return sorted(available)

    def _cortex2_participant_id(
        self,
        source_participant_id: str,
    ) -> str:

        source_participant_id = str(
            source_participant_id
        )

        if (
            source_participant_id
            in self.ambiguous_source_ids
        ):
            return None

        if source_participant_id in self.id_map:
            return self.id_map[
                source_participant_id
            ]

        # If a mapping table was supplied, absence from that
        # table should not silently create a study identity.
        if self.mapping_supplied:
            return None

        # With no external mapping, source identity is the
        # only participant identity available.
        return source_participant_id

    def _paths(
        self,
        stream: str,
        source_participant_id: str | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ):

        if stream not in self.STREAM_PREFIX:
            raise ValueError(
                f"Unsupported mindLAMP stream: {stream}"
            )

        if self.source_index is not None:
            return self.source_index.candidate_files(
                stream,
                source_participant_id=source_participant_id,
                start_ms=start_ms,
                end_ms=end_ms,
            )

        prefix = self.STREAM_PREFIX[
            stream
        ]

        directory = (
            self.passive_root
            / prefix
        )

        if source_participant_id:
            pattern = (
                f"{prefix}_"
                f"{source_participant_id}_*.csv"
            )
        else:
            pattern = (
                f"{prefix}_*.csv"
            )

        return sorted(
            directory.glob(pattern)
        )

    @staticmethod
    def _uid_from_filename(
        path: Path,
        prefix: str,
    ) -> str:

        name = path.stem

        expected = f"{prefix}_"

        if not name.startswith(expected):
            raise ValueError(
                f"Unexpected filename: {path.name}"
            )

        remainder = name[
            len(expected):
        ]

        # MindLAMP UID is before the next underscore.
        return remainder.split(
            "_",
            1
        )[0]

    @staticmethod
    def _timestamps(
        chunk: pd.DataFrame,
    ):

        ms = pd.to_numeric(
            chunk["timestamp"],
            errors="coerce",
        ).astype("Int64")

        utc = pd.to_datetime(
            ms,
            unit="ms",
            utc=True,
            errors="coerce",
        )

        return ms, utc

    def _shared_fields(
        self,
        chunk,
        *,
        uid,
        stream,
        path,
    ):

        out = pd.DataFrame(
            index=chunk.index
        )

        timestamp_ms, timestamp_utc = (
            self._timestamps(chunk)
        )

        out["study_id"] = (
            self.study_id
        )

        out["participant_id"] = (
            self._cortex2_participant_id(
                uid
            )
        )

        out["source_participant_id"] = (
            uid
        )

        out["timestamp_ms"] = (
            timestamp_ms
        )

        out["timestamp_utc"] = (
            timestamp_utc
        )

        out["source_platform"] = (
            self.source_platform
        )

        out["source_stream"] = (
            stream
        )

        out["source_file"] = str(
            path.resolve()
        )

        # Preserve original exported record index when present.
        unnamed = [
            c for c in chunk.columns
            if str(c).startswith("Unnamed:")
        ]

        if unnamed:
            out["source_record_id"] = (
                chunk[unnamed[0]]
                .astype(str)
            )

        return out

    def _gps(
        self,
        chunk,
        uid,
        path,
    ):

        out = self._shared_fields(
            chunk,
            uid=uid,
            stream="gps",
            path=path,
        )

        out["latitude"] = pd.to_numeric(
            chunk["latitude"],
            errors="coerce",
        )

        out["longitude"] = pd.to_numeric(
            chunk["longitude"],
            errors="coerce",
        )

        if "altitude" in chunk:
            out["altitude_m"] = (
                pd.to_numeric(
                    chunk["altitude"],
                    errors="coerce",
                )
            )

        if "accuracy" in chunk:
            out["accuracy_m"] = (
                pd.to_numeric(
                    chunk["accuracy"],
                    errors="coerce",
                )
            )

        return out

    def _accelerometer(
        self,
        chunk,
        uid,
        path,
    ):

        out = self._shared_fields(
            chunk,
            uid=uid,
            stream="accelerometer",
            path=path,
        )

        for axis in ["x", "y", "z"]:
            out[axis] = pd.to_numeric(
                chunk[axis],
                errors="coerce",
            )

        # Do not guess physical units.
        out["unit"] = "unknown"

        return out

    def _screen(
        self,
        chunk,
        uid,
        path,
    ):

        out = self._shared_fields(
            chunk,
            uid=uid,
            stream="screen",
            path=path,
        )

        original = (
            chunk["representation"]
            .astype(str)
        )

        out[
            "representation_original"
        ] = original

        mapping = {
            "screen_on": "screen_on",
            "screen_off": "screen_off",
            "locked": "lock",
            "unlocked": "unlock",

            # Also support already-normalized variants.
            "lock": "lock",
            "unlock": "unlock",
        }

        out["event"] = (
            original
            .map(mapping)
            .fillna("unknown")
        )

        if "value" in chunk:
            out["value"] = pd.to_numeric(
                chunk["value"],
                errors="coerce",
            )

        if "battery_level" in chunk:
            out["battery_level_fraction"] = (
                pd.to_numeric(
                    chunk["battery_level"],
                    errors="coerce",
                )
            )

        return out


    def _device_usage(
        self,
        chunk,
        uid,
        path,
    ):

        out = self._shared_fields(
            chunk,
            uid=uid,
            stream="device_usage",
            path=path,
        )

        numeric_map = {
            "duration":
                "interval_duration_ms",

            "totalUnlockDuration":
                "total_unlock_duration_ms",

            "totalUnlocks":
                "total_unlocks",

            "totalScreenWakes":
                "total_screen_wakes",
        }

        for source_col, canonical_col in (
            numeric_map.items()
        ):
            if source_col in chunk:
                out[canonical_col] = (
                    pd.to_numeric(
                        chunk[source_col],
                        errors="coerce",
                    )
                )

        raw_map = {
            "applicationUsageByCategory":
                "application_usage_by_category_raw",

            "notificationUsageByCategory":
                "notification_usage_by_category_raw",

            "webUsageByCategory":
                "web_usage_by_category_raw",
        }

        for source_col, canonical_col in (
            raw_map.items()
        ):
            if source_col in chunk:
                out[canonical_col] = (
                    chunk[source_col]
                    .astype("string")
                )

        return out


    def _nearby_identifier_hash(
        self,
        device_type,
        address,
    ):
        """
        Produce a deterministic study-scoped pseudonymous token.

        Raw network/device addresses and names are intentionally
        not propagated into the Cortex 2.0 canonical output.
        """

        if pd.isna(address):
            return None

        raw = (
            f"{self.study_id}|"
            f"{self.source_platform}|"
            f"{device_type}|"
            f"{address}"
        )

        return hashlib.blake2b(
            raw.encode("utf-8"),
            digest_size=16,
        ).hexdigest()


    def _nearby_device(
        self,
        chunk,
        uid,
        path,
    ):

        out = self._shared_fields(
            chunk,
            uid=uid,
            stream="nearby_device",
            path=path,
        )

        if "type" in chunk:
            out["device_type"] = (
                chunk["type"]
                .astype("string")
                .str.lower()
            )

        if "strength" in chunk:
            out["signal_strength_raw"] = (
                pd.to_numeric(
                    chunk["strength"],
                    errors="coerce",
                )
            )

        if "address" in chunk:

            if "type" in chunk:
                types = (
                    chunk["type"]
                    .astype("string")
                    .str.lower()
                )
            else:
                types = pd.Series(
                    [None] * len(chunk),
                    index=chunk.index,
                )

            out["nearby_identifier_hash"] = [
                self._nearby_identifier_hash(
                    device_type,
                    address,
                )
                for device_type, address in zip(
                    types,
                    chunk["address"],
                )
            ]

        return out


    def iter_canonical(
        self,
        stream: str,
        *,
        chunksize: int = 1_000_000,
        source_participant_id: str | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> Iterator[pd.DataFrame]:

        if stream not in self.STREAM_PREFIX:
            raise ValueError(
                f"Unsupported stream: {stream}"
            )

        prefix = self.STREAM_PREFIX[
            stream
        ]

        schema = self.SCHEMA_NAME[
            stream
        ]

        paths = self._paths(
            stream,
            source_participant_id,
            start_ms,
            end_ms,
        )

        for path in paths:

            uid = self._uid_from_filename(
                path,
                prefix,
            )

            for chunk in pd.read_csv(
                path,
                chunksize=chunksize,
            ):

                if stream == "gps":
                    out = self._gps(
                        chunk,
                        uid,
                        path,
                    )

                elif stream == "accelerometer":
                    out = self._accelerometer(
                        chunk,
                        uid,
                        path,
                    )

                elif stream == "screen":
                    out = self._screen(
                        chunk,
                        uid,
                        path,
                    )

                elif stream == "device_usage":
                    out = self._device_usage(
                        chunk,
                        uid,
                        path,
                    )

                elif stream == "nearby_device":
                    out = self._nearby_device(
                        chunk,
                        uid,
                        path,
                    )

                else:
                    raise ValueError(
                        f"Unsupported stream: {stream}"
                    )

                out = out.reset_index(
                    drop=True
                )

                if start_ms is not None:
                    out = out[
                        out["timestamp_ms"] >= start_ms
                    ]

                if end_ms is not None:
                    out = out[
                        out["timestamp_ms"] < end_ms
                    ]

                if len(out) == 0:
                    continue

                out = out.reset_index(
                    drop=True
                )

                report = validate_canonical(
                    out,
                    schema,
                )

                if not report.valid:
                    raise ValueError(
                        f"{path.name}: "
                        f"{report.errors}"
                    )

                yield out
