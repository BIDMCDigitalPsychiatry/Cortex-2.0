from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import ClassVar

import pandas as pd

from cortex2_engine.adapters.base import BaseAdapter
from cortex2_engine.canonical.validation import validate_canonical
from cortex2_engine.io.source_index import SourceFileIndex


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

    Device usage / nearby devices:
        device_usage_<mindlamp_uid>_*.csv
        nearby_device_<mindlamp_uid>_*.csv

    Telephony, surveys, cognitive tasks, steps, sleep:
        telephony_<mindlamp_uid>_*.csv
        survey_<mindlamp_uid>_*.csv
        cognitive_<mindlamp_uid>_*.csv
        steps_<mindlamp_uid>_*.csv
        sleep_<mindlamp_uid>_*.csv

    The column names accepted for the last five streams are documented on
    each ``_<stream>()`` method. They follow mindLAMP sensor/activity field
    names (``lamp.calls``, ``lamp.sms``, ``lamp.steps``, ``lamp.sleep``,
    activity temporal slices) and are assumptions until checked against a
    real export (docs/open_questions.md).

    Battery is not a separate stream: it rides on
    ``screen.battery_level_fraction``.

    The adapter does not perform feature engineering,
    filtering by data quality, interpolation, or imputation.
    """

    adapter_name = "mindlamp"
    adapter_version = "1.0"

    STREAM_PREFIX: ClassVar[dict[str, str]] = {
        "gps": "gps",
        "accelerometer": "acc",
        "screen": "screen",
        "device_usage": "device_usage",
        "nearby_device": "nearby_device",
        "telephony": "telephony",
        "survey": "survey",
        "cognitive": "cognitive",
        "steps": "steps",
        "sleep": "sleep",
    }

    SCHEMA_NAME: ClassVar[dict[str, str]] = {
        "gps": "gps",
        "accelerometer": "accelerometer",
        "screen": "screen",
        "device_usage": "device_usage",
        "nearby_device": "nearby_device",
        "telephony": "telephony",
        "survey": "survey",
        "cognitive": "cognitive",
        "steps": "steps",
        "sleep": "sleep",
    }

    # Direction vocabularies. Integer codes follow Android CallLog.Calls.TYPE
    # (1 incoming, 2 outgoing, 3 missed) and Telephony.Sms.TYPE (1 inbox,
    # 2 sent), which agree on 1 and 2.
    TELEPHONY_DIRECTION: ClassVar[dict[str, str]] = {
        "incoming": "incoming",
        "inbound": "incoming",
        "received": "incoming",
        "inbox": "incoming",
        "outgoing": "outgoing",
        "outbound": "outgoing",
        "sent": "outgoing",
        "missed": "missed",
        "1": "incoming",
        "2": "outgoing",
        "3": "missed",
    }

    TELEPHONY_CHANNEL: ClassVar[dict[str, str]] = {
        "call": "call",
        "calls": "call",
        "lamp.calls": "call",
        "sms": "sms",
        "lamp.sms": "sms",
        "text": "sms",
    }

    # HealthKit HKCategoryValueSleepAnalysis codes and common spellings.
    SLEEP_REPRESENTATION: ClassVar[dict[str, str]] = {
        "0": "in_bed",
        "1": "in_sleep",
        "2": "in_awake",
        "3": "in_sleep",
        "4": "in_sleep",
        "5": "in_sleep",
        "inbed": "in_bed",
        "insleep": "in_sleep",
        "asleep": "in_sleep",
        "sleep": "in_sleep",
        "asleepunspecified": "in_sleep",
        "asleepcore": "in_sleep",
        "asleepdeep": "in_sleep",
        "asleeprem": "in_sleep",
        "core": "in_sleep",
        "deep": "in_sleep",
        "rem": "in_sleep",
        "inawake": "in_awake",
        "awake": "in_awake",
    }

    BOOLEAN_TEXT: ClassVar[dict[str, bool]] = {
        "true": True,
        "1": True,
        "1.0": True,
        "yes": True,
        "correct": True,
        "false": False,
        "0": False,
        "0.0": False,
        "no": False,
        "incorrect": False,
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


    def _pseudonymous_hash(
        self,
        kind: object,
        value: object,
    ) -> str | None:
        """Produce a deterministic study-scoped pseudonymous token.

        Parameters
        ----------
        kind
            Identifier namespace (for example the nearby-device type or
            ``"telephony_contact"``). Tokens from different namespaces never
            collide by construction.
        value
            Raw identifier. Missing values return ``None``.

        Returns
        -------
        str or None
            32-character hex blake2b digest of
            ``study_id|source_platform|kind|value``.

        Notes
        -----
        Raw addresses, phone numbers and device traces are intentionally not
        propagated into the canonical output. The study id acts as the key that
        scopes tokens to one study.
        """

        if value is None or pd.isna(value):
            return None

        raw = (
            f"{self.study_id}|"
            f"{self.source_platform}|"
            f"{kind}|"
            f"{value}"
        )

        return hashlib.blake2b(
            raw.encode("utf-8"),
            digest_size=16,
        ).hexdigest()

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

        return self._pseudonymous_hash(
            device_type,
            address,
        )


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

    @staticmethod
    def _first_present(
        chunk: pd.DataFrame,
        candidates: tuple[str, ...],
    ) -> str | None:
        """Return the first candidate column present in ``chunk``."""

        for column in candidates:
            if column in chunk.columns:
                return column

        return None

    @staticmethod
    def _require_one_of(
        chunk: pd.DataFrame,
        candidates: tuple[str, ...],
        *,
        stream: str,
        field: str,
    ) -> str:
        """Return the first present candidate or raise listing all of them."""

        column = MindLAMPAdapter._first_present(
            chunk,
            candidates,
        )

        if column is None:
            raise ValueError(
                f"{stream}: missing {field} column; expected one of "
                f"{list(candidates)}"
            )

        return column

    @staticmethod
    def _coalesce(
        chunk: pd.DataFrame,
        candidates: tuple[str, ...],
    ) -> pd.Series | None:
        """Row-wise first non-null value across the present candidates.

        Mixed exports put call and SMS fields in different columns of the same
        file (``call_type`` / ``sms_type``), so a single column is not enough.
        Returns ``None`` when no candidate column is present.
        """

        present = [
            c for c in candidates
            if c in chunk.columns
        ]

        if not present:
            return None

        result = chunk[present[0]].astype("object")

        for column in present[1:]:
            result = result.where(
                result.notna(),
                chunk[column].astype("object"),
            )

        return result

    @staticmethod
    def _normalized_text(
        values: pd.Series,
    ) -> pd.Series:
        """Lower-case, strip, and drop a trailing ``.0`` from text codes."""

        return (
            values
            .astype("string")
            .str.strip()
            .str.lower()
            .str.replace(
                r"\.0$",
                "",
                regex=True,
            )
        )

    @staticmethod
    def _snake_case(
        value: object,
    ) -> object:
        """Normalise a task name such as ``"Jewels A"`` to ``jewels_a``."""

        if value is None or pd.isna(value):
            return value

        text = re.sub(
            r"[^0-9a-z]+",
            "_",
            str(value).strip().lower(),
        )

        return text.strip("_")

    @staticmethod
    def _contact_value(
        value: object,
    ) -> object:
        """Normalise a phone number or trace before hashing.

        Values made only of digits and phone punctuation are reduced to their
        digits so that formatting differences do not split one contact into
        several tokens. Opaque traces are hashed as given (stripped).
        """

        if value is None or pd.isna(value):
            return None

        text = str(value).strip()

        if text and re.fullmatch(
            r"[\d\s\-+().]+",
            text,
        ):
            digits = re.sub(
                r"\D",
                "",
                text,
            )

            return digits or None

        return text or None

    def _telephony(
        self,
        chunk: pd.DataFrame,
        uid: str,
        path: Path,
    ) -> pd.DataFrame:
        """Canonicalise call/SMS metadata.

        Accepted columns
        ----------------
        channel
            ``channel`` or ``sensor`` (``lamp.calls`` / ``lamp.sms``). If both
            are absent, rows with ``call_*`` fields are calls and rows with
            ``sms_*`` fields are SMS.
        direction
            ``direction``, ``call_type``, ``sms_type`` or ``type``; text or
            Android integer codes.
        duration
            ``duration_ms`` (milliseconds) or ``call_duration`` (seconds, the
            mindLAMP ``lamp.calls`` convention).
        counterpart
            ``contact``, ``number``, ``address``, ``trace``, ``call_trace`` or
            ``sms_trace``; hashed with the ``telephony_contact`` namespace, so
            the same counterpart has the same token across calls and SMS.
        """

        out = self._shared_fields(
            chunk,
            uid=uid,
            stream="telephony",
            path=path,
        )

        channel_column = self._first_present(
            chunk,
            (
                "channel",
                "sensor",
            ),
        )

        if channel_column is not None:
            channel = (
                self._normalized_text(
                    chunk[channel_column]
                )
                .map(self.TELEPHONY_CHANNEL)
            )

        else:
            call_columns = [
                c for c in chunk.columns
                if str(c).startswith("call_")
            ]

            sms_columns = [
                c for c in chunk.columns
                if str(c).startswith("sms_")
            ]

            is_call = (
                chunk[call_columns].notna().any(axis=1)
                if call_columns
                else pd.Series(False, index=chunk.index)
            )

            is_sms = (
                chunk[sms_columns].notna().any(axis=1)
                if sms_columns
                else pd.Series(False, index=chunk.index)
            )

            channel = pd.Series(
                pd.NA,
                index=chunk.index,
                dtype="object",
            )

            channel[is_sms] = "sms"
            channel[is_call] = "call"

        out["channel"] = (
            pd.Series(channel, index=chunk.index)
            .fillna("unknown")
            .astype(str)
        )

        direction_candidates = (
            "direction",
            "call_type",
            "sms_type",
            "type",
        )

        self._require_one_of(
            chunk,
            direction_candidates,
            stream="telephony",
            field="direction",
        )

        direction = self._coalesce(
            chunk,
            direction_candidates,
        )

        out["direction"] = (
            self._normalized_text(
                direction
            )
            .map(self.TELEPHONY_DIRECTION)
            .fillna("unknown")
            .astype(str)
        )

        if "duration_ms" in chunk:
            out["duration_ms"] = pd.to_numeric(
                chunk["duration_ms"],
                errors="coerce",
            )

        elif "call_duration" in chunk:
            out["duration_ms"] = (
                pd.to_numeric(
                    chunk["call_duration"],
                    errors="coerce",
                )
                * 1000.0
            )

        contact = self._coalesce(
            chunk,
            (
                "contact",
                "number",
                "address",
                "trace",
                "call_trace",
                "sms_trace",
            ),
        )

        if contact is not None:
            out["contact_hash"] = [
                self._pseudonymous_hash(
                    "telephony_contact",
                    self._contact_value(value),
                )
                for value in contact
            ]

        return out

    def _survey(
        self,
        chunk: pd.DataFrame,
        uid: str,
        path: Path,
    ) -> pd.DataFrame:
        """Canonicalise one-row-per-item survey responses.

        Accepted columns
        ----------------
        instrument
            ``instrument``, ``activity_name`` or ``survey``.
        item
            ``item`` or ``question``.
        value
            ``value``: numeric parse into ``value_numeric``; the original
            text is kept in ``value_text``.
        latency
            ``response_latency_ms`` or ``duration`` (milliseconds, mindLAMP
            temporal-slice convention).
        activity
            ``activity_id`` or ``activity``.
        """

        out = self._shared_fields(
            chunk,
            uid=uid,
            stream="survey",
            path=path,
        )

        instrument_column = self._require_one_of(
            chunk,
            (
                "instrument",
                "activity_name",
                "survey",
            ),
            stream="survey",
            field="instrument",
        )

        item_column = self._require_one_of(
            chunk,
            (
                "item",
                "question",
            ),
            stream="survey",
            field="item",
        )

        out["instrument"] = (
            chunk[instrument_column]
            .astype("string")
            .str.strip()
        )

        out["item"] = (
            chunk[item_column]
            .astype("string")
            .str.strip()
        )

        if "value" in chunk:
            out["value_numeric"] = pd.to_numeric(
                chunk["value"],
                errors="coerce",
            )

            out["value_text"] = (
                chunk["value"]
                .astype("string")
            )

        latency_column = self._first_present(
            chunk,
            (
                "response_latency_ms",
                "duration",
            ),
        )

        if latency_column is not None:
            out["response_latency_ms"] = pd.to_numeric(
                chunk[latency_column],
                errors="coerce",
            )

        activity_column = self._first_present(
            chunk,
            (
                "activity_id",
                "activity",
            ),
        )

        if activity_column is not None:
            out["activity_id"] = (
                chunk[activity_column]
                .astype("string")
            )

        return out

    def _cognitive(
        self,
        chunk: pd.DataFrame,
        uid: str,
        path: Path,
    ) -> pd.DataFrame:
        """Canonicalise per-trial cognitive-task responses.

        Accepted columns
        ----------------
        task
            ``task`` or ``activity_name``; normalised to snake_case.
        session
            ``session_id``, ``activity_event_id`` or ``activity_timestamp``
            (the parent activity event; trials of one play share it).
        level
            ``level``.
        response time
            ``response_time_ms`` or ``duration`` (milliseconds).
        correctness
            ``correct`` or ``status``; text or 0/1, parsed to nullable bool.
        """

        out = self._shared_fields(
            chunk,
            uid=uid,
            stream="cognitive",
            path=path,
        )

        task_column = self._require_one_of(
            chunk,
            (
                "task",
                "activity_name",
            ),
            stream="cognitive",
            field="task",
        )

        session_column = self._require_one_of(
            chunk,
            (
                "session_id",
                "activity_event_id",
                "activity_timestamp",
            ),
            stream="cognitive",
            field="session_id",
        )

        out["task"] = (
            chunk[task_column]
            .map(self._snake_case)
            .astype("string")
        )

        out["session_id"] = (
            chunk[session_column]
            .astype("string")
        )

        if "level" in chunk:
            out["level"] = pd.to_numeric(
                chunk["level"],
                errors="coerce",
            )

        rt_column = self._first_present(
            chunk,
            (
                "response_time_ms",
                "duration",
            ),
        )

        if rt_column is not None:
            out["response_time_ms"] = pd.to_numeric(
                chunk[rt_column],
                errors="coerce",
            )

        correct_column = self._first_present(
            chunk,
            (
                "correct",
                "status",
            ),
        )

        if correct_column is not None:
            out["correct"] = (
                self._normalized_text(
                    chunk[correct_column]
                )
                .map(self.BOOLEAN_TEXT)
                .astype("boolean")
            )

        return out

    def _steps(
        self,
        chunk: pd.DataFrame,
        uid: str,
        path: Path,
    ) -> pd.DataFrame:
        """Canonicalise step counts.

        Accepted columns
        ----------------
        count
            ``step_count``, ``steps`` or ``value``.
        interval
            ``interval_duration_ms`` or ``duration`` (milliseconds).
        """

        out = self._shared_fields(
            chunk,
            uid=uid,
            stream="steps",
            path=path,
        )

        count_column = self._require_one_of(
            chunk,
            (
                "step_count",
                "steps",
                "value",
            ),
            stream="steps",
            field="step count",
        )

        out["step_count"] = pd.to_numeric(
            chunk[count_column],
            errors="coerce",
        )

        interval_column = self._first_present(
            chunk,
            (
                "interval_duration_ms",
                "duration",
            ),
        )

        if interval_column is not None:
            out["interval_duration_ms"] = pd.to_numeric(
                chunk[interval_column],
                errors="coerce",
            )

        return out

    def _sleep(
        self,
        chunk: pd.DataFrame,
        uid: str,
        path: Path,
    ) -> pd.DataFrame:
        """Canonicalise wearable/platform sleep segments.

        Accepted columns
        ----------------
        state
            ``representation`` or ``value``; HealthKit sleep-analysis codes
            (0 in bed, 1/3/4/5 asleep, 2 awake) or their names.
        duration
            ``duration_ms`` or ``duration`` (milliseconds).
        """

        out = self._shared_fields(
            chunk,
            uid=uid,
            stream="sleep",
            path=path,
        )

        state_column = self._require_one_of(
            chunk,
            (
                "representation",
                "value",
            ),
            stream="sleep",
            field="representation",
        )

        duration_column = self._require_one_of(
            chunk,
            (
                "duration_ms",
                "duration",
            ),
            stream="sleep",
            field="duration",
        )

        original = (
            chunk[state_column]
            .astype("string")
        )

        out["representation_original"] = original

        key = (
            self._normalized_text(original)
            .str.replace(
                r"[^0-9a-z]",
                "",
                regex=True,
            )
            .str.removeprefix(
                "hkcategoryvaluesleepanalysis"
            )
        )

        out["representation"] = (
            key
            .map(self.SLEEP_REPRESENTATION)
            .fillna("unknown")
            .astype(str)
        )

        out["duration_ms"] = pd.to_numeric(
            chunk[duration_column],
            errors="coerce",
        )

        return out

    def _builder(
        self,
        stream: str,
    ) -> Callable[[pd.DataFrame, str, Path], pd.DataFrame]:
        """Return the canonicalisation method for one stream."""

        builders = {
            "gps": self._gps,
            "accelerometer": self._accelerometer,
            "screen": self._screen,
            "device_usage": self._device_usage,
            "nearby_device": self._nearby_device,
            "telephony": self._telephony,
            "survey": self._survey,
            "cognitive": self._cognitive,
            "steps": self._steps,
            "sleep": self._sleep,
        }

        if stream not in builders:
            raise ValueError(
                f"Unsupported stream: {stream}"
            )

        return builders[stream]


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

        build = self._builder(
            stream
        )

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

                out = build(
                    chunk,
                    uid,
                    path,
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
