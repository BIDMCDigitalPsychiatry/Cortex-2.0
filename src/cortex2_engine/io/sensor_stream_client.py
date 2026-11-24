"""
Client for the mindLAMP sensor-stream read API.

The API is defined by the mindLAMP sensor-stream contract
(LAMP-protocol ``docs/sensor-stream.md``):

    GET /participant/{id}/sensor-stream            frame metadata, paged
    GET /participant/{id}/sensor-stream/{file_id}  compressed frame bytes

Both routes authenticate with an ``x-api-key`` header. Install the
optional dependencies with:

    pip install "cortex2-engine[telemetry]"
"""

import hashlib
import os
from urllib.parse import quote

import numpy as np
import pandas as pd

from cortex2_engine.io.telemetry_binary import decode_accel_frame

ACCELEROMETER_SPECS = (
    "lamp.accelerometer10hz",
    "lamp.accelerometer20hz",
    "lamp.accelerometer50hz",
    "lamp.accelerometer100hz",
)

LOAD_COLUMNS = ["timestamp", "x", "y", "z"]

LIST_PAGE_LIMIT = 1000

SERVER_ADDRESS_ENV = "LAMP_SERVER_ADDRESS"
API_KEY_ENV = "LAMP_API_KEY"

_RECORD_COLUMNS = ["timestamp", "x", "y", "z"]


class SensorStreamIntegrityError(ValueError):
    """Raised when downloaded frame bytes don't match their sha256."""


def _normalize_address(
    address: str,
) -> str:
    address = address.strip().rstrip("/")

    if "://" not in address:
        address = "https://" + address

    return address


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.Series(dtype="int64"),
            "x": pd.Series(dtype="float32"),
            "y": pd.Series(dtype="float32"),
            "z": pd.Series(dtype="float32"),
        }
    )


class SensorStreamClient:
    """
    Read binary accelerometer frames from a LAMP server.

    Args:
        server_address: The server base URL. Defaults to the
            ``LAMP_SERVER_ADDRESS`` environment variable. When it has
            no scheme, ``https://`` is added.
        api_key: The API key sent as ``x-api-key``. Defaults to the
            ``LAMP_API_KEY`` environment variable.
        session: An object with a ``requests``-compatible ``get``
            method, such as ``requests.Session``. Defaults to the
            ``requests`` module.
        timeout: The per-request timeout in seconds.

    Raises:
        ValueError: The server address or API key is missing.
        ImportError: ``session`` is omitted and ``requests`` isn't
            installed.
    """

    def __init__(
        self,
        server_address: str | None = None,
        api_key: str | None = None,
        session=None,
        *,
        timeout: float = 60.0,
    ):
        server_address = server_address or os.environ.get(
            SERVER_ADDRESS_ENV
        )
        api_key = api_key or os.environ.get(API_KEY_ENV)

        if not server_address:
            raise ValueError(
                f"No server address: pass server_address or set "
                f"{SERVER_ADDRESS_ENV}"
            )

        if not api_key:
            raise ValueError(
                f"No API key: pass api_key or set {API_KEY_ENV}"
            )

        if session is None:
            try:
                import requests
            except ImportError as exc:
                raise ImportError(
                    "SensorStreamClient needs the 'requests' package. "
                    'Install it with: pip install "cortex2-engine[telemetry]"'
                ) from exc

            session = requests

        self.base_url = _normalize_address(server_address)
        self.timeout = timeout
        self._api_key = api_key
        self._http = session

    def _headers(self) -> dict:
        return {"x-api-key": self._api_key}

    def _stream_url(
        self,
        participant_id: str,
    ) -> str:
        return (
            f"{self.base_url}/participant/"
            f"{quote(str(participant_id), safe='')}/sensor-stream"
        )

    def list_frames(
        self,
        participant_id: str,
        sensor_spec: str,
        start_ms: int | None,
        end_ms: int | None,
    ) -> list[dict]:
        """
        List metadata for frames that overlap ``[start_ms, end_ms)``.

        Follows the ``next`` cursor until every page is read. Pass
        ``None`` for an open bound.

        Raises:
            ValueError: ``sensor_spec`` isn't an accelerometer spec,
                or the server repeats a cursor.
            requests.HTTPError: The server returns an error status.
        """
        if sensor_spec not in ACCELEROMETER_SPECS:
            raise ValueError(
                f"Unsupported sensor_spec {sensor_spec!r}; expected "
                f"one of {ACCELEROMETER_SPECS}"
            )

        params = {
            "sensor_spec": sensor_spec,
            "limit": LIST_PAGE_LIMIT,
        }

        if start_ms is not None:
            params["from"] = int(start_ms)

        if end_ms is not None:
            params["to"] = int(end_ms)

        url = self._stream_url(participant_id)
        frames: list[dict] = []
        seen_cursors = set()

        while True:
            response = self._http.get(
                url,
                params=params,
                headers=self._headers(),
                timeout=self.timeout,
            )

            try:
                response.raise_for_status()
                body = response.json()
            finally:
                response.close()

            frames.extend(body.get("data") or [])
            cursor = body.get("next")

            if not cursor:
                return frames

            if cursor in seen_cursors:
                raise ValueError(
                    f"Server repeated paging cursor {cursor!r}"
                )

            seen_cursors.add(cursor)
            params = {**params, "cursor": cursor}

    def download_frame(
        self,
        participant_id: str,
        file_id: str,
        *,
        expected_sha256: str | None = None,
    ) -> bytes:
        """
        Download one frame's compressed bytes.

        Raises:
            SensorStreamIntegrityError: The bytes don't match
                ``expected_sha256``.
            requests.HTTPError: The server returns an error status.
        """
        url = (
            f"{self._stream_url(participant_id)}/"
            f"{quote(str(file_id), safe='')}"
        )

        response = self._http.get(
            url,
            headers=self._headers(),
            stream=True,
            timeout=self.timeout,
        )

        digest = hashlib.sha256()
        chunks = []

        try:
            response.raise_for_status()

            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    digest.update(chunk)
                    chunks.append(chunk)
        finally:
            response.close()

        if (
            expected_sha256 is not None
            and digest.hexdigest() != expected_sha256.lower()
        ):
            raise SensorStreamIntegrityError(
                f"Frame {file_id} sha256 {digest.hexdigest()} != "
                f"expected {expected_sha256}"
            )

        return b"".join(chunks)

    def load(
        self,
        participant_id: str,
        sensor_spec: str,
        start_ms: int | None,
        end_ms: int | None,
        *,
        ascending: bool = True,
    ) -> pd.DataFrame:
        """
        Load accelerometer records in ``[start_ms, end_ms)``.

        Downloads, verifies, and decodes every overlapping frame. A
        window can have several frames (restarts, late batches), so
        records are merged and records repeated across frames are
        dropped. Identical records within one frame are kept, because
        two samples can share a millisecond and a value.

        Args:
            participant_id: The LAMP participant ID.
            sensor_spec: One of the four ``lamp.accelerometer*hz``
                specs.
            start_ms: Inclusive start, epoch ms, or ``None``.
            end_ms: Exclusive end, epoch ms, or ``None``.
            ascending: Sort by ``timestamp`` ascending (default) or
                descending. Equal timestamps keep their frame order.

        Returns:
            A DataFrame with exactly the columns ``timestamp``
            (``int64``, epoch ms) and ``x``, ``y``, ``z``
            (``float32``, m/s²).

        Raises:
            SensorStreamIntegrityError: A listing entry has no
                ``sha256``, or a frame's bytes don't match it.
            TelemetryDecodeError: A frame fails a decoder check.
        """
        metas = self.list_frames(
            participant_id,
            sensor_spec,
            start_ms,
            end_ms,
        )

        # Sort frames first so equal timestamps keep a stable order.
        metas = sorted(
            metas,
            key=lambda m: (
                int(m["start_timestamp"]),
                str(m.get("id", "")),
            ),
        )

        parts = []

        for frame_index, meta in enumerate(metas):
            # Fail closed: never decode bytes you can't verify.
            expected_sha256 = meta.get("sha256")

            if not expected_sha256:
                raise SensorStreamIntegrityError(
                    f"Frame {meta.get('id')} has no sha256 in its "
                    f"listing entry, so its bytes can't be verified"
                )

            blob = self.download_frame(
                participant_id,
                meta["id"],
                expected_sha256=expected_sha256,
            )

            part = decode_accel_frame(
                blob,
                codec=meta["codec"],
                sample_count=int(meta["sample_count"]),
                codec_format_version=int(meta["codec_format_version"]),
                start_timestamp=int(meta["start_timestamp"]),
                end_timestamp=int(meta["end_timestamp"]),
            )

            # The occurrence number separates real repeats inside a
            # frame from copies of the same record in another frame.
            part["_occurrence"] = part.groupby(
                _RECORD_COLUMNS,
                sort=False,
                dropna=False,
            ).cumcount()
            parts.append(part)

        if not parts:
            return _empty_frame()

        df = pd.concat(parts, ignore_index=True)

        df = df.drop_duplicates(
            subset=_RECORD_COLUMNS + ["_occurrence"],
            keep="first",
        )

        mask = np.ones(len(df), dtype=bool)

        if start_ms is not None:
            mask &= (df["timestamp"] >= int(start_ms)).to_numpy()

        if end_ms is not None:
            mask &= (df["timestamp"] < int(end_ms)).to_numpy()

        df = df.loc[mask, LOAD_COLUMNS]

        df = df.sort_values(
            "timestamp",
            ascending=ascending,
            kind="mergesort",
        )

        if df.empty:
            return _empty_frame()

        return df.reset_index(drop=True)
