"""
Client for the mindLAMP sensor-stream read API.
"""

import pandas as pd

ACCELEROMETER_SPECS = (
    "lamp.accelerometer10hz",
    "lamp.accelerometer20hz",
    "lamp.accelerometer50hz",
    "lamp.accelerometer100hz",
)

LOAD_COLUMNS = ["timestamp", "x", "y", "z"]


class SensorStreamIntegrityError(ValueError):
    """Raised when downloaded frame bytes don't match their sha256."""


class SensorStreamClient:
    def __init__(
        self,
        server_address: str | None = None,
        api_key: str | None = None,
        session=None,
    ):
        raise NotImplementedError

    def list_frames(
        self,
        participant_id: str,
        sensor_spec: str,
        start_ms: int | None,
        end_ms: int | None,
    ) -> list[dict]:
        raise NotImplementedError

    def download_frame(
        self,
        participant_id: str,
        file_id: str,
        *,
        expected_sha256: str | None = None,
    ) -> bytes:
        raise NotImplementedError

    def load(
        self,
        participant_id: str,
        sensor_spec: str,
        start_ms: int | None,
        end_ms: int | None,
        *,
        ascending: bool = True,
    ) -> pd.DataFrame:
        raise NotImplementedError
