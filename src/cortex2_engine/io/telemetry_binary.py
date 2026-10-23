"""
Decoder for binary accelerometer telemetry frames.

The wire format is defined by the mindLAMP sensor-stream contract
(LAMP-protocol ``docs/sensor-stream.md``). A frame holds 20-byte
little-endian records compressed with OpenZL or zstd.
"""

import numpy as np
import pandas as pd

ACCEL_DTYPE = np.dtype(
    [
        ("timestamp", "<i8"),
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
    ]
)

ACCEL_RECORD_SIZE = 20

assert ACCEL_DTYPE.itemsize == ACCEL_RECORD_SIZE

OPENZL_FORMAT_VERSION = 24
OPENZL_MAGIC_BASE = 0xD7B1A5C0

CODECS = ("openzl", "zstd")


class TelemetryDecodeError(ValueError):
    """Raised when a frame fails a contract decoder check."""


def decompress_frame(
    blob: bytes,
    codec: str,
    *,
    sample_count: int,
    codec_format_version: int,
) -> bytes:
    raise NotImplementedError


def decode_accel_frame(
    blob: bytes,
    *,
    codec: str,
    sample_count: int,
    codec_format_version: int,
    start_timestamp: int | None = None,
    end_timestamp: int | None = None,
) -> pd.DataFrame:
    raise NotImplementedError
