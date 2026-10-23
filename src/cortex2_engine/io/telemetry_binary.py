"""
Decoder for binary accelerometer telemetry frames.

The wire format is defined by the mindLAMP sensor-stream contract
(LAMP-protocol ``docs/sensor-stream.md``). A frame holds 20-byte
little-endian records compressed with OpenZL or zstd.

The codecs are optional dependencies. Install them with:

    pip install "cortex2-engine[telemetry]"
"""

import struct

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

_EXTRA_HINT = 'Install it with: pip install "cortex2-engine[telemetry]"'


class TelemetryDecodeError(ValueError):
    """Raised when a frame fails a contract decoder check."""


def _import_openzl():
    try:
        import openzl.ext as zl
    except ImportError as exc:
        raise ImportError(
            "Decoding openzl frames needs the 'openzl' package. "
            + _EXTRA_HINT
        ) from exc

    return zl


def _import_zstandard():
    try:
        import zstandard
    except ImportError as exc:
        raise ImportError(
            "Decoding zstd frames needs the 'zstandard' package. "
            + _EXTRA_HINT
        ) from exc

    return zstandard


def openzl_format_version(
    blob: bytes,
) -> int:
    """
    Return the OpenZL format version encoded in a frame's magic number.

    The Python binding doesn't expose it, so this reads the first
    4 bytes as a little-endian ``uint32``.
    """
    if len(blob) < 4:
        raise TelemetryDecodeError(
            "OpenZL frame is shorter than its 4-byte magic number"
        )

    return struct.unpack("<I", blob[:4])[0] - OPENZL_MAGIC_BASE


def _check_record_length(
    length: int,
    sample_count: int,
    *,
    label: str = "Decompressed length",
) -> None:
    # Check 2: whole records only.
    if length % ACCEL_RECORD_SIZE != 0:
        raise TelemetryDecodeError(
            f"{label} {length} is not a multiple of "
            f"{ACCEL_RECORD_SIZE}"
        )

    # Check 3: record count matches the metadata.
    decoded_count = length // ACCEL_RECORD_SIZE

    if decoded_count != sample_count:
        raise TelemetryDecodeError(
            f"Frame holds {decoded_count} records but "
            f"sample_count is {sample_count}"
        )


def _decompress_openzl(
    blob: bytes,
    sample_count: int,
) -> bytes:
    zl = _import_openzl()

    # Apply checks 1 to 3 to the frame header first, so a small frame
    # that declares a huge output is rejected before any allocation.
    try:
        info = zl.FrameInfo(blob)
        declared_outputs = info.num_outputs
    except Exception as exc:
        raise TelemetryDecodeError(
            f"OpenZL frame header is invalid: {exc}"
        ) from exc

    if declared_outputs != 1:
        raise TelemetryDecodeError(
            f"OpenZL frame must have exactly one output, "
            f"declares {declared_outputs}"
        )

    try:
        declared_length = info.output_content_size(0)
    except Exception as exc:
        raise TelemetryDecodeError(
            f"OpenZL frame header has no output size: {exc}"
        ) from exc

    _check_record_length(
        declared_length,
        sample_count,
        label="Declared decompressed length",
    )

    try:
        outputs = zl.DCtx().decompress(blob)
    except Exception as exc:
        raise TelemetryDecodeError(
            f"OpenZL decompression failed: {exc}"
        ) from exc

    # Check 1: exactly one output.
    if len(outputs) != 1:
        raise TelemetryDecodeError(
            f"OpenZL frame must have exactly one output, "
            f"got {len(outputs)}"
        )

    return outputs[0].content.as_bytes()


def _decompress_zstd(
    blob: bytes,
    sample_count: int,
) -> bytes:
    zstandard = _import_zstandard()
    bound = sample_count * ACCEL_RECORD_SIZE

    try:
        declared_length = zstandard.get_frame_parameters(blob).content_size
    except zstandard.ZstdError as exc:
        raise TelemetryDecodeError(
            f"zstd frame header is invalid: {exc}"
        ) from exc

    # python-zstandard allocates a declared content size whatever
    # max_output_size says, so check it before decompressing.
    if declared_length not in (
        zstandard.CONTENTSIZE_UNKNOWN,
        zstandard.CONTENTSIZE_ERROR,
    ):
        _check_record_length(
            declared_length,
            sample_count,
            label="Declared decompressed length",
        )

    try:
        # The bound lets frames without a content size decode too.
        return zstandard.ZstdDecompressor().decompress(
            blob,
            max_output_size=bound,
        )
    except zstandard.ZstdError as exc:
        raise TelemetryDecodeError(
            f"zstd decompression failed within the "
            f"sample_count * 20 = {bound} byte bound: {exc}"
        ) from exc


def decompress_frame(
    blob: bytes,
    codec: str,
    *,
    sample_count: int,
    codec_format_version: int,
) -> bytes:
    """
    Decompress one frame and apply contract decoder checks 1 to 4.

    Checks 1 to 3 run first on the size the frame header declares,
    so a frame that would decompress past ``sample_count * 20``
    bytes is rejected before its output is allocated.

    Args:
        blob: The compressed frame bytes.
        codec: ``"openzl"`` or ``"zstd"``.
        sample_count: The frame's ``sample_count`` metadata.
        codec_format_version: The frame's ``codec_format_version``
            metadata. Only OpenZL frames are checked against it.

    Returns:
        The packed records, ``sample_count * 20`` bytes.

    Raises:
        TelemetryDecodeError: The frame fails a decoder check.
        ValueError: ``codec`` is unknown or ``sample_count`` is
            less than 1.
        ImportError: The codec's package isn't installed.
    """
    if codec not in CODECS:
        raise ValueError(
            f"Unknown codec {codec!r}; expected one of {CODECS}"
        )

    if sample_count < 1:
        raise ValueError(
            f"sample_count must be at least 1, got {sample_count}"
        )

    if codec == "openzl":
        raw = _decompress_openzl(blob, sample_count)
    else:
        raw = _decompress_zstd(blob, sample_count)

    # Checks 2 and 3 again, on the bytes actually decoded.
    _check_record_length(len(raw), sample_count)

    # Check 4: format version matches the metadata.
    if codec == "openzl":
        frame_version = openzl_format_version(blob)

        if frame_version != codec_format_version:
            raise TelemetryDecodeError(
                f"OpenZL frame format version {frame_version} != "
                f"codec_format_version {codec_format_version}"
            )

    return raw


def decode_accel_frame(
    blob: bytes,
    *,
    codec: str,
    sample_count: int,
    codec_format_version: int,
    start_timestamp: int | None = None,
    end_timestamp: int | None = None,
) -> pd.DataFrame:
    """
    Decode one accelerometer frame into a DataFrame.

    Applies all five contract decoder checks, in order. Check 5
    runs only for the timestamps you pass.

    Returns:
        A DataFrame with columns ``timestamp`` (``int64``, epoch
        ms) and ``x``, ``y``, ``z`` (``float32``, m/s²), in frame
        order.

    Raises:
        TelemetryDecodeError: The frame fails a decoder check.
    """
    raw = decompress_frame(
        blob,
        codec,
        sample_count=sample_count,
        codec_format_version=codec_format_version,
    )

    records = np.frombuffer(raw, dtype=ACCEL_DTYPE)

    # Check 5: first and last timestamps match the metadata.
    first = int(records["timestamp"][0])
    last = int(records["timestamp"][-1])

    if start_timestamp is not None and first != int(start_timestamp):
        raise TelemetryDecodeError(
            f"First record timestamp {first} != "
            f"start_timestamp {start_timestamp}"
        )

    if end_timestamp is not None and last != int(end_timestamp):
        raise TelemetryDecodeError(
            f"Last record timestamp {last} != "
            f"end_timestamp {end_timestamp}"
        )

    # astype copies into native-endian, writable columns.
    return pd.DataFrame(
        {
            "timestamp": records["timestamp"].astype(np.int64),
            "x": records["x"].astype(np.float32),
            "y": records["y"].astype(np.float32),
            "z": records["z"].astype(np.float32),
        }
    )
