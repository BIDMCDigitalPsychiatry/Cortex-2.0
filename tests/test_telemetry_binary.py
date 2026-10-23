import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cortex2_engine.io.telemetry_binary import (
    ACCEL_DTYPE,
    TelemetryDecodeError,
    decode_accel_frame,
    decompress_frame,
)

FIXTURES = Path(__file__).parent / "fixtures" / "sensor_stream"

META = json.loads((FIXTURES / "meta.json").read_text())

FRAME_BIN = (FIXTURES / "frame.bin").read_bytes()


def expected_records() -> pd.DataFrame:
    records = json.loads((FIXTURES / "records.json").read_text())
    return pd.DataFrame(
        {
            "timestamp": np.array(
                [r["timestamp"] for r in records],
                dtype="int64",
            ),
            "x": np.array([r["x"] for r in records], dtype="float32"),
            "y": np.array([r["y"] for r in records], dtype="float32"),
            "z": np.array([r["z"] for r in records], dtype="float32"),
        }
    )


def golden_kwargs(name: str) -> dict:
    meta = META[name]
    return {
        "codec": meta["codec"],
        "sample_count": meta["sample_count"],
        "codec_format_version": meta["codec_format_version"],
        "start_timestamp": meta["start_timestamp"],
        "end_timestamp": meta["end_timestamp"],
    }


def openzl_compress(
    buffers: list[bytes],
    format_version: int = 24,
) -> bytes:
    """Compresses buffers with the generic OpenZL graph (one output each)."""
    zl = pytest.importorskip("openzl.ext")
    compressor = zl.Compressor()
    graph = zl.graphs.Compress()(compressor)
    compressor.select_starting_graph(graph)
    cctx = zl.CCtx()
    cctx.ref_compressor(compressor)
    cctx.set_parameter(zl.CParam.FormatVersion, format_version)
    return cctx.compress([zl.Input(zl.Type.Serial, b) for b in buffers])


def zstd_compress(raw: bytes, *, content_size: bool = True) -> bytes:
    zstandard = pytest.importorskip("zstandard")
    return zstandard.ZstdCompressor(
        level=3,
        write_content_size=content_size,
    ).compress(raw)


def pack(rows: list[tuple[int, float, float, float]]) -> bytes:
    return np.array(rows, dtype=ACCEL_DTYPE).tobytes()


def test_accel_dtype_is_20_byte_little_endian_record():
    assert ACCEL_DTYPE.itemsize == 20
    assert ACCEL_DTYPE.names == ("timestamp", "x", "y", "z")
    assert ACCEL_DTYPE["timestamp"].str == "<i8"
    for axis in ("x", "y", "z"):
        assert ACCEL_DTYPE[axis].str == "<f4"


@pytest.mark.parametrize(
    "name",
    ["frame.openzl", "frame.zstd", "frame_nosize.zstd"],
)
def test_decode_golden_frame_equals_records_json(name):
    blob = (FIXTURES / name).read_bytes()

    df = decode_accel_frame(blob, **golden_kwargs(name))

    assert list(df.columns) == ["timestamp", "x", "y", "z"]
    assert df["timestamp"].dtype == np.dtype("int64")
    for axis in ("x", "y", "z"):
        assert df[axis].dtype == np.dtype("float32")
    pd.testing.assert_frame_equal(df, expected_records())


def test_decode_single_record_golden_frame():
    blob = (FIXTURES / "frame_single.openzl").read_bytes()
    raw = (FIXTURES / "frame_single.bin").read_bytes()

    df = decode_accel_frame(blob, **golden_kwargs("frame_single.openzl"))

    expected = np.frombuffer(raw, dtype=ACCEL_DTYPE)
    assert len(df) == 1
    assert int(df["timestamp"].iloc[0]) == int(expected["timestamp"][0])
    assert df["z"].iloc[0] == expected["z"][0]


@pytest.mark.parametrize(
    "name",
    ["frame.openzl", "frame.zstd", "frame_nosize.zstd"],
)
def test_decompress_frame_returns_packed_records(name):
    meta = META[name]
    blob = (FIXTURES / name).read_bytes()

    raw = decompress_frame(
        blob,
        meta["codec"],
        sample_count=meta["sample_count"],
        codec_format_version=meta["codec_format_version"],
    )

    assert raw == FRAME_BIN


def test_timestamp_checks_are_optional():
    blob = (FIXTURES / "frame.openzl").read_bytes()

    df = decode_accel_frame(
        blob,
        codec="openzl",
        sample_count=500,
        codec_format_version=24,
    )

    assert len(df) == 500


# Decoder check 1: exactly one output (OpenZL only).


def test_check1_rejects_openzl_frame_with_two_outputs():
    blob = openzl_compress([FRAME_BIN, FRAME_BIN])

    with pytest.raises(TelemetryDecodeError, match="one output"):
        decompress_frame(
            blob,
            "openzl",
            sample_count=500,
            codec_format_version=24,
        )


# Decoder check 2: length is a multiple of 20.


@pytest.mark.parametrize("codec", ["openzl", "zstd"])
def test_check2_rejects_length_not_multiple_of_20(codec):
    raw = FRAME_BIN[:30]
    blob = openzl_compress([raw]) if codec == "openzl" else zstd_compress(raw)

    with pytest.raises(TelemetryDecodeError, match="multiple of 20"):
        decompress_frame(
            blob,
            codec,
            sample_count=1,
            codec_format_version=24 if codec == "openzl" else 0,
        )


# Decoder check 3: length / 20 equals sample_count.


@pytest.mark.parametrize(
    "name, sample_count",
    [
        ("frame.openzl", 499),
        ("frame.openzl", 501),
        ("frame.zstd", 499),
        ("frame.zstd", 501),
        ("frame_nosize.zstd", 501),
    ],
)
def test_check3_rejects_sample_count_mismatch(name, sample_count):
    meta = META[name]
    blob = (FIXTURES / name).read_bytes()

    with pytest.raises(TelemetryDecodeError, match="sample_count"):
        decompress_frame(
            blob,
            meta["codec"],
            sample_count=sample_count,
            codec_format_version=meta["codec_format_version"],
        )


def test_check3_rejects_nosize_zstd_frame_larger_than_sample_count():
    # Without a content size, the output bound is sample_count * 20, so the
    # decoder fails before it can count records. It must still reject.
    blob = (FIXTURES / "frame_nosize.zstd").read_bytes()

    with pytest.raises(TelemetryDecodeError):
        decompress_frame(
            blob,
            "zstd",
            sample_count=499,
            codec_format_version=0,
        )


# Decoder check 4: format version matches (OpenZL only).


def test_check4_rejects_format_version_mismatch_in_frame():
    blob = openzl_compress([FRAME_BIN], format_version=25)

    with pytest.raises(TelemetryDecodeError, match="format version"):
        decompress_frame(
            blob,
            "openzl",
            sample_count=500,
            codec_format_version=24,
        )


def test_check4_rejects_format_version_mismatch_in_metadata():
    blob = (FIXTURES / "frame.openzl").read_bytes()

    with pytest.raises(TelemetryDecodeError, match="format version"):
        decompress_frame(
            blob,
            "openzl",
            sample_count=500,
            codec_format_version=27,
        )


# Decoder check 5: first and last timestamps match the metadata.


def test_check5_rejects_start_timestamp_mismatch():
    kwargs = golden_kwargs("frame.openzl")
    kwargs["start_timestamp"] += 1

    with pytest.raises(TelemetryDecodeError, match="start_timestamp"):
        decode_accel_frame((FIXTURES / "frame.openzl").read_bytes(), **kwargs)


def test_check5_rejects_end_timestamp_mismatch():
    kwargs = golden_kwargs("frame.zstd")
    kwargs["end_timestamp"] -= 20

    with pytest.raises(TelemetryDecodeError, match="end_timestamp"):
        decode_accel_frame((FIXTURES / "frame.zstd").read_bytes(), **kwargs)


# The checks run in contract order: the first failure wins.


def test_check1_runs_before_check4():
    blob = openzl_compress([FRAME_BIN, FRAME_BIN], format_version=25)

    with pytest.raises(TelemetryDecodeError, match="one output"):
        decompress_frame(
            blob,
            "openzl",
            sample_count=500,
            codec_format_version=24,
        )


def test_check2_runs_before_check3_and_check4():
    blob = openzl_compress([FRAME_BIN[:30]], format_version=25)

    with pytest.raises(TelemetryDecodeError, match="multiple of 20"):
        decompress_frame(
            blob,
            "openzl",
            sample_count=7,
            codec_format_version=24,
        )


def test_check3_runs_before_check4():
    blob = openzl_compress([FRAME_BIN], format_version=25)

    with pytest.raises(TelemetryDecodeError, match="sample_count"):
        decompress_frame(
            blob,
            "openzl",
            sample_count=499,
            codec_format_version=24,
        )


def test_check4_runs_before_check5():
    blob = openzl_compress([FRAME_BIN], format_version=25)

    with pytest.raises(TelemetryDecodeError, match="format version"):
        decode_accel_frame(
            blob,
            codec="openzl",
            sample_count=500,
            codec_format_version=24,
            start_timestamp=0,
            end_timestamp=0,
        )


# Corrupt input and bad arguments.


@pytest.mark.parametrize("name", ["frame.openzl", "frame.zstd"])
def test_truncated_frame_raises(name):
    meta = META[name]
    blob = (FIXTURES / name).read_bytes()

    with pytest.raises(TelemetryDecodeError):
        decompress_frame(
            blob[: len(blob) // 2],
            meta["codec"],
            sample_count=meta["sample_count"],
            codec_format_version=meta["codec_format_version"],
        )


def test_unknown_codec_raises_value_error():
    with pytest.raises(ValueError, match="codec"):
        decompress_frame(
            FRAME_BIN,
            "gzip",
            sample_count=500,
            codec_format_version=0,
        )


@pytest.mark.parametrize("sample_count", [0, -1])
def test_non_positive_sample_count_raises_value_error(sample_count):
    with pytest.raises(ValueError, match="sample_count"):
        decompress_frame(
            (FIXTURES / "frame.zstd").read_bytes(),
            "zstd",
            sample_count=sample_count,
            codec_format_version=0,
        )


def test_decoded_records_are_writable_copies():
    df = decode_accel_frame(
        (FIXTURES / "frame.zstd").read_bytes(),
        **golden_kwargs("frame.zstd"),
    )

    df.loc[0, "x"] = 1.5

    assert df.loc[0, "x"] == np.float32(1.5)


def test_decode_custom_zstd_frame_round_trip():
    rows = [
        (1_790_000_100_000, 0.0, 0.0, 9.80665),
        (1_790_000_100_000, 0.5, -0.5, 9.7),
        (1_790_000_100_010, 1.0, 2.0, 3.0),
    ]

    df = decode_accel_frame(
        zstd_compress(pack(rows)),
        codec="zstd",
        sample_count=3,
        codec_format_version=0,
        start_timestamp=1_790_000_100_000,
        end_timestamp=1_790_000_100_010,
    )

    assert df["timestamp"].tolist() == [r[0] for r in rows]
    assert df["y"].tolist() == [np.float32(r[2]) for r in rows]


# Optional dependencies are imported lazily.


def test_missing_openzl_raises_import_error_naming_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "openzl", None)
    monkeypatch.setitem(sys.modules, "openzl.ext", None)

    with pytest.raises(ImportError, match=r"cortex2-engine\[telemetry\]"):
        decompress_frame(
            (FIXTURES / "frame.openzl").read_bytes(),
            "openzl",
            sample_count=500,
            codec_format_version=24,
        )


def test_missing_zstandard_raises_import_error_naming_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "zstandard", None)

    with pytest.raises(ImportError, match=r"cortex2-engine\[telemetry\]"):
        decompress_frame(
            (FIXTURES / "frame.zstd").read_bytes(),
            "zstd",
            sample_count=500,
            codec_format_version=0,
        )
