import hashlib
import json
from pathlib import Path
from unittest import mock
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import pytest

from cortex2_engine.io.sensor_stream_client import (
    SensorStreamClient,
    SensorStreamIntegrityError,
)
from cortex2_engine.io.telemetry_binary import ACCEL_DTYPE

requests = pytest.importorskip("requests")
zstandard = pytest.importorskip("zstandard")

FIXTURES = Path(__file__).parent / "fixtures" / "sensor_stream"

META = json.loads((FIXTURES / "meta.json").read_text())

PID = "U1234567890"
SPEC = "lamp.accelerometer50hz"
BASE = "https://lamp.example.org"
KEY = "test-api-key"


def records_frame() -> pd.DataFrame:
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


class FakeResponse:
    def __init__(self, status_code=200, *, payload=None, content=b""):
        self.status_code = status_code
        self._payload = payload
        self._content = content
        self.closed = False

    def json(self):
        return self._payload

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self._content), 7):
            yield self._content[i : i + 7]

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)

    def close(self):
        self.closed = True


class FakeServer:
    """Serves a list of (metadata, blob) frames like the sensor-stream API."""

    def __init__(self, frames, *, page_size=1000):
        self.frames = frames
        self.page_size = page_size
        self.calls = []

    def get(self, url, params=None, headers=None, stream=False, timeout=None):
        self.calls.append(
            {
                "url": url,
                "params": dict(params or {}),
                "headers": dict(headers or {}),
                "stream": stream,
            }
        )
        path = urlsplit(url).path
        prefix = f"/participant/{PID}/sensor-stream"
        if path == prefix:
            return self._list(params or {})
        if path.startswith(prefix + "/"):
            file_id = path[len(prefix) + 1 :]
            for meta, blob in self.frames:
                if meta["id"] == file_id:
                    return FakeResponse(content=blob)
        return FakeResponse(404, payload={"error": "404.not-found"})

    def _list(self, params):
        start = int(params.get("cursor", 0))
        page = [m for m, _ in self.frames][start : start + self.page_size]
        end = start + len(page)
        nxt = str(end) if end < len(self.frames) else None
        return FakeResponse(payload={"data": page, "next": nxt})

    def list_calls(self):
        return [c for c in self.calls if c["url"].endswith("/sensor-stream")]

    def download_calls(self):
        return [c for c in self.calls if not c["url"].endswith("/sensor-stream")]


def golden_frame(name="frame.openzl", file_id="f-golden"):
    blob = (FIXTURES / name).read_bytes()
    meta = dict(META[name])
    meta["id"] = file_id
    return meta, blob


def zstd_frame(rows, file_id, *, spec=SPEC):
    raw = np.array(rows, dtype=ACCEL_DTYPE).tobytes()
    blob = zstandard.ZstdCompressor(level=3).compress(raw)
    meta = {
        "id": file_id,
        "sensor_spec": spec,
        "start_timestamp": rows[0][0],
        "end_timestamp": rows[-1][0],
        "sample_count": len(rows),
        "codec": "zstd",
        "codec_format_version": 0,
        "platform": "ios",
        "unit": "m/s^2",
        "byte_length": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
    }
    return meta, blob


def make_client(server):
    return SensorStreamClient(server_address=BASE, api_key=KEY, session=server)


# Configuration.


def test_reads_server_address_and_api_key_from_env(monkeypatch):
    monkeypatch.setenv("LAMP_SERVER_ADDRESS", "lamp.example.org")
    monkeypatch.setenv("LAMP_API_KEY", KEY)
    server = FakeServer([])

    SensorStreamClient(session=server).list_frames(PID, SPEC, 0, 10)

    call = server.calls[0]
    assert call["url"] == f"{BASE}/participant/{PID}/sensor-stream"
    assert call["headers"]["x-api-key"] == KEY


def test_adds_https_when_address_has_no_scheme():
    server = FakeServer([])
    client = SensorStreamClient(
        server_address="lamp.example.org/",
        api_key=KEY,
        session=server,
    )

    client.list_frames(PID, SPEC, 0, 10)

    assert server.calls[0]["url"].startswith("https://lamp.example.org/participant/")


def test_keeps_explicit_http_scheme():
    server = FakeServer([])
    client = SensorStreamClient(
        server_address="http://localhost:3000",
        api_key=KEY,
        session=server,
    )

    client.list_frames(PID, SPEC, 0, 10)

    assert server.calls[0]["url"] == (
        f"http://localhost:3000/participant/{PID}/sensor-stream"
    )


def test_arguments_override_env(monkeypatch):
    monkeypatch.setenv("LAMP_SERVER_ADDRESS", "env.example.org")
    monkeypatch.setenv("LAMP_API_KEY", "env-key")
    server = FakeServer([])

    make_client(server).list_frames(PID, SPEC, 0, 10)

    assert server.calls[0]["url"].startswith(BASE + "/")
    assert server.calls[0]["headers"]["x-api-key"] == KEY


@pytest.mark.parametrize("missing", ["LAMP_SERVER_ADDRESS", "LAMP_API_KEY"])
def test_missing_configuration_raises(monkeypatch, missing):
    monkeypatch.setenv("LAMP_SERVER_ADDRESS", "lamp.example.org")
    monkeypatch.setenv("LAMP_API_KEY", KEY)
    monkeypatch.delenv(missing)

    with pytest.raises(ValueError, match=missing):
        SensorStreamClient(session=FakeServer([]))


def test_never_sends_basic_or_bearer_auth():
    server = FakeServer([golden_frame()])

    make_client(server).load(PID, SPEC, None, None)

    for call in server.calls:
        assert call["headers"] == {"x-api-key": KEY}


# Listing.


def test_list_sends_contract_query_parameters():
    server = FakeServer([])

    make_client(server).list_frames(PID, SPEC, 100, 200)

    assert server.calls[0]["params"] == {
        "sensor_spec": SPEC,
        "from": 100,
        "to": 200,
        "limit": 1000,
    }


def test_list_omits_open_bounds():
    server = FakeServer([])

    make_client(server).list_frames(PID, SPEC, None, None)

    assert server.calls[0]["params"] == {"sensor_spec": SPEC, "limit": 1000}


def test_list_follows_next_cursor_across_pages():
    frames = [
        zstd_frame([(1_790_000_100_000 + 300_000 * i, 0.0, 0.0, 1.0)], f"f{i}")
        for i in range(5)
    ]
    server = FakeServer(frames, page_size=2)

    listed = make_client(server).list_frames(PID, SPEC, None, None)

    assert [m["id"] for m in listed] == ["f0", "f1", "f2", "f3", "f4"]
    calls = server.list_calls()
    assert len(calls) == 3
    assert "cursor" not in calls[0]["params"]
    assert calls[1]["params"]["cursor"] == "2"
    assert calls[2]["params"]["cursor"] == "4"


def test_list_stops_when_server_repeats_cursor():
    session = mock.Mock()
    session.get.return_value = FakeResponse(payload={"data": [], "next": "same"})

    with pytest.raises(ValueError, match="cursor"):
        SensorStreamClient(BASE, KEY, session=session).list_frames(PID, SPEC, 0, 1)


def test_load_keeps_records_with_nan_values():
    t0 = 1_790_000_100_000
    frame = zstd_frame(
        [(t0, float("nan"), 0.0, 0.0), (t0 + 10, 1.0, 0.0, 0.0)],
        "f",
    )
    server = FakeServer([frame, (dict(frame[0], id="g"), frame[1])])

    df = make_client(server).load(PID, SPEC, None, None)

    assert df["timestamp"].tolist() == [t0, t0 + 10]
    assert np.isnan(df["x"].iloc[0])


def test_list_rejects_unknown_sensor_spec():
    with pytest.raises(ValueError, match="sensor_spec"):
        make_client(FakeServer([])).list_frames(PID, "lamp.gps1m", 0, 10)


def test_list_raises_on_http_error():
    session = mock.Mock()
    session.get.return_value = FakeResponse(403, payload={"error": "403"})

    with pytest.raises(requests.HTTPError):
        SensorStreamClient(BASE, KEY, session=session).list_frames(PID, SPEC, 0, 1)


def test_participant_id_is_url_quoted():
    session = mock.Mock()
    session.get.return_value = FakeResponse(payload={"data": [], "next": None})

    SensorStreamClient(BASE, KEY, session=session).list_frames("a/b c", SPEC, 0, 10)

    url = session.get.call_args.args[0]
    assert url == f"{BASE}/participant/a%2Fb%20c/sensor-stream"


# Downloading.


def test_download_uses_requests_get_stream_without_session(monkeypatch):
    meta, blob = golden_frame()
    response = FakeResponse(content=blob)
    get = mock.Mock(return_value=response)
    monkeypatch.setattr(requests, "get", get)
    client = SensorStreamClient(server_address="lamp.example.org", api_key=KEY)

    data = client.download_frame(PID, meta["id"], expected_sha256=meta["sha256"])

    assert data == blob
    args, kwargs = get.call_args
    assert args[0] == f"{BASE}/participant/{PID}/sensor-stream/{meta['id']}"
    assert kwargs["stream"] is True
    assert kwargs["headers"] == {"x-api-key": KEY}
    assert response.closed


def test_download_sha256_mismatch_raises():
    meta, blob = golden_frame()
    meta["sha256"] = "0" * 64
    server = FakeServer([(meta, blob)])

    with pytest.raises(SensorStreamIntegrityError, match="sha256"):
        make_client(server).download_frame(
            PID,
            meta["id"],
            expected_sha256=meta["sha256"],
        )


def test_load_rejects_corrupted_download():
    meta, blob = golden_frame()
    corrupted = blob[:-1] + bytes([blob[-1] ^ 0xFF])
    server = FakeServer([(meta, corrupted)])

    with pytest.raises(SensorStreamIntegrityError):
        make_client(server).load(PID, SPEC, None, None)


@pytest.mark.parametrize("sha256", [None, ""])
def test_load_rejects_listing_entry_without_sha256(sha256):
    meta, blob = golden_frame()
    if sha256 is None:
        del meta["sha256"]
    else:
        meta["sha256"] = sha256
    server = FakeServer([(meta, blob)])

    with pytest.raises(SensorStreamIntegrityError, match="sha256"):
        make_client(server).load(PID, SPEC, None, None)


def test_download_sha256_check_ignores_case():
    meta, blob = golden_frame()
    server = FakeServer([(meta, blob)])

    data = make_client(server).download_frame(
        PID,
        meta["id"],
        expected_sha256=meta["sha256"].upper(),
    )

    assert data == blob


# Loading.


@pytest.mark.parametrize("name", ["frame.openzl", "frame.zstd", "frame_nosize.zstd"])
def test_load_golden_frame_equals_records_json(name):
    server = FakeServer([golden_frame(name)])

    df = make_client(server).load(PID, SPEC, None, None)

    assert list(df.columns) == ["timestamp", "x", "y", "z"]
    assert df["timestamp"].dtype == np.dtype("int64")
    for axis in ("x", "y", "z"):
        assert df[axis].dtype == np.dtype("float32")
    pd.testing.assert_frame_equal(df, records_frame())
    assert all(c["stream"] for c in server.download_calls())


def test_load_merges_frames_in_one_window_and_drops_exact_duplicates():
    t0 = 1_790_000_100_000
    # Two frames for the same window: a restart re-sent t0+20 and t0+40.
    # Within one frame, two identical at-rest records share t0 and are kept.
    first = zstd_frame(
        [
            (t0, 0.0, 0.0, 9.80665),
            (t0, 0.0, 0.0, 9.80665),
            (t0 + 20, 1.0, 0.0, 9.0),
            (t0 + 40, 2.0, 0.0, 9.0),
        ],
        "a",
    )
    second = zstd_frame(
        [
            (t0 + 20, 1.0, 0.0, 9.0),
            (t0 + 40, 2.0, 0.0, 9.0),
            (t0 + 40, 2.5, 0.0, 9.0),
            (t0 + 60, 3.0, 0.0, 9.0),
        ],
        "b",
    )
    server = FakeServer([first, second])

    df = make_client(server).load(PID, SPEC, None, None)

    assert df["timestamp"].tolist() == [t0, t0, t0 + 20, t0 + 40, t0 + 40, t0 + 60]
    assert df["x"].tolist() == [0.0, 0.0, 1.0, 2.0, 2.5, 3.0]
    assert df.index.tolist() == list(range(6))


def test_load_drops_identical_frame_listed_twice():
    meta, blob = golden_frame()
    copy = dict(meta, id="f-copy")
    server = FakeServer([(meta, blob), (copy, blob)])

    df = make_client(server).load(PID, SPEC, None, None)

    pd.testing.assert_frame_equal(df, records_frame())


def test_load_sorts_ascending_by_default_across_out_of_order_frames():
    t0 = 1_790_000_100_000
    later = zstd_frame([(t0 + 300_000, 3.0, 0.0, 0.0)], "later")
    earlier = zstd_frame([(t0, 1.0, 0.0, 0.0), (t0 + 10, 2.0, 0.0, 0.0)], "earlier")
    server = FakeServer([later, earlier])

    df = make_client(server).load(PID, SPEC, None, None)

    assert df["timestamp"].tolist() == [t0, t0 + 10, t0 + 300_000]


def test_load_descending_when_requested():
    t0 = 1_790_000_100_000
    frame = zstd_frame(
        [(t0, 1.0, 0.0, 0.0), (t0 + 10, 2.0, 0.0, 0.0), (t0 + 20, 3.0, 0.0, 0.0)],
        "f",
    )
    server = FakeServer([frame])

    df = make_client(server).load(PID, SPEC, None, None, ascending=False)

    assert df["timestamp"].tolist() == [t0 + 20, t0 + 10, t0]
    assert df.index.tolist() == [0, 1, 2]


def test_load_keeps_order_of_equal_timestamps_stable():
    t0 = 1_790_000_100_000
    frame = zstd_frame(
        [(t0, 1.0, 0.0, 0.0), (t0, 2.0, 0.0, 0.0), (t0 + 10, 3.0, 0.0, 0.0)],
        "f",
    )
    server = FakeServer([frame])

    df = make_client(server).load(PID, SPEC, None, None)

    assert df["x"].tolist() == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("ascending", [True, False])
def test_load_keeps_frame_order_of_many_equal_timestamps(ascending):
    # 200 rows over 7 timestamps: numpy's quicksort is unstable at this
    # size, so only a stable sort keeps x in frame order within each tie.
    t0 = 1_790_000_100_000
    rows = sorted(
        [(t0 + (i % 7), float(i), 0.0, 0.0) for i in range(200)],
        key=lambda r: r[0],
    )
    server = FakeServer([zstd_frame(rows, "f")])

    df = make_client(server).load(PID, SPEC, None, None, ascending=ascending)

    timestamps = sorted({r[0] for r in rows}, reverse=not ascending)
    expected_x = [r[1] for t in timestamps for r in rows if r[0] == t]
    assert df["timestamp"].is_monotonic_increasing == ascending
    assert df["x"].tolist() == expected_x


def test_load_applies_half_open_interval():
    records = records_frame()
    start = int(records["timestamp"].iloc[10])
    end = int(records["timestamp"].iloc[20])
    server = FakeServer([golden_frame()])

    df = make_client(server).load(PID, SPEC, start, end)

    assert df["timestamp"].iloc[0] == start
    assert df["timestamp"].iloc[-1] < end
    assert len(df) == 10
    assert server.list_calls()[0]["params"]["from"] == start
    assert server.list_calls()[0]["params"]["to"] == end


def test_load_end_equal_to_start_returns_empty():
    start = int(records_frame()["timestamp"].iloc[10])
    server = FakeServer([golden_frame()])

    df = make_client(server).load(PID, SPEC, start, start)

    assert df.empty


def test_load_empty_result_has_columns_and_dtypes():
    server = FakeServer([])

    df = make_client(server).load(PID, SPEC, 0, 10)

    assert list(df.columns) == ["timestamp", "x", "y", "z"]
    assert len(df) == 0
    assert df["timestamp"].dtype == np.dtype("int64")
    for axis in ("x", "y", "z"):
        assert df[axis].dtype == np.dtype("float32")
    assert server.download_calls() == []


def test_load_passes_metadata_to_decoder_checks():
    meta, blob = golden_frame()
    meta["end_timestamp"] += 1
    server = FakeServer([(meta, blob)])

    with pytest.raises(ValueError, match="end_timestamp"):
        make_client(server).load(PID, SPEC, None, None)
