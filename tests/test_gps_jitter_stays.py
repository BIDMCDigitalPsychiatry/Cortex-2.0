import numpy as np
import pandas as pd
import pytest
from gps_synthetic import BASE_LAT, BASE_LON, MIN_MS, offset, track

from cortex2_engine.features.gps_geometry import (
    edge_midpoints,
    haversine_m,
    project_local_m,
)
from cortex2_engine.features.gps_jitter import DROP_REASONS, filter_gps_jitter
from cortex2_engine.features.gps_stays import build_stays, dwell_threshold_ms


def test_haversine_and_projection_agree_locally():
    lat, lon = offset(3000.0, 4000.0)
    assert haversine_m(BASE_LAT, BASE_LON, lat, lon) == pytest.approx(5000.0, rel=1e-3)

    east, north = project_local_m(lat, lon, lat0_deg=BASE_LAT, lon0_deg=BASE_LON)
    assert float(east) == pytest.approx(3000.0, rel=1e-6)
    assert float(north) == pytest.approx(4000.0, rel=1e-6)


def test_projection_uses_cosine_of_radians_not_degrees():
    # At 60 degrees N one degree of longitude is half a degree of latitude.
    east, _ = project_local_m(60.0, 1.0, lat0_deg=60.0, lon0_deg=0.0)
    _, north = project_local_m(61.0, 0.0, lat0_deg=60.0, lon0_deg=0.0)
    assert float(east) / float(north) == pytest.approx(0.5, rel=1e-9)


def test_edge_midpoints_requires_columns():
    with pytest.raises(ValueError, match="Missing required GPS edge columns"):
        edge_midpoints(pd.DataFrame({"supported": [True]}))


def test_jitter_counts_each_reason_once():
    gps = track(0, 10 * MIN_MS, lambda t: (0.0, 0.0))
    gps.loc[1, "latitude"] = np.nan
    gps.loc[2, ["latitude", "longitude"]] = 0.0
    gps.loc[3, "latitude"] = 95.0
    gps.loc[4, "accuracy_m"] = 500.0
    spike_lat, spike_lon = offset(20_000.0, 0.0)  # 20 km in one minute
    gps.loc[6, ["latitude", "longitude"]] = [spike_lat, spike_lon]

    filtered, counts = filter_gps_jitter(gps)

    assert counts["missing_coordinates"] == 1
    assert counts["invalid_coordinates"] == 2
    assert counts["accuracy_exceeds_max"] == 1
    assert counts["speed_spike"] == 1
    assert counts["n_input"] == 10
    assert counts["n_kept"] == 5 == len(filtered)
    assert sum(counts[r] for r in DROP_REASONS) + counts["n_kept"] == 10
    assert 6 not in filtered.index


def test_jitter_keeps_sustained_fast_travel():
    # A train at 200 km/h is fast but not a spike.
    gps = track(0, 10 * MIN_MS, lambda t: (t / 3_600_000 * 200_000.0, 0.0))
    filtered, counts = filter_gps_jitter(gps, max_speed_kmh=150.0)
    assert counts["speed_spike"] == 0
    assert len(filtered) == 10


def test_jitter_accepts_legacy_accuracy_and_empty():
    gps = track(0, 3 * MIN_MS, lambda t: (0.0, 0.0)).rename(
        columns={"accuracy_m": "accuracy"}
    )
    gps.loc[0, "accuracy"] = 1000.0
    filtered, counts = filter_gps_jitter(gps)
    assert counts["accuracy_exceeds_max"] == 1 and len(filtered) == 2

    empty, counts = filter_gps_jitter(gps.iloc[0:0])
    assert empty.empty and counts["n_kept"] == 0

    with pytest.raises(ValueError):
        filter_gps_jitter(gps, accuracy_max_m=0)
    with pytest.raises(ValueError, match="Missing required GPS columns"):
        filter_gps_jitter(gps.drop(columns="longitude"))


def _episode(start_min, end_min, east_m, north_m=0.0):
    lat, lon = offset(east_m, north_m)
    return {
        "episode_start_ms": start_min * MIN_MS,
        "episode_end_ms": end_min * MIN_MS,
        "duration_ms": (end_min - start_min) * MIN_MS,
        "representative_latitude": lat,
        "representative_longitude": lon,
    }


def test_dwell_threshold_rises_with_sparse_sampling():
    assert dwell_threshold_ms(t_floor_ms=600_000, dwell_multiplier=3, sampling_interval_ms=60_000) == 600_000
    assert dwell_threshold_ms(t_floor_ms=600_000, dwell_multiplier=3, sampling_interval_ms=300_000) == 900_000


def test_build_stays_merges_nearby_and_applies_dynamic_threshold():
    episodes = pd.DataFrame([
        _episode(0, 8, 0.0),
        _episode(9, 14, 20.0),       # within 50 m: same stay, 13 min observed
        _episode(20, 32, 2000.0),    # far away: new stay, 12 min observed
        _episode(40, 45, 4000.0),    # 5 min: below any threshold
    ])

    dense = build_stays(episodes, sampling_interval_ms=60_000)
    assert len(dense) == 2
    first = dense.iloc[0]
    assert first["n_episodes"] == 2
    assert first["observed_duration_ms"] == 13 * MIN_MS
    assert first["span_ms"] == 14 * MIN_MS  # includes the 1 min gap
    assert first["max_radius_m"] < 50

    # 5-minute sampling raises the threshold to 15 min: neither stay survives.
    sparse = build_stays(episodes, sampling_interval_ms=300_000)
    assert sparse.empty
    assert sparse.columns[0] == "stay_id"


def test_build_stays_validates_inputs():
    with pytest.raises(ValueError, match="sampling_interval_ms"):
        build_stays(pd.DataFrame(), sampling_interval_ms=0)
    with pytest.raises(ValueError, match="r_stay_m"):
        build_stays(pd.DataFrame(), sampling_interval_ms=1, r_stay_m=0)
    with pytest.raises(ValueError, match="Missing required episode columns"):
        build_stays(pd.DataFrame({"episode_start_ms": [0]}), sampling_interval_ms=1)
