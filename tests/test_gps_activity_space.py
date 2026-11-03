import numpy as np
import pandas as pd
import pytest
from gps_synthetic import HOUR_MS, MIN_MS, track

from cortex2_engine.features.gps_activity_space import (
    compute_daily_activity_space,
    convex_hull,
    convex_hull_area_km2,
    polygon_area,
    standard_deviational_ellipse,
)
from cortex2_engine.features.gps_intervals import build_gps_intervals


def test_convex_hull_of_square_with_interior_points():
    pts = np.array([[0, 0], [1000, 0], [1000, 1000], [0, 1000], [500, 500], [200, 700]])
    hull = convex_hull(pts)
    assert len(hull) == 4
    assert convex_hull_area_km2(pts) == pytest.approx(1.0)


def test_degenerate_hulls_have_zero_area():
    assert polygon_area(convex_hull(np.array([[0, 0], [0, 0]]))) == 0.0
    collinear = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    assert convex_hull_area_km2(collinear) == 0.0


def test_sde_recovers_axes_area_and_bearing():
    rng = np.random.default_rng(0)
    # Major axis along north-east (bearing 45): sd 2000 m and 500 m.
    major = rng.normal(0, 2000, 20_000)
    minor = rng.normal(0, 500, 20_000)
    angle = np.radians(45.0)
    east = major * np.sin(angle) + minor * np.cos(angle)
    north = major * np.cos(angle) - minor * np.sin(angle)

    sde = standard_deviational_ellipse(np.column_stack([east, north]))

    assert sde["sde_major_m"] == pytest.approx(2000, rel=0.03)
    assert sde["sde_minor_m"] == pytest.approx(500, rel=0.03)
    assert sde["anisotropy"] == pytest.approx(0.75, abs=0.02)
    assert sde["major_axis_bearing_deg"] == pytest.approx(45.0, abs=1.5)
    assert sde["sde_area_km2"] == pytest.approx(np.pi * 2000 * 500 / 1e6, rel=0.05)


def test_sde_weights_and_degenerate_cases():
    pts = np.array([[0.0, 0.0], [1000.0, 0.0], [0.0, 0.0]])
    weighted = standard_deviational_ellipse(pts, np.array([1.0, 0.0, 1.0]))
    assert weighted["sde_area_km2"] == 0.0
    assert np.isnan(weighted["anisotropy"])  # a point: orientation undefined

    east_west = standard_deviational_ellipse(pts)
    assert east_west["major_axis_bearing_deg"] == pytest.approx(90.0)

    assert np.isnan(standard_deviational_ellipse(np.empty((0, 2)))["sde_area_km2"])
    assert np.isnan(standard_deviational_ellipse(pts, np.zeros(3))["sde_area_km2"])


def _edges(positions, hours=8.0, accuracy_m=10.0):
    gps = track(0, int(hours * HOUR_MS), positions, accuracy_m=accuracy_m)
    return build_gps_intervals(gps, window_start_ms=0, window_end_ms=int(hours * HOUR_MS), max_gap_ms=600_000)


def test_daily_activity_space_for_moving_participant():
    # A 2 km x 1 km rectangle traversed over the day.
    corners = [(0, 0), (2000, 0), (2000, 1000), (0, 1000)]
    out = compute_daily_activity_space(
        _edges(lambda t: corners[int(t // (2 * HOUR_MS)) % 4])
    )
    assert out["convex_hull_area_km2"] == pytest.approx(2.0, rel=0.05)
    assert out["convex_hull_area_km2_status"] == "observed"
    assert out["sde_area_km2_status"] == "observed"
    assert out["major_axis_bearing_deg"] == pytest.approx(90.0, abs=1.0)
    assert out["zero_area_floor_km2"] == pytest.approx(1e-4)


def test_still_participant_is_observed_zero_not_missing():
    jitter = np.random.default_rng(1).normal(0, 1, (10_000, 2))
    out = compute_daily_activity_space(
        _edges(lambda t: tuple(jitter[int(t // MIN_MS) % 10_000]))
    )
    assert out["convex_hull_area_km2"] == 0.0
    assert out["convex_hull_area_km2_status"] == "observed_zero"
    assert out["sde_area_km2_status"] == "observed_zero"
    assert out["anisotropy_status"] == "not_applicable"
    assert np.isnan(out["major_axis_bearing_deg"])


def test_activity_space_insufficient_and_empty():
    short = compute_daily_activity_space(_edges(lambda t: (t / 1000.0, 0.0), hours=2))
    assert short["convex_hull_area_km2_status"] == "insufficient_observability"
    assert np.isnan(short["convex_hull_area_km2"])

    empty_edges = _edges(lambda t: (0.0, 0.0)).iloc[0:0]
    empty = compute_daily_activity_space(empty_edges, median_accuracy_m=10.0)
    assert empty["sde_area_km2_status"] == "no_raw_data"

    no_acc = _edges(lambda t: (t / 1000.0, 0.0)).drop(columns=["start_accuracy", "end_accuracy"])
    assert compute_daily_activity_space(no_acc)["zero_area_floor_km2"] == 0.0
    assert isinstance(pd.Series(compute_daily_activity_space(no_acc)), pd.Series)
