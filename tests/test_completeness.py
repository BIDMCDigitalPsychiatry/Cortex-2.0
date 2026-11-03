import numpy as np
import pandas as pd
import pytest
from acc_synthetic import HOUR, samples
from gps_synthetic import track

from cortex2_engine.features.acc_daily import compute_daily_activity_features
from cortex2_engine.features.communication import compute_communication_features
from cortex2_engine.features.gps_activity_space import compute_daily_activity_space
from cortex2_engine.features.gps_intervals import build_gps_intervals
from cortex2_engine.features.gps_location_features import (
    compute_daily_location_features,
)
from cortex2_engine.features.gps_mobility_features import (
    compute_daily_mobility_features,
)
from cortex2_engine.features.gps_spatial_semantics import add_gps_spatial_semantics
from cortex2_engine.features.gps_trajectory_plausibility import (
    add_gps_trajectory_plausibility,
)
from cortex2_engine.features.screen_bouts import (
    build_screen_bouts,
    compute_daily_screen_features,
)
from cortex2_engine.features.wearable_sleep import compute_wearable_sleep
from cortex2_engine.quality.completeness import (
    completeness_weight_from_coverage,
    completeness_weight_from_intervals,
    completeness_weight_from_mask,
    missing_mask_from_intervals,
    multi_stream_weight,
    participant_day_completeness_weight,
    window_weight,
)


def test_weight_formula_and_extremes():
    assert participant_day_completeness_weight(1.0, 0, 0, 24) == 1.0
    # 75% coverage, 6 h longest gap of 24 h: 0.75 * 0.75
    assert participant_day_completeness_weight(0.75, 6 * HOUR, 1, 24) == pytest.approx(0.5625)
    assert participant_day_completeness_weight(0.75, 6 * HOUR, 1, 24, beta=0) == pytest.approx(0.75)
    assert participant_day_completeness_weight(0.75, 6 * HOUR, 1, 24, alpha=2) == pytest.approx(0.75**3)
    assert participant_day_completeness_weight(0.0, 24 * HOUR, 1, 24) == 0.0
    assert np.isnan(participant_day_completeness_weight(0.5, 0, 0, 0))
    assert np.isnan(participant_day_completeness_weight(np.nan, 0, 0, 24))


def test_dst_day_normalises_by_its_own_length():
    # The same 6 h gap weighs less on a 25 h day than on a 23 h day.
    w23 = participant_day_completeness_weight(17 / 23, 6 * HOUR, 1, 23)
    w25 = participant_day_completeness_weight(19 / 25, 6 * HOUR, 1, 25)
    assert w25 > w23


def test_weight_input_validation():
    for bad in (
        {"coverage_fraction": 1.5},
        {"longest_gap_ms": -1},
        {"longest_gap_ms": 25 * HOUR},
        {"n_missing_runs": 30},
        {"n_missing_runs": 0},  # disagrees with a positive gap
    ):
        args = {"coverage_fraction": 0.5, "longest_gap_ms": HOUR, "n_missing_runs": 1, "n_bins": 24} | bad
        with pytest.raises(ValueError):
            participant_day_completeness_weight(**args)
    with pytest.raises(ValueError):
        participant_day_completeness_weight(1, 0, 0, 24, alpha=-1)
    with pytest.raises(ValueError):
        participant_day_completeness_weight(1, 0, 0, 24, bin_size_ms=0)


def test_concentrated_gaps_weigh_less_than_scattered():
    scattered = np.zeros(24, bool)
    scattered[::4] = True
    block = np.zeros(24, bool)
    block[:6] = True
    a = completeness_weight_from_mask(scattered, bin_size_ms=HOUR)
    b = completeness_weight_from_mask(block, bin_size_ms=HOUR)
    assert a > b
    assert b == pytest.approx(0.75 * 0.75)
    assert np.isnan(completeness_weight_from_mask([], bin_size_ms=HOUR))


def test_multi_stream_geometric_mean():
    assert multi_stream_weight({"gps": 0.25, "acc": 1.0}) == pytest.approx(0.5)
    assert multi_stream_weight({"gps": 0.0, "acc": 1.0}) == 0.0
    assert multi_stream_weight({"gps": np.nan, "acc": 0.81}) == pytest.approx(0.81)
    assert np.isnan(multi_stream_weight({}))
    with pytest.raises(ValueError):
        multi_stream_weight({"gps": 1.2})


def test_intervals_merge_and_coverage_bound():
    mask = missing_mask_from_intervals(
        [(0, 2 * HOUR), (HOUR, 3 * HOUR), (np.nan, 5)],
        window_start_ms=0, window_end_ms=6 * HOUR, bin_size_ms=HOUR,
    )
    assert mask.tolist() == [False, False, False, True, True, True]

    exact = completeness_weight_from_intervals(
        [(0, 3 * HOUR), (4 * HOUR, 6 * HOUR)], window_start_ms=0, window_end_ms=6 * HOUR
    )
    bound = completeness_weight_from_coverage(5 * HOUR, 6 * HOUR)
    assert bound <= exact
    assert np.isnan(completeness_weight_from_coverage(None, HOUR))
    assert np.isnan(completeness_weight_from_coverage(1.0, 0))
    assert np.isnan(window_weight(window_start_ms=None, window_end_ms=1))
    with pytest.raises(ValueError):
        missing_mask_from_intervals([], window_start_ms=1, window_end_ms=1)


def test_every_daily_dict_carries_completeness_weight():
    day = 24 * HOUR

    acc = samples(0, 12 * HOUR, moving=True)
    a = compute_daily_activity_features(acc, window_start_ms=0, window_end_ms=day)
    assert a["completeness_weight"] == pytest.approx(0.25, abs=0.01)  # 0.5 coverage x (1 - 0.5 gap)

    bouts = build_screen_bouts(
        pd.DataFrame({"timestamp_ms": [HOUR, 2 * HOUR], "event": ["screen_on", "screen_off"]}),
        window_start_ms=0, window_end_ms=day,
    )
    s = compute_daily_screen_features(bouts, supported_duration_ms=day, window_start_ms=0, window_end_ms=day)
    assert s["completeness_weight"] == 1.0
    assert np.isnan(compute_daily_screen_features(bouts, supported_duration_ms=day)["completeness_weight"])

    gps = track(0, 12 * HOUR, lambda t: (t / 1000.0, 0.0))
    edges = build_gps_intervals(gps, window_start_ms=0, window_end_ms=day, max_gap_ms=600_000)
    semantic = add_gps_trajectory_plausibility(add_gps_spatial_semantics(edges))
    m = compute_daily_mobility_features(semantic, window_start_ms=0, window_end_ms=day)
    assert m["completeness_weight"] == pytest.approx(0.25, abs=0.02)
    assert np.isnan(compute_daily_mobility_features(semantic)["completeness_weight"])

    sp = compute_daily_activity_space(edges, window_start_ms=0, window_end_ms=day)
    assert sp["completeness_weight"] == pytest.approx(m["completeness_weight"])

    loc = compute_daily_location_features(
        pd.DataFrame({"place_candidate_id": [1], "duration_ms": [6 * HOUR]}),
        supported_duration_ms=12 * HOUR, window_start_ms=0, window_end_ms=day,
    )
    assert loc["completeness_weight"] == pytest.approx(0.25)

    tel = pd.DataFrame({"timestamp_ms": [HOUR], "channel": ["call"], "direction": ["outgoing"]})
    c = compute_communication_features(tel, window_start_ms=0, window_end_ms=day, source_observed_ms=day)
    assert c["completeness_weight"] == 1.0

    w = compute_wearable_sleep(
        pd.DataFrame(columns=["timestamp_ms", "representation", "duration_ms"]),
        night_start_ms=0, night_end_ms=day,
    )
    assert np.isnan(w["completeness_weight"])
