import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from acc_synthetic import HOUR, samples

from cortex2_engine.features.acc_daily import compute_daily_activity_features
from cortex2_engine.features.degradation_sweep import (
    ReferenceDay,
    degrade_streams,
    reliability_thresholds,
    run_degradation_sweep,
    summarise_sweep,
    within_person_mad,
    write_reliability_thresholds,
)
from cortex2_engine.features.gps_degradation import (
    apply_missing_bin_mask_to_points,
    make_missing_bin_mask,
)
from cortex2_engine.features.missingness_simulation import (
    make_balanced_k_run_missing_bin_mask,
)

REPO = Path(__file__).resolve().parents[1]
DAY = 24 * HOUR


def _active_day(seed, active_hours):
    """Moving for ``active_hours`` (from 08:00), still otherwise; 0.2 Hz."""
    return pd.concat([
        samples(0, 8 * HOUR, moving=False, seed=seed, hz=0.2),
        samples(8 * HOUR, (8 + active_hours) * HOUR, moving=True, seed=seed + 1, hz=0.2),
        samples((8 + active_hours) * HOUR, DAY, moving=False, seed=seed + 2, hz=0.2),
    ], ignore_index=True)


def _acc_fn(streams, start, end):
    return compute_daily_activity_features(
        streams["accelerometer"], window_start_ms=start, window_end_ms=end, epoch_ms=5000, min_valid_minutes=120
    )


def _reference_days():
    days = []
    for pid, base in (("P1", 4), ("P2", 8)):
        for k, extra in enumerate((0, 1, 2)):
            days.append(ReferenceDay(pid, {"accelerometer": _active_day(10 * k, base + extra)}, 0, DAY))
    return days


MINUTE = 60_000


def _toy_fn(streams, start, end):
    """Total of per-minute values; insufficient below 6 observed hours."""
    rows = streams["toy"]
    n = len(rows)
    if n < 360:
        return {"total_x": float("nan"), "total_x_status": "insufficient_observability"}
    return {"total_x": float(rows["x"].sum()), "total_x_status": "observed"}


def _toy_days():
    """Per-minute value 1.0, 1.3 or 1.6: totals 1440, 1872, 2304 (MAD 432)."""
    t = np.arange(0, DAY, MINUTE)
    return [
        ReferenceDay(pid, {"toy": pd.DataFrame({"timestamp_ms": t, "x": level})}, 0, DAY)
        for pid in ("P1", "P2")
        for level in (1.0, 1.3, 1.6)
    ]


@pytest.fixture(scope="module")
def sweep():
    return run_degradation_sweep(
        _toy_days(), _toy_fn, ["total_x"],
        coverages=(0.9, 0.5, 0.15), n_runs=(1, 3), n_replicates=2,
    )


def test_sweep_shape_and_errors_grow_as_coverage_falls(sweep):
    assert set(sweep.columns) >= {"participant_id", "coverage", "n_runs", "abs_error"}
    assert len(sweep) == 6 * 3 * 2 * 2
    # Day-level reference values are identical across cells.
    assert sweep.groupby("day_index")["reference_value"].nunique().max() == 1

    s = summarise_sweep(sweep)
    one = s[s["n_runs"] == 1].set_index("coverage")
    # Exact coverage: the error is the missing share of the day's total.
    assert one.loc[0.9, "median_abs_error"] == pytest.approx(0.1 * 1872, rel=0.01)
    assert one.loc[0.5, "median_abs_error"] == pytest.approx(0.5 * 1872, rel=0.01)
    # At 15 % coverage every day is insufficient: unavailable, not "error".
    assert one.loc[0.15, "fraction_unavailable"] == 1.0
    assert np.isnan(one.loc[0.15, "median_abs_error"])


def test_within_person_mad_and_thresholds(sweep):
    mad = within_person_mad(sweep)
    assert mad["total_x"] == pytest.approx(432.0)
    assert np.isnan(within_person_mad(sweep, min_days_per_participant=4)["total_x"])

    t = reliability_thresholds(summarise_sweep(sweep)).set_index("feature")
    assert t.loc["total_x", "failure_coverage"] == 0.5
    assert t.loc["total_x", "min_reliable_coverage"] == 0.9


def test_accelerometer_feature_runs_through_the_sweep():
    days = _reference_days()[:1]
    out = run_degradation_sweep(
        days, _acc_fn, ["active_minutes", "mean_enmo_mg"],
        coverages=(0.9,), n_runs=(1,), n_replicates=2,
    )
    assert len(out) == 4
    assert np.isfinite(out["reference_value"]).all()
    assert (out.loc[out["feature"] == "active_minutes", "reference_value"] == 240).all()
    assert np.isfinite(out["abs_error"]).all()


def test_threshold_edge_cases():
    summary = pd.DataFrame({
        "feature": ["a", "a", "b", "b", "c"],
        "coverage": [0.9, 0.5, 0.9, 0.5, 0.9],
        "n_runs": 1,
        "exceeds_mad": [False, False, True, True, False],
        "within_person_mad": [1.0, 1.0, 1.0, 1.0, np.nan],
    })
    t = reliability_thresholds(summary).set_index("feature")
    assert np.isnan(t.loc["a", "failure_coverage"]) and t.loc["a", "min_reliable_coverage"] == 0.5
    assert t.loc["b", "failure_coverage"] == 0.9 and np.isnan(t.loc["b", "min_reliable_coverage"])
    assert np.isnan(t.loc["c", "min_reliable_coverage"])


def test_degrade_streams_masks_selected_streams_only():
    rows = pd.DataFrame({"timestamp_ms": np.arange(0, 4 * HOUR, 60_000)})
    mask = np.array([True, False, False, False])
    out = degrade_streams(
        {"gps": rows, "screen": rows, "empty": rows.iloc[0:0]}, mask,
        window_start_ms=0, window_end_ms=4 * HOUR, bin_size_ms=HOUR, degrade=["gps", "empty"],
    )
    assert len(out["gps"]) == 180 and len(out["screen"]) == 240 and out["empty"].empty


def test_sweep_validation_and_infeasible_cells():
    days = _reference_days()[:1]
    with pytest.raises(ValueError):
        run_degradation_sweep(days, _acc_fn, ["active_minutes"], n_replicates=0)
    with pytest.raises(ValueError):
        run_degradation_sweep(days, _acc_fn, [])
    # 1000 runs cannot fit: the cell is skipped rather than failing the sweep.
    out = run_degradation_sweep(days, _acc_fn, ["active_minutes"], coverages=(0.5,), n_runs=(1000,), n_replicates=1)
    assert out.empty

    weird = run_degradation_sweep(
        days, lambda s, a, b: {"x": "text", "y": 1.0, "y_status": "no_raw_data"}, ["x", "y"],
        coverages=(0.5,), n_runs=(1,), n_replicates=1,
    )
    assert weird["reference_value"].isna().all()


def test_write_reliability_thresholds_requires_reference_set(tmp_path):
    registry = tmp_path / "features.yaml"
    shutil.copy(REPO / "feature_registry" / "features.yaml", registry)

    with pytest.raises(ValueError, match="reference_set is required"):
        write_reliability_thresholds({"active_minutes": 0.6}, registry, reference_set=None)

    updated = write_reliability_thresholds(
        {"active_minutes": 0.6, "mean_enmo_mg": np.nan}, registry, reference_set="bidmc_ref_v1"
    )
    assert updated == ["active_minutes"]
    again = write_reliability_thresholds({"active_minutes": 0.7}, registry, reference_set="bidmc_ref_v2")
    assert again == ["active_minutes"]

    data = yaml.safe_load(registry.read_text())["features"]
    assert data["active_minutes"]["reliability_threshold"] == 0.7
    assert data["active_minutes"]["reliability_reference_set"] == "bidmc_ref_v2"
    assert data["mean_enmo_mg"]["reliability_threshold"] is None
    assert "# --- Phase 3" in registry.read_text()  # comments kept

    with pytest.raises(ValueError, match="not in registry"):
        write_reliability_thresholds({"nope": 0.5}, registry, reference_set="x")

    registry.write_text("features:\n  bare:\n    unit: x\n")
    with pytest.raises(ValueError, match="no reliability_threshold"):
        write_reliability_thresholds({"bare": 0.5}, registry, reference_set="x")


def test_shipped_registry_thresholds_stay_null():
    data = yaml.safe_load((REPO / "feature_registry" / "features.yaml").read_text())["features"]
    native = [k for k, v in data.items() if v.get("native")]
    assert all(data[k]["reliability_threshold"] is None for k in native)


def test_existing_mask_generators_validate_inputs():
    # Validation branches of the reused mask helpers.
    base = {"window_start_ms": 0, "window_end_ms": DAY, "bin_size_ms": HOUR, "seed": 0}
    for bad in (
        {"window_end_ms": 0},
        {"target_coverage": 1.0},
        {"bin_size_ms": 0},
        {"n_runs": 0},
        {"target_coverage": 0.99},  # zero missing bins
        {"n_runs": 20, "target_coverage": 0.5},  # too many runs
    ):
        args = {"target_coverage": 0.5, "n_runs": 2} | base | bad
        with pytest.raises(ValueError):
            make_balanced_k_run_missing_bin_mask(**args)

    for bad in (
        {"window_end_ms": 0},
        {"target_coverage": 0.0},
        {"bin_size_ms": 0},
        {"pattern": "spiral"},
    ):
        args = {"target_coverage": 0.5, "pattern": "contiguous"} | base | bad
        with pytest.raises(ValueError):
            make_missing_bin_mask(**args)

    full, _ = make_missing_bin_mask(target_coverage=1.0, pattern="distributed", **base)
    assert not full.any()

    pts = pd.DataFrame({"timestamp_ms": [0, HOUR]})
    with pytest.raises(ValueError):
        apply_missing_bin_mask_to_points(pts, window_start_ms=0, window_end_ms=DAY, bin_size_ms=HOUR, missing_mask=np.array([]))
    assert apply_missing_bin_mask_to_points(
        pts.iloc[0:0], window_start_ms=0, window_end_ms=DAY, bin_size_ms=HOUR, missing_mask=full
    ).empty
