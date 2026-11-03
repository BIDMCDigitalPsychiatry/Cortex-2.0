"""End-to-end: mindLAMP files -> `cortex2 features` -> long parquet."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from acc_synthetic import samples
from gps_synthetic import HOUR_MS, local_ms, offset
from typer.testing import CliRunner

from cortex2_engine.cli import app
from cortex2_engine.features.daily_driver import (
    FEATURE_SPECS,
    LONG_COLUMNS,
    DriverParameters,
)
from cortex2_engine.features.status import EMITTABLE_STATUSES, VALUE_BEARING_STATUSES

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "mindlamp"
TZ = "America/New_York"
UID = "U1"
OBSERVED_DAYS = ["2026-05-04", "2026-05-05"]
EMPTY_DAY = "2026-05-06"


def _write_study(root: Path) -> Path:
    raw = root / "raw"
    for name in ("gps", "acc", "screen", "telephony"):
        (raw / name).mkdir(parents=True)

    gps, acc, screen = [], [], []
    for date in OBSERVED_DAYS:
        t = np.arange(local_ms(date, 0), local_ms(date, 24), 60_000)
        hours = (t - local_ms(date, 0)) / HOUR_MS
        east = np.where((hours >= 9) & (hours < 17), 3000.0, 0.0)
        lat, lon = offset(east, np.zeros_like(east))
        gps.append(pd.DataFrame({"timestamp": t, "latitude": lat, "longitude": lon, "accuracy": 10.0}))

        a = samples(local_ms(date, 8), local_ms(date, 20), moving=True)
        acc.append(a.rename(columns={"timestamp_ms": "timestamp"}))

        for h in range(7, 23, 2):
            screen.append({"timestamp": local_ms(date, h), "representation": "screen_on", "battery_level": 0.8})
            screen.append({"timestamp": local_ms(date, h + 0.25), "representation": "screen_off", "battery_level": 0.8})

    pd.concat(gps).to_csv(raw / "gps" / f"gps_{UID}_1.csv", index=False)
    pd.concat(acc).to_csv(raw / "acc" / f"acc_{UID}_1.csv", index=False)
    pd.DataFrame(screen).to_csv(raw / "screen" / f"screen_{UID}_1.csv", index=False)

    tel = pd.read_csv(next((FIXTURES / "telephony").glob("*.csv")))
    tel["timestamp"] = [local_ms("2026-05-04", 10 + i) for i in range(len(tel))]
    tel.to_csv(raw / "telephony" / f"telephony_{UID}_1.csv", index=False)

    pd.DataFrame({
        "participant_id": ["P1"],
        "mindlamp_uid": [UID],
        "clinical_start": ["2026-05-04"],
        "clinical_end": ["2026-05-06"],
    }).to_csv(root / "windows.csv", index=False)

    config = {
        "name": "e2e",
        "timezone": TZ,
        "day_start": "00:00",
        "output_dir": "out",
        "windows": {"path": "windows.csv"},
        "adapter": {"kind": "mindlamp", "passive_root": "raw"},
    }
    path = root / "study.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture(scope="module")
def long_table(tmp_path_factory):
    root = tmp_path_factory.mktemp("study")
    config = _write_study(root)
    result = CliRunner().invoke(
        app, ["features", "--config", str(config), "--start", "2026-05-04", "--end", "2026-05-06"]
    )
    assert result.exit_code == 0, result.output
    path = Path(result.output.strip().splitlines()[-1])
    assert path == root / "out" / "features" / "daily_features_long.parquet"
    df = pd.read_parquet(path)
    shutil.rmtree(root)
    return df


def test_long_schema_and_dates(long_table):
    assert list(long_table.columns) == LONG_COLUMNS
    assert sorted(long_table["date_local"].unique()) == [*OBSERVED_DAYS, EMPTY_DAY]
    assert long_table["parameter_set_hash"].nunique() == 1
    assert long_table["parameter_set_hash"].iloc[0] == DriverParameters().hash()
    n_features = sum(len(v) for v in FEATURE_SPECS.values())
    assert len(long_table) == 3 * n_features
    assert not long_table.duplicated(["episode_id", "date_local", "feature"]).any()


def test_statuses_are_valid_and_zeros_are_observed(long_table):
    emittable = {s.value for s in EMITTABLE_STATUSES}
    bearing = {s.value for s in VALUE_BEARING_STATUSES}
    assert set(long_table["feature_status"]) <= emittable

    zeros = long_table[long_table["value"] == 0]
    assert set(zeros["feature_status"]) <= {"observed_zero"}

    no_value = long_table[~long_table["feature_status"].isin(bearing)]
    assert no_value["value"].isna().all()


def test_unobserved_day_has_no_bare_zeros(long_table):
    day = long_table[long_table["date_local"] == EMPTY_DAY].set_index("feature")
    assert (day["value"].fillna(-1) != 0).all()
    assert day.loc["mean_enmo_mg", "feature_status"] == "no_raw_data"
    assert day.loc["convex_hull_area_km2", "feature_status"] == "no_raw_data"
    assert day.loc["screen_on_ms", "feature_status"] == "no_raw_data"
    # Telephony exists in the episode but the day has no screen-supported time.
    assert day.loc["call_count", "feature_status"] == "no_raw_data"


def test_observed_day_values(long_table):
    day = long_table[long_table["date_local"] == "2026-05-04"].set_index("feature")
    assert day.loc["mean_enmo_mg", "feature_status"] == "observed"
    assert day.loc["mean_enmo_mg", "unit"] == "mg"
    assert day.loc["active_minutes", "observed_duration_ms"] == pytest.approx(12 * HOUR_MS, rel=0.01)
    assert day.loc["supported_displacement_distance_m", "value"] == pytest.approx(6000, rel=0.05)
    assert day.loc["supported_radius_of_gyration_m", "feature_status"] == "observed"
    assert day.loc["n_bouts", "value"] == 8
    assert day.loc["call_count", "value"] > 0
    assert 0 < day.loc["mean_enmo_mg", "completeness_weight"] <= 1
    assert day.loc["convex_hull_area_km2", "observability"] == pytest.approx(1.0, abs=0.01)
    # Two nights cannot establish a home (min_nights = 5).
    assert day.loc["home_time_ms", "feature_status"] == "insufficient_observability"


def test_cli_rejects_bad_dates(tmp_path):
    config = _write_study(tmp_path)
    runner = CliRunner()
    bad = runner.invoke(app, ["features", "--config", str(config), "--start", "May 4"])
    assert bad.exit_code != 0
    reversed_ = runner.invoke(
        app, ["features", "--config", str(config), "--start", "2026-05-06", "--end", "2026-05-04"]
    )
    assert reversed_.exit_code != 0


def test_driver_features_are_registered_with_matching_units():
    registry = yaml.safe_load((REPO / "feature_registry" / "features.yaml").read_text())["features"]
    for family, spec in FEATURE_SPECS.items():
        for feature, (unit, _) in spec.items():
            assert feature in registry, (family, feature)
            assert registry[feature]["unit"] == unit, feature


def test_registry_integrity():
    registry = yaml.safe_load((REPO / "feature_registry" / "features.yaml").read_text())["features"]
    # Guards against an accidental truncation of the file.
    assert len(registry) >= 79
    required = {
        "engine", "native", "source_stream", "unit",
        "reliability_metric", "minimum_observability", "reference",
    }
    for name, entry in registry.items():
        if entry.get("native"):
            assert required <= set(entry), (name, sorted(required - set(entry)))
            assert entry["engine"] == "cortex2", name
