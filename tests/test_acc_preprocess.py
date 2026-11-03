import numpy as np
import pandas as pd
import pytest
from acc_synthetic import HOUR, MINUTE, SECOND, samples

from cortex2_engine.features.acc_preprocess import (
    STANDARD_GRAVITY_M_S2,
    aggregate_epochs,
    detect_unit,
    enmo_g,
    resample_to_epochs,
)


def test_enmo_truncates_below_one_g():

    assert enmo_g(
        np.array([0.0, 0.0, 2.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.5, 1.0, 0.0]),
    ).tolist() == pytest.approx([0.0, 0.0, 1.0])


def test_detect_unit_g_and_m_s2():

    acc = samples(0, 10 * MINUTE, moving=False)

    assert detect_unit(acc) == "g"

    scaled = acc.assign(
        x=acc["x"] * STANDARD_GRAVITY_M_S2,
        y=acc["y"] * STANDARD_GRAVITY_M_S2,
        z=acc["z"] * STANDARD_GRAVITY_M_S2,
    )

    assert detect_unit(scaled) == "m_s2"


def test_detect_unit_gravity_removed_is_unknown():

    acc = samples(0, MINUTE, moving=False).assign(z=0.0)

    assert detect_unit(acc) == "unknown"
    assert detect_unit(acc.iloc[0:0]) == "unknown"


def test_detect_unit_rejects_bad_band():

    with pytest.raises(ValueError):
        detect_unit(samples(0, MINUTE, moving=False), gravity_band_g=(1.3, 0.7))


def test_resample_grid_covers_window_including_unobserved():

    acc = samples(0, 10 * SECOND, moving=False)

    epochs = resample_to_epochs(
        acc,
        window_start_ms=0,
        window_end_ms=20 * SECOND,
    )

    assert len(epochs) == 20
    assert epochs["observed"].sum() == 10
    assert epochs.loc[15, "n_samples"] == 0
    assert np.isnan(epochs.loc[15, "enmo_mg"])
    assert epochs.attrs["unit"] == "g"


def test_resample_converts_m_s2_to_g():

    acc = pd.DataFrame({
        "timestamp_ms": [0, 500],
        "x": [0.0, 0.0],
        "y": [0.0, 0.0],
        "z": [2 * STANDARD_GRAVITY_M_S2, 2 * STANDARD_GRAVITY_M_S2],
    })

    epochs = resample_to_epochs(
        acc,
        window_start_ms=0,
        window_end_ms=SECOND,
        unit="m_s2",
    )

    assert epochs.loc[0, "mean_z_g"] == pytest.approx(2.0)
    assert epochs.loc[0, "enmo_mg"] == pytest.approx(1000.0)
    assert epochs.loc[0, "n_samples"] == 2


def test_resample_unknown_unit_keeps_counts_but_no_values():

    acc = samples(0, 5 * SECOND, moving=False).assign(z=0.0)

    epochs = resample_to_epochs(acc, window_start_ms=0, window_end_ms=5 * SECOND)

    assert epochs.attrs["unit"] == "unknown"
    assert epochs["n_samples"].sum() == 5
    assert epochs["enmo_mg"].isna().all()


def test_resample_validates_inputs():

    acc = samples(0, SECOND, moving=False)

    with pytest.raises(ValueError):
        resample_to_epochs(acc, window_start_ms=10, window_end_ms=10)

    with pytest.raises(ValueError):
        resample_to_epochs(acc, window_start_ms=0, window_end_ms=10, epoch_ms=0)

    with pytest.raises(ValueError):
        resample_to_epochs(acc, window_start_ms=0, window_end_ms=10, unit="km")

    with pytest.raises(ValueError):
        resample_to_epochs(
            acc.drop(columns="z"),
            window_start_ms=0,
            window_end_ms=10,
        )


def test_aggregate_epochs_to_minutes_with_exclusion():

    acc = samples(0, 2 * MINUTE, moving=True)

    fine = resample_to_epochs(acc, window_start_ms=0, window_end_ms=3 * MINUTE)

    exclude = np.zeros(len(fine), dtype=bool)
    exclude[60:100] = True

    minutes = aggregate_epochs(fine, exclude=exclude)

    assert minutes["epoch_start_ms"].tolist() == [0, MINUTE, 2 * MINUTE]
    assert minutes["observed_ms"].tolist() == [60_000, 20_000, 0]
    assert minutes["excluded_ms"].tolist() == [0, 40_000, 0]
    assert minutes["valid"].tolist() == [True, False, False]
    assert minutes.loc[0, "enmo_mg"] > 0


def test_aggregate_epochs_validation():

    fine = resample_to_epochs(
        samples(0, MINUTE, moving=False),
        window_start_ms=0,
        window_end_ms=HOUR,
    )

    with pytest.raises(ValueError):
        aggregate_epochs(fine, epoch_ms=1500)

    with pytest.raises(ValueError):
        aggregate_epochs(fine, min_observed_fraction=2.0)

    with pytest.raises(ValueError):
        aggregate_epochs(fine, exclude=np.zeros(3, dtype=bool))

    assert aggregate_epochs(fine.iloc[0:0]).empty
