"""Daily accelerometer activity features for one participant-day.

Construct
---------
How much, and how fragmented, the participant's body movement was on one local
day, estimated only from time the phone was observed *and* carried.

Pipeline: raw samples -> unit detection -> 1 s epochs (ENMO) -> non-wear
(van Hees et al. 2011, phone-adapted) -> 1 min epochs with non-wear removed ->
mean ENMO and fragmentation (Di et al. 2017).

What this is NOT
----------------
- Not a step count or energy expenditure.
- Not circadian rhythm: IS/IV/RA need several days and are computed by
  ``acc_circadian.compute_circadian_features`` over a multi-day window, from
  the minute epochs returned by :func:`prepare_minute_epochs`.
- Not comparable across devices without calibration: ENMO from a phone in a
  pocket is attenuated relative to a wrist device.

Denominators are observed (worn) time, never 24 h. A DST day is 23 h or 25 h
long, and ``day_duration_ms`` reports that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cortex2_engine.features.acc_fragmentation import (
    FRAGMENTATION_KEYS,
    MINUTE_MS,
    activity_fragmentation,
)
from cortex2_engine.features.acc_nonwear import NonwearResult, detect_nonwear
from cortex2_engine.features.acc_preprocess import (
    UNIT_UNKNOWN,
    aggregate_epochs,
    resample_to_epochs,
)
from cortex2_engine.features.status import (
    FeatureStatus,
    gated_value,
    status_from_counts,
)

ACC_ALGORITHM = "cortex2.acc_daily"
ACC_ALGORITHM_VERSION = "1.0"

VALUE_KEYS = ["mean_enmo_mg", "acc_nonwear_fraction", *FRAGMENTATION_KEYS]


@dataclass
class MinuteEpochs:
    """Intermediate products shared by daily and circadian features.

    Attributes
    ----------
    minutes
        1-minute epochs (``aggregate_epochs`` output) with non-wear excluded.
    nonwear
        Non-wear result on the fine epochs.
    unit
        Detected or supplied input unit.
    n_samples
        Raw samples inside the window.
    """

    minutes: pd.DataFrame
    nonwear: NonwearResult
    unit: str
    n_samples: int


def prepare_minute_epochs(
    acc: pd.DataFrame,
    *,
    window_start_ms: int,
    window_end_ms: int,
    epoch_ms: int = 1000,
    unit: str | None = None,
    battery: pd.DataFrame | None = None,
    nonwear_window_ms: int = 3_600_000,
    nonwear_step_ms: int = 900_000,
    sd_threshold_mg: float = 13.0,
    range_threshold_mg: float = 50.0,
    min_axes: int = 2,
    minute_min_observed_fraction: float = 0.5,
) -> MinuteEpochs:
    """Run preprocessing and non-wear detection for one window.

    Parameters
    ----------
    acc
        Canonical accelerometer samples (``timestamp_ms``, ``x``, ``y``,
        ``z``).
    window_start_ms, window_end_ms
        Half-open window, normally one local day from
        ``cortex2_engine.time.iter_local_days``.
    epoch_ms
        Fine epoch length.
    unit
        ``"g"``, ``"m_s2"`` or ``None`` to detect.
    battery
        Optional screen rows for charging annotation.
    nonwear_window_ms, nonwear_step_ms, sd_threshold_mg, range_threshold_mg, min_axes
        Passed to ``detect_nonwear``.
    minute_min_observed_fraction
        Observed, worn share of a minute for it to be valid.

    Returns
    -------
    MinuteEpochs
        Intermediate products.
    """

    fine = resample_to_epochs(
        acc,
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
        epoch_ms=epoch_ms,
        unit=unit,
    )

    nonwear = detect_nonwear(
        fine,
        window_ms=nonwear_window_ms,
        step_ms=nonwear_step_ms,
        sd_threshold_mg=sd_threshold_mg,
        range_threshold_mg=range_threshold_mg,
        min_axes=min_axes,
        battery=battery,
    )

    minutes = aggregate_epochs(
        fine,
        epoch_ms=MINUTE_MS,
        min_observed_fraction=minute_min_observed_fraction,
        exclude=nonwear.nonwear,
    )

    return MinuteEpochs(
        minutes=minutes,
        nonwear=nonwear,
        unit=fine.attrs.get("unit", UNIT_UNKNOWN),
        n_samples=int(fine["n_samples"].sum()),
    )


def _all_status(
    keys: list[str],
    status: FeatureStatus,
) -> dict:

    out = {}

    for key in keys:
        out[key] = np.nan
        out[f"{key}_status"] = status.value

    return out


def compute_daily_activity_features(
    acc: pd.DataFrame,
    *,
    window_start_ms: int,
    window_end_ms: int,
    epoch_ms: int = 1000,
    unit: str | None = None,
    battery: pd.DataFrame | None = None,
    active_threshold_mg: float = 30.0,
    min_valid_minutes: int = 600,
    minute_min_observed_fraction: float = 0.5,
    nonwear_window_ms: int = 3_600_000,
    nonwear_step_ms: int = 900_000,
    sd_threshold_mg: float = 13.0,
    range_threshold_mg: float = 50.0,
    min_axes: int = 2,
) -> dict:
    """Compute daily accelerometer features for one participant-day.

    Parameters
    ----------
    acc
        Canonical accelerometer samples for the day (extra rows outside the
        window are ignored).
    window_start_ms, window_end_ms
        Half-open local-day bounds.
    epoch_ms
        Fine epoch length (1 s).
    unit
        ``"g"``, ``"m_s2"`` or ``None`` to detect.
    battery
        Optional screen rows with ``battery_level_fraction``.
    active_threshold_mg
        Active-minute ENMO threshold (30 mg).
    min_valid_minutes
        Minimum worn, observed minutes (600 = 10 h) for activity values.
    minute_min_observed_fraction
        Observed share of a minute for it to be valid.
    nonwear_window_ms, nonwear_step_ms, sd_threshold_mg, range_threshold_mg, min_axes
        Non-wear parameters (van Hees et al. 2011 defaults).

    Returns
    -------
    dict
        Flat participant-day record: ``mean_enmo_mg``, non-wear summary,
        fragmentation features (each with ``_status``), plus ``acc_unit``,
        ``acc_n_samples``, ``acc_valid_minutes``, ``day_duration_ms``,
        ``acc_observed_fraction``, ``acc_error_type``, ``algorithm`` and
        ``algorithm_version``.
    """

    window_start_ms = int(window_start_ms)
    window_end_ms = int(window_end_ms)
    day_duration_ms = window_end_ms - window_start_ms

    prepared = prepare_minute_epochs(
        acc,
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
        epoch_ms=epoch_ms,
        unit=unit,
        battery=battery,
        nonwear_window_ms=nonwear_window_ms,
        nonwear_step_ms=nonwear_step_ms,
        sd_threshold_mg=sd_threshold_mg,
        range_threshold_mg=range_threshold_mg,
        min_axes=min_axes,
        minute_min_observed_fraction=minute_min_observed_fraction,
    )

    summary = dict(prepared.nonwear.summary)

    context = {
        "acc_unit": prepared.unit,
        "acc_n_samples": prepared.n_samples,
        "day_duration_ms": float(day_duration_ms),
        "acc_observed_fraction": summary["acc_observed_ms"] / day_duration_ms,
        "acc_error_type": None,
        "algorithm": ACC_ALGORITHM,
        "algorithm_version": ACC_ALGORITHM_VERSION,
    }

    if prepared.n_samples == 0:
        out = _all_status(VALUE_KEYS, FeatureStatus.NO_RAW_DATA)
        out.update({k: v for k, v in summary.items() if k not in out})
        out["acc_valid_minutes"] = 0
        out.update(context)
        return out

    if prepared.unit == UNIT_UNKNOWN:
        out = _all_status(VALUE_KEYS, FeatureStatus.EXTRACTION_FAILED)
        out.update({
            k: v for k, v in summary.items()
            if k not in out and not k.startswith("acc_nonwear_fraction")
        })
        out["acc_valid_minutes"] = 0
        context["acc_error_type"] = "accelerometer_unit_unknown"
        out.update(context)
        return out

    minutes = prepared.minutes

    fragmentation = activity_fragmentation(
        minutes,
        active_threshold_mg=active_threshold_mg,
        min_valid_minutes=min_valid_minutes,
    )

    valid = minutes["valid"].to_numpy(dtype=bool)
    valid_minutes = int(valid.sum())

    mean_enmo = (
        float(minutes.loc[valid, "enmo_mg"].mean())
        if valid_minutes
        else np.nan
    )

    enmo_status = status_from_counts(
        valid_minutes * MINUTE_MS,
        min_valid_minutes * MINUTE_MS,
        mean_enmo,
    )

    out = {
        "mean_enmo_mg": gated_value(mean_enmo, enmo_status),
        "mean_enmo_mg_status": enmo_status.value,
    }

    out.update(summary)
    out.update(fragmentation)
    out["acc_valid_minutes"] = valid_minutes
    out.update(context)

    return out
