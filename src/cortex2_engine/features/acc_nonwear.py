"""Device not-carried ("non-wear") detection from accelerometer stillness.

Construct
---------
Time during which the phone was sampled but lying still, i.e. evidently not
on the person. This follows the van Hees et al. (2011) rule as implemented in
GGIR. In sliding windows, an axis is *still* when its standard deviation is
below ``sd_threshold_mg`` **and** its value range is below
``range_threshold_mg``; a window is non-wear when at least ``min_axes`` axes
are still.

Smartphone adaptation
---------------------
A phone is carried, not worn, and the rule's meaning shifts accordingly:

- Non-wear means "phone not on the body" (on a desk, bedside table, charger).
  It is still the right exclusion for activity features, which describe body
  movement.
- Epochs with **no samples** are *unobserved*, never non-wear. Only observed
  epochs inside a still window are marked; a window must have at least
  ``min_window_observed_fraction`` observed time to be judged.
- Non-wear is the **union** of still windows, so every detected non-wear
  period lasts at least ``window_ms`` and a still period of exactly
  ``window_ms`` (aligned to ``step_ms``) is recovered in full. GGIR instead
  labels only the centre block of each window.
- Optional battery evidence (screen stream ``battery_level_fraction``) marks
  non-wear intervals during which the battery rose (``charging``). It annotates
  the intervals and does not change the classification.

What this is NOT
----------------
Not sleep detection: a phone on a bedside table at night is non-wear here;
``sleep_window`` combines this with screen evidence. Not a quality score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cortex2_engine.features.acc_preprocess import AXES
from cortex2_engine.features.status import (
    FeatureStatus,
    gated_value,
    status_from_counts,
)

INTERVAL_COLUMNS = [
    "start_ms",
    "end_ms",
    "duration_ms",
    "nonwear_observed_ms",
    "charging",
]


@dataclass
class NonwearResult:
    """Output of :func:`detect_nonwear`.

    Attributes
    ----------
    nonwear
        Boolean array aligned with the input epochs (True = observed and
        non-wear).
    intervals
        Merged non-wear intervals (``INTERVAL_COLUMNS``).
    summary
        Flat dict: ``acc_observed_ms``, ``acc_evaluable_ms``,
        ``acc_nonwear_ms``, ``acc_wear_ms``, ``acc_nonwear_fraction`` (+
        ``_status``), ``acc_nonwear_charging_ms``.
    """

    nonwear: np.ndarray
    intervals: pd.DataFrame
    summary: dict = field(default_factory=dict)


def _block_sums(
    values: np.ndarray,
    block: np.ndarray,
    n_blocks: int,
) -> np.ndarray:

    return np.bincount(
        block,
        weights=np.nan_to_num(values),
        minlength=n_blocks,
    )


def _block_extreme(
    values: np.ndarray,
    block: np.ndarray,
    n_blocks: int,
    *,
    maximum: bool,
) -> np.ndarray:

    fill = -np.inf if maximum else np.inf
    out = np.full(n_blocks, fill)
    finite = np.isfinite(values)

    (np.maximum if maximum else np.minimum).at(
        out,
        block[finite],
        values[finite],
    )

    return out


def _rolling(
    values: np.ndarray,
    k: int,
    how: str,
) -> np.ndarray:
    """Rolling aggregate over k consecutive blocks, aligned to window start."""

    series = pd.Series(values)

    rolled = getattr(
        series.rolling(k, min_periods=k),
        how,
    )()

    return rolled.shift(-(k - 1)).to_numpy()


def _charging_flags(
    intervals: pd.DataFrame,
    battery: pd.DataFrame | None,
    *,
    min_rise_fraction: float,
) -> np.ndarray:

    flags = np.zeros(len(intervals), dtype=bool)

    if battery is None or battery.empty or intervals.empty:
        return flags

    if not {"timestamp_ms", "battery_level_fraction"} <= set(battery.columns):
        raise ValueError(
            "battery must contain timestamp_ms and battery_level_fraction"
        )

    b = battery[["timestamp_ms", "battery_level_fraction"]].copy()
    b["timestamp_ms"] = pd.to_numeric(b["timestamp_ms"], errors="coerce")
    b["battery_level_fraction"] = pd.to_numeric(
        b["battery_level_fraction"],
        errors="coerce",
    )
    b = b.dropna().sort_values("timestamp_ms", kind="mergesort")

    t = b["timestamp_ms"].to_numpy()
    level = b["battery_level_fraction"].to_numpy()

    for i, (start, end) in enumerate(
        zip(intervals["start_ms"], intervals["end_ms"])
    ):
        inside = level[(t >= start) & (t < end)]

        if inside.size >= 2 and inside[-1] - inside[0] >= min_rise_fraction:
            flags[i] = True

    return flags


def detect_nonwear(
    epochs: pd.DataFrame,
    *,
    window_ms: int = 3_600_000,
    step_ms: int = 900_000,
    sd_threshold_mg: float = 13.0,
    range_threshold_mg: float = 50.0,
    min_axes: int = 2,
    min_window_observed_fraction: float = 0.5,
    battery: pd.DataFrame | None = None,
    charging_min_rise_fraction: float = 0.01,
    minimum_evaluable_ms: float = 3_600_000,
) -> NonwearResult:
    """Classify observed epochs as non-wear (phone not carried).

    Parameters
    ----------
    epochs
        Output of ``acc_preprocess.resample_to_epochs`` (contiguous grid).
    window_ms
        Sliding-window length (van Hees: 60 min).
    step_ms
        Window step (van Hees: 15 min). ``window_ms`` must be a multiple of
        ``step_ms``, and ``step_ms`` a multiple of the epoch length.
    sd_threshold_mg
        Per-axis SD below which an axis can be still (13 mg).
    range_threshold_mg
        Per-axis range below which an axis can be still (50 mg).
    min_axes
        Number of still axes for a still window (2 of 3).
    min_window_observed_fraction
        Minimum observed share of a window for it to be judged at all.
    battery
        Optional screen-stream rows (``timestamp_ms``,
        ``battery_level_fraction``) used only to flag charging intervals.
    charging_min_rise_fraction
        Battery rise within an interval that marks it as charging.
    minimum_evaluable_ms
        Minimum time covered by judged windows for ``acc_nonwear_fraction``
        to be reported.

    Returns
    -------
    NonwearResult
        Epoch mask, interval table and daily summary.
    """

    required = {
        "epoch_start_ms",
        "epoch_end_ms",
        "n_samples",
        "observed",
        *[f"{s}_{a}_g" for a in AXES for s in ("sum", "min", "max")],
        *[f"sumsq_{a}_g2" for a in AXES],
    }

    missing = required - set(epochs.columns)

    if missing:
        raise ValueError(
            "Missing required epoch columns: " + ", ".join(sorted(missing))
        )

    if not 1 <= min_axes <= len(AXES):
        raise ValueError("min_axes must be between 1 and 3")

    if sd_threshold_mg <= 0 or range_threshold_mg <= 0:
        raise ValueError("thresholds must be > 0")

    n_epochs = len(epochs)

    empty_summary = {
        "acc_observed_ms": 0.0,
        "acc_evaluable_ms": 0.0,
        "acc_nonwear_ms": 0.0,
        "acc_wear_ms": 0.0,
        "acc_nonwear_fraction": np.nan,
        "acc_nonwear_fraction_status": FeatureStatus.NO_RAW_DATA.value,
        "acc_nonwear_charging_ms": 0.0,
    }

    if n_epochs == 0:
        return NonwearResult(
            nonwear=np.zeros(0, dtype=bool),
            intervals=pd.DataFrame(columns=INTERVAL_COLUMNS),
            summary=empty_summary,
        )

    start = epochs["epoch_start_ms"].to_numpy(dtype=np.int64)
    end = epochs["epoch_end_ms"].to_numpy(dtype=np.int64)
    epoch_ms = int(end[0] - start[0])

    if step_ms % epoch_ms != 0 or window_ms % step_ms != 0:
        raise ValueError(
            "window_ms must be a multiple of step_ms, and step_ms a multiple "
            f"of the epoch length ({epoch_ms} ms)"
        )

    observed = epochs["observed"].to_numpy(dtype=bool)
    duration = (end - start).astype(float)
    origin = int(start[0])

    block = ((start - origin) // step_ms).astype(np.int64)
    n_blocks = int(block[-1]) + 1
    k = window_ms // step_ms

    n = _block_sums(epochs["n_samples"].to_numpy(dtype=float), block, n_blocks)
    observed_ms = _block_sums(duration * observed, block, n_blocks)
    block_ms = _block_sums(duration, block, n_blocks)

    window_n = _rolling(n, k, "sum")
    window_observed = _rolling(observed_ms, k, "sum")
    window_total = _rolling(block_ms, k, "sum")

    still_axes = np.zeros(n_blocks, dtype=int)

    with np.errstate(invalid="ignore", divide="ignore"):

        for axis in AXES:
            s = _rolling(
                _block_sums(epochs[f"sum_{axis}_g"].to_numpy(), block, n_blocks),
                k,
                "sum",
            )
            ss = _rolling(
                _block_sums(
                    epochs[f"sumsq_{axis}_g2"].to_numpy(),
                    block,
                    n_blocks,
                ),
                k,
                "sum",
            )
            lo = _rolling(
                _block_extreme(
                    epochs[f"min_{axis}_g"].to_numpy(dtype=float),
                    block,
                    n_blocks,
                    maximum=False,
                ),
                k,
                "min",
            )
            hi = _rolling(
                _block_extreme(
                    epochs[f"max_{axis}_g"].to_numpy(dtype=float),
                    block,
                    n_blocks,
                    maximum=True,
                ),
                k,
                "max",
            )

            variance = np.maximum(ss - s * s / window_n, 0.0) / (window_n - 1)
            sd_mg = 1000.0 * np.sqrt(variance)
            range_mg = 1000.0 * (hi - lo)

            still_axes += (
                (sd_mg < sd_threshold_mg)
                & (range_mg < range_threshold_mg)
            ).astype(int)

        judged = (
            np.isfinite(window_total)
            & (window_n >= 2)
            & (window_observed / window_total >= min_window_observed_fraction)
        )

    still_window = judged & (still_axes >= min_axes)

    # Union of still windows over blocks (difference array).
    marks = np.zeros(n_blocks + 1, dtype=int)
    starts_idx = np.flatnonzero(still_window)
    np.add.at(marks, starts_idx, 1)
    np.add.at(marks, np.minimum(starts_idx + k, n_blocks), -1)
    nonwear_block = np.cumsum(marks[:-1]) > 0

    judged_marks = np.zeros(n_blocks + 1, dtype=int)
    judged_idx = np.flatnonzero(judged)
    np.add.at(judged_marks, judged_idx, 1)
    np.add.at(judged_marks, np.minimum(judged_idx + k, n_blocks), -1)
    evaluable_block = np.cumsum(judged_marks[:-1]) > 0

    nonwear = nonwear_block[block] & observed
    evaluable = evaluable_block[block] & observed

    intervals = _merge_intervals(start, end, nonwear_block[block], observed)

    intervals["charging"] = _charging_flags(
        intervals,
        battery,
        min_rise_fraction=charging_min_rise_fraction,
    )

    observed_total = float((duration * observed).sum())
    evaluable_ms = float((duration * evaluable).sum())
    nonwear_ms = float((duration * nonwear).sum())

    fraction = nonwear_ms / evaluable_ms if evaluable_ms > 0 else np.nan

    fraction_status = status_from_counts(
        evaluable_ms,
        minimum_evaluable_ms,
        fraction,
        n_records=int(epochs["n_samples"].sum()),
    )

    summary = {
        "acc_observed_ms": observed_total,
        "acc_evaluable_ms": evaluable_ms,
        "acc_nonwear_ms": nonwear_ms,
        "acc_wear_ms": observed_total - nonwear_ms,
        "acc_nonwear_fraction": gated_value(fraction, fraction_status),
        "acc_nonwear_fraction_status": fraction_status.value,
        "acc_nonwear_charging_ms": float(
            intervals.loc[intervals["charging"], "nonwear_observed_ms"].sum()
        ),
    }

    return NonwearResult(
        nonwear=nonwear,
        intervals=intervals[INTERVAL_COLUMNS],
        summary=summary,
    )


def _merge_intervals(
    start: np.ndarray,
    end: np.ndarray,
    flagged: np.ndarray,
    observed: np.ndarray,
) -> pd.DataFrame:
    """Merge consecutive flagged epochs into intervals."""

    if not flagged.any():
        return pd.DataFrame(
            {c: pd.Series(dtype=float) for c in INTERVAL_COLUMNS[:-1]}
        ).assign(charging=pd.Series(dtype=bool))

    edges = np.diff(np.concatenate([[0], flagged.astype(int), [0]]))
    run_start = np.flatnonzero(edges == 1)
    run_end = np.flatnonzero(edges == -1)

    duration = (end - start).astype(float)
    observed_ms = duration * (flagged & observed)
    cumulative = np.concatenate([[0.0], np.cumsum(observed_ms)])

    out = pd.DataFrame({
        "start_ms": start[run_start],
        "end_ms": end[run_end - 1],
    })

    out["duration_ms"] = (out["end_ms"] - out["start_ms"]).astype(float)
    out["nonwear_observed_ms"] = cumulative[run_end] - cumulative[run_start]

    return out
