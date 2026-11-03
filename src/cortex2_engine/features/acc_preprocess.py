"""Accelerometer preprocessing: unit detection, ENMO and epoch summaries.

Construct
---------
Device acceleration summarised on a regular epoch grid, in units of standard
gravity (g), so that downstream features (non-wear, circadian rhythm,
fragmentation) see one representation regardless of platform.

What this is NOT
----------------
- Not calibration. Auto-calibration against local gravity (van Hees et al.
  2014) needs long still periods and is left to a later module; ENMO here
  assumes the device's own calibration.
- Not gap filling. Epochs without samples stay unobserved
  (``n_samples == 0``); nothing is interpolated.
- Not a wear decision. See ``acc_nonwear``.

Units
-----
mindLAMP exports do not state the unit (Android commonly reports m/s^2, iOS
CoreMotion reports g). ``detect_unit`` decides from the data: a device at rest
or carried measures a vector magnitude near 1 g, i.e. about 9.81 m/s^2. If
gravity has been removed upstream ("user acceleration"), the magnitude is
near 0 and the unit is reported as ``unknown``. ENMO is then not computable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cortex2_engine.time import coerce_timestamp_ms

STANDARD_GRAVITY_M_S2 = 9.80665

UNIT_G = "g"
UNIT_M_S2 = "m_s2"
UNIT_UNKNOWN = "unknown"

AXES = ("x", "y", "z")

EPOCH_COLUMNS = [
    "epoch_start_ms",
    "epoch_end_ms",
    "n_samples",
    "observed",
    "enmo_mg",
    *[f"{stat}_{axis}_g" for axis in AXES for stat in ("mean", "sum")],
    *[f"sumsq_{axis}_g2" for axis in AXES],
    *[f"{stat}_{axis}_g" for axis in AXES for stat in ("min", "max")],
]


def _require_columns(
    df: pd.DataFrame,
    required: set[str],
    *,
    what: str,
) -> None:

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required {what} columns: "
            + ", ".join(sorted(missing))
        )


def vector_magnitude(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Euclidean norm of three axes, element-wise."""

    return np.sqrt(
        np.asarray(x, dtype=float) ** 2
        + np.asarray(y, dtype=float) ** 2
        + np.asarray(z, dtype=float) ** 2
    )


def detect_unit(
    acc: pd.DataFrame,
    *,
    gravity_band_g: tuple[float, float] = (0.7, 1.3),
) -> str:
    """Infer whether accelerometer axes are in g or m/s^2.

    Parameters
    ----------
    acc
        Raw samples with ``x``, ``y``, ``z``.
    gravity_band_g
        Range, in g, that the median vector magnitude must fall in for the
        signal to be read as including gravity.

    Returns
    -------
    str
        ``"g"`` if the median magnitude is within ``gravity_band_g``,
        ``"m_s2"`` if it is within the same band scaled by 9.80665, otherwise
        ``"unknown"`` (including empty input).

    Notes
    -----
    The median, not the mean, is used so that vigorous movement does not
    shift the decision.
    """

    _require_columns(acc, set(AXES), what="accelerometer")

    lo, hi = gravity_band_g

    if not 0 < lo < hi:
        raise ValueError(
            "gravity_band_g must satisfy 0 < low < high"
        )

    magnitude = vector_magnitude(
        pd.to_numeric(acc["x"], errors="coerce"),
        pd.to_numeric(acc["y"], errors="coerce"),
        pd.to_numeric(acc["z"], errors="coerce"),
    )

    magnitude = magnitude[np.isfinite(magnitude)]

    if magnitude.size == 0:
        return UNIT_UNKNOWN

    median = float(np.median(magnitude))

    if lo <= median <= hi:
        return UNIT_G

    if lo * STANDARD_GRAVITY_M_S2 <= median <= hi * STANDARD_GRAVITY_M_S2:
        return UNIT_M_S2

    return UNIT_UNKNOWN


def enmo_g(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Euclidean norm minus one, negative values truncated to zero.

    Parameters
    ----------
    x, y, z
        Axis values in g.

    Returns
    -------
    numpy.ndarray
        ``max(sqrt(x^2 + y^2 + z^2) - 1, 0)`` in g (van Hees et al. 2013).
        NaN where any axis is missing.
    """

    return np.maximum(
        vector_magnitude(x, y, z) - 1.0,
        0.0,
    )


def resample_to_epochs(
    acc: pd.DataFrame,
    *,
    window_start_ms: int,
    window_end_ms: int,
    epoch_ms: int = 1000,
    unit: str | None = None,
    gravity_band_g: tuple[float, float] = (0.7, 1.3),
) -> pd.DataFrame:
    """Summarise raw samples on a regular epoch grid, in g.

    Parameters
    ----------
    acc
        Canonical accelerometer samples (``timestamp_ms``, ``x``, ``y``,
        ``z``). The legacy ``timestamp`` column is accepted with a warning.
    window_start_ms, window_end_ms
        Half-open window. The grid starts at ``window_start_ms``; every epoch
        in the window gets a row, observed or not.
    epoch_ms
        Epoch length.
    unit
        ``"g"``, ``"m_s2"`` or ``None`` to detect with ``detect_unit``.
    gravity_band_g
        Passed to ``detect_unit``.

    Returns
    -------
    pandas.DataFrame
        One row per epoch with ``EPOCH_COLUMNS``. ``observed`` is True when
        the epoch has at least one sample. Axis sums, sums of squares, minima
        and maxima are kept so that windowed SD and range can be computed
        exactly from epochs (``acc_nonwear``). ``enmo_mg`` is the mean
        per-sample ENMO in milli-g. ``df.attrs["unit"]`` records the input
        unit; when it is ``"unknown"`` all g-based columns are NaN.
    """

    window_start_ms = int(window_start_ms)
    window_end_ms = int(window_end_ms)
    epoch_ms = int(epoch_ms)

    if window_end_ms <= window_start_ms:
        raise ValueError(
            "window_end_ms must be greater than window_start_ms"
        )

    if epoch_ms <= 0:
        raise ValueError("epoch_ms must be > 0")

    acc = coerce_timestamp_ms(
        acc,
        caller="resample_to_epochs",
    )

    _require_columns(
        acc,
        {"timestamp_ms", *AXES},
        what="accelerometer",
    )

    n_epochs = -(-(window_end_ms - window_start_ms) // epoch_ms)

    starts = window_start_ms + epoch_ms * np.arange(n_epochs, dtype=np.int64)

    out = pd.DataFrame({
        "epoch_start_ms": starts,
        "epoch_end_ms": np.minimum(starts + epoch_ms, window_end_ms),
    })

    x = pd.DataFrame({
        "timestamp_ms": pd.to_numeric(acc["timestamp_ms"], errors="coerce"),
        **{
            axis: pd.to_numeric(acc[axis], errors="coerce")
            for axis in AXES
        },
    }).dropna()

    x = x[
        (x["timestamp_ms"] >= window_start_ms)
        & (x["timestamp_ms"] < window_end_ms)
    ]

    if unit is None:
        unit = detect_unit(
            x,
            gravity_band_g=gravity_band_g,
        )

    if unit not in {UNIT_G, UNIT_M_S2, UNIT_UNKNOWN}:
        raise ValueError(
            f"unit must be 'g', 'm_s2' or 'unknown', got {unit!r}"
        )

    scale = (
        1.0 / STANDARD_GRAVITY_M_S2
        if unit == UNIT_M_S2
        else 1.0
    )

    epoch_index = (
        (x["timestamp_ms"].to_numpy(dtype=np.int64) - window_start_ms)
        // epoch_ms
    )

    counts = np.bincount(
        epoch_index,
        minlength=n_epochs,
    )

    out["n_samples"] = counts.astype(np.int64)
    out["observed"] = counts > 0

    if unit == UNIT_UNKNOWN:
        for column in EPOCH_COLUMNS:
            if column not in out.columns:
                out[column] = np.nan

        out.attrs["unit"] = unit
        out.attrs["epoch_ms"] = epoch_ms

        return out[EPOCH_COLUMNS]

    values = {
        axis: x[axis].to_numpy(dtype=float) * scale
        for axis in AXES
    }

    with np.errstate(invalid="ignore", divide="ignore"):

        for axis in AXES:
            v = values[axis]

            sums = np.bincount(epoch_index, weights=v, minlength=n_epochs)
            sumsq = np.bincount(epoch_index, weights=v * v, minlength=n_epochs)

            mins = np.full(n_epochs, np.inf)
            maxs = np.full(n_epochs, -np.inf)
            np.minimum.at(mins, epoch_index, v)
            np.maximum.at(maxs, epoch_index, v)

            empty = counts == 0
            mins[empty] = np.nan
            maxs[empty] = np.nan

            out[f"sum_{axis}_g"] = np.where(empty, np.nan, sums)
            out[f"sumsq_{axis}_g2"] = np.where(empty, np.nan, sumsq)
            out[f"mean_{axis}_g"] = sums / counts
            out[f"min_{axis}_g"] = mins
            out[f"max_{axis}_g"] = maxs

        sample_enmo_mg = 1000.0 * enmo_g(
            values["x"],
            values["y"],
            values["z"],
        )

        out["enmo_mg"] = (
            np.bincount(epoch_index, weights=sample_enmo_mg, minlength=n_epochs)
            / counts
        )

    out.attrs["unit"] = unit
    out.attrs["epoch_ms"] = epoch_ms

    return out[EPOCH_COLUMNS]


def aggregate_epochs(
    epochs: pd.DataFrame,
    *,
    epoch_ms: int = 60_000,
    min_observed_fraction: float = 0.5,
    exclude: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Aggregate fine epochs (e.g. 1 s) into coarser epochs (e.g. 1 min).

    Parameters
    ----------
    epochs
        Output of ``resample_to_epochs``.
    epoch_ms
        Target epoch length; must be a multiple of the input epoch length.
    min_observed_fraction
        Minimum fraction of fine epochs (after exclusion) that must be
        observed for the coarse epoch to be ``valid``.
    exclude
        Boolean mask aligned with ``epochs`` marking fine epochs to drop
        before aggregation (typically non-wear).

    Returns
    -------
    pandas.DataFrame
        ``epoch_start_ms``, ``epoch_end_ms``, ``observed_ms`` (observed,
        non-excluded time), ``excluded_ms``, ``observed_fraction``,
        ``enmo_mg`` (sample-weighted mean over observed, non-excluded fine
        epochs) and ``valid``.
    """

    _require_columns(
        epochs,
        {"epoch_start_ms", "epoch_end_ms", "n_samples", "observed", "enmo_mg"},
        what="epoch",
    )

    if not 0.0 <= min_observed_fraction <= 1.0:
        raise ValueError("min_observed_fraction must be in [0, 1]")

    columns = [
        "epoch_start_ms",
        "epoch_end_ms",
        "observed_ms",
        "excluded_ms",
        "observed_fraction",
        "enmo_mg",
        "valid",
    ]

    if epochs.empty:
        return pd.DataFrame(columns=columns)

    start = epochs["epoch_start_ms"].to_numpy(dtype=np.int64)
    end = epochs["epoch_end_ms"].to_numpy(dtype=np.int64)
    fine_ms = int(end[0] - start[0])

    if epoch_ms % fine_ms != 0:
        raise ValueError(
            "epoch_ms must be a multiple of the input epoch length "
            f"({fine_ms} ms)"
        )

    excluded = (
        np.zeros(len(epochs), dtype=bool)
        if exclude is None
        else np.asarray(exclude, dtype=bool)
    )

    if excluded.shape[0] != len(epochs):
        raise ValueError("exclude must align with epochs")

    origin = int(start[0])
    group = (start - origin) // epoch_ms

    duration = (end - start).astype(float)
    observed = epochs["observed"].to_numpy(dtype=bool) & ~excluded
    n = epochs["n_samples"].to_numpy(dtype=float) * observed
    enmo = np.nan_to_num(epochs["enmo_mg"].to_numpy(dtype=float)) * n

    frame = pd.DataFrame({
        "group": group,
        "observed_ms": duration * observed,
        "excluded_ms": duration * excluded,
        "duration_ms": duration,
        "n": n,
        "enmo_weighted": enmo,
    })

    agg = frame.groupby("group", sort=True).sum()

    out = pd.DataFrame({
        "epoch_start_ms": origin + agg.index.to_numpy(dtype=np.int64) * epoch_ms,
    })

    out["epoch_end_ms"] = (
        out["epoch_start_ms"]
        + agg["duration_ms"].to_numpy(dtype=np.int64)
    )

    out["observed_ms"] = agg["observed_ms"].to_numpy()
    out["excluded_ms"] = agg["excluded_ms"].to_numpy()

    with np.errstate(invalid="ignore", divide="ignore"):
        out["observed_fraction"] = (
            agg["observed_ms"].to_numpy() / agg["duration_ms"].to_numpy()
        )

        out["enmo_mg"] = np.where(
            agg["n"].to_numpy() > 0,
            agg["enmo_weighted"].to_numpy() / agg["n"].to_numpy(),
            np.nan,
        )

    out["valid"] = (
        (out["observed_fraction"] >= min_observed_fraction)
        & out["enmo_mg"].notna()
    )

    return out[columns]
