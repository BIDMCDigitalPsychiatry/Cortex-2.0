"""Physical-activity fragmentation from minute-level ENMO (Di et al. 2017).

Construct
---------
How activity is accumulated across a day: in few long bouts or many short ones.
Minutes are *active* when ENMO >= ``active_threshold_mg`` and *sedentary*
otherwise. Bouts are maximal runs of consecutive valid minutes in one state.

- ``astp`` (active-to-sedentary transition probability, per minute) =
  number of active bouts / active minutes = 1 / mean active bout length.
- ``satp`` (sedentary-to-active transition probability, per minute) =
  number of sedentary bouts / sedentary minutes.
- ``mean_active_bout_min``, ``active_bout_gini`` (Gini coefficient of active
  bout lengths; needs at least 2 bouts).

Missing data
------------
Invalid minutes (unobserved or non-wear) break bouts; they are never treated
as sedentary. A bout that touches an invalid minute or the window edge is
*censored* (its true length is unknown) and is counted in
``n_censored_bouts``. Censored bouts are still included in the transition
probabilities, as in Di et al.; the count lets analyses check sensitivity.

What this is NOT
----------------
Not intensity-calibrated MVPA: 30 mg is a light-activity ENMO cut-point used
for wrist devices, and a phone in a pocket or bag sees attenuated
acceleration. Treat the threshold as a parameter to calibrate, not a
validated phone cut-point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cortex2_engine.features.status import (
    FeatureStatus,
    gated_value,
    status_from_counts,
)

MINUTE_MS = 60_000

FRAGMENTATION_KEYS = [
    "astp_per_min",
    "satp_per_min",
    "mean_active_bout_min",
    "active_bout_gini",
    "active_minutes",
    "sedentary_minutes",
]


def gini_coefficient(
    values: np.ndarray,
) -> float:
    """Gini coefficient of non-negative values (0 = all equal).

    Parameters
    ----------
    values
        Non-negative values; at least two are needed.

    Returns
    -------
    float
        ``(2 * sum_i i * x_(i)) / (n * sum x) - (n + 1) / n`` with x sorted
        ascending and i = 1..n; NaN for fewer than two values or a zero sum.
    """

    x = np.sort(np.asarray(values, dtype=float))

    if x.size < 2 or x.sum() <= 0:
        return float("nan")

    i = np.arange(1, x.size + 1)

    return float(
        2.0 * (i * x).sum() / (x.size * x.sum())
        - (x.size + 1.0) / x.size
    )


def _bouts(
    state: np.ndarray,
    valid: np.ndarray,
) -> pd.DataFrame:
    """Runs of equal state over valid minutes; invalid minutes break runs."""

    columns = ["active", "length_min", "censored"]

    if not valid.any():
        return pd.DataFrame(columns=columns)

    # Run id changes whenever validity or state changes.
    code = np.where(valid, state.astype(int), -1)
    change = np.concatenate([[True], code[1:] != code[:-1]])

    starts = np.flatnonzero(change)
    ends = np.concatenate([starts[1:], [code.size]])

    run_code = code[starts]
    keep = run_code >= 0

    before_invalid = np.concatenate([[True], code[:-1] < 0])
    after_invalid = np.concatenate([code[1:] < 0, [True]])

    out = pd.DataFrame({
        "active": run_code[keep] == 1,
        "length_min": (ends - starts)[keep].astype(float),
        "censored": (
            before_invalid[starts] | after_invalid[ends - 1]
        )[keep],
    })

    return out


def activity_fragmentation(
    epochs_1min: pd.DataFrame,
    *,
    active_threshold_mg: float = 30.0,
    min_valid_minutes: int = 600,
) -> dict:
    """Daily activity fragmentation from 1-minute epochs.

    Parameters
    ----------
    epochs_1min
        Contiguous 1-minute epochs with ``enmo_mg`` and ``valid`` (output of
        ``acc_preprocess.aggregate_epochs`` with non-wear excluded).
    active_threshold_mg
        ENMO at or above which a minute is active.
    min_valid_minutes
        Minimum valid (observed, worn) minutes for any value to be reported
        (default 600 = 10 h, the usual accelerometry wear-day criterion).

    Returns
    -------
    dict
        ``FRAGMENTATION_KEYS`` with ``_status`` each, plus ``valid_minutes``,
        ``n_active_bouts``, ``n_sedentary_bouts`` and ``n_censored_bouts``.

    Notes
    -----
    ``active_minutes == 0`` on an adequately observed day is
    ``observed_zero``; ``astp``, the mean bout and the Gini are then
    ``not_applicable`` (undefined without active bouts).
    """

    missing = {"enmo_mg", "valid"} - set(epochs_1min.columns)

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing))
        )

    if active_threshold_mg < 0 or min_valid_minutes < 0:
        raise ValueError("thresholds must be >= 0")

    enmo = pd.to_numeric(epochs_1min["enmo_mg"], errors="coerce").to_numpy(
        dtype=float,
    )

    valid = (
        epochs_1min["valid"].fillna(False).to_numpy(dtype=bool)
        & np.isfinite(enmo)
    )

    active = valid & (enmo >= active_threshold_mg)

    bouts = _bouts(active, valid)

    valid_minutes = int(valid.sum())
    active_minutes = float(active.sum())
    sedentary_minutes = float(valid_minutes - active_minutes)

    active_lengths = bouts.loc[bouts["active"], "length_min"].to_numpy()
    n_active = int(active_lengths.size)
    n_sedentary = int((~bouts["active"]).sum()) if not bouts.empty else 0

    raw = {
        "astp_per_min": (
            n_active / active_minutes if active_minutes > 0 else np.nan
        ),
        "satp_per_min": (
            n_sedentary / sedentary_minutes
            if sedentary_minutes > 0
            else np.nan
        ),
        "mean_active_bout_min": (
            float(active_lengths.mean()) if n_active else np.nan
        ),
        "active_bout_gini": gini_coefficient(active_lengths),
        "active_minutes": active_minutes,
        "sedentary_minutes": sedentary_minutes,
    }

    observed_ms = valid_minutes * MINUTE_MS
    minimum_ms = min_valid_minutes * MINUTE_MS

    out: dict = {}

    for key in FRAGMENTATION_KEYS:
        status = status_from_counts(
            observed_ms,
            minimum_ms,
            raw[key],
            n_records=0 if epochs_1min.empty else None,
        )

        out[key] = gated_value(raw[key], status)
        out[f"{key}_status"] = status.value

    out["valid_minutes"] = valid_minutes
    out["n_active_bouts"] = n_active
    out["n_sedentary_bouts"] = n_sedentary
    out["n_censored_bouts"] = (
        int(bouts["censored"].sum()) if not bouts.empty else 0
    )

    # A day with no active minutes has a defined zero but no bouts to
    # describe; make the distinction explicit for the bout-based keys.
    if out["active_minutes_status"] == FeatureStatus.OBSERVED_ZERO.value:
        for key in ("astp_per_min", "mean_active_bout_min", "active_bout_gini"):
            out[f"{key}_status"] = FeatureStatus.NOT_APPLICABLE.value

    return out
